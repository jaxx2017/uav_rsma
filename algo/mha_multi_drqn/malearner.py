from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW

from types import SimpleNamespace as SN

from algo.mha_multi_drqn.utils import *

from algo.mha_multi_drqn.buffer import ReplayBuffer

from algo.mha_multi_drqn.agents.agents import Agents

import random

from algo.mha_multi_drqn.agents.qmixer import QMixer

import time

class MultiAgentQLearner:
    """Multi-Agent Q learning algorithm"""

    def __init__(self, env_info, args):
        self.args = args
        self.device = args.device
        # Extract drqn_env info
        self.gt_features_dim = env_info['gt_features_dim']
        self.other_features_dim = env_info['other_features_dim']
        self.n_heads = args.n_heads
        self.state_shape = env_info['state_shape']
        self.n_moves = env_info['n_moves']
        self.n_powers = env_info['n_powers']
        self.n_agents = env_info['n_agents']

        self.policy_net = Agents(gt_features_dim=self.gt_features_dim,
                                 num_heads=self.n_heads,
                                 other_features_dim=self.other_features_dim,
                                 move_dim=self.n_moves,
                                 power_dim=self.n_powers,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  # Policy Network
        self.target_net = Agents(gt_features_dim=self.gt_features_dim,
                                 num_heads=self.n_heads,
                                 other_features_dim=self.other_features_dim,
                                 move_dim=self.n_moves,
                                 power_dim=self.n_powers,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  # Target Network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # print(f"policy network: \n{self.policy_net}")
        self.max_seq_len = args.max_seq_len if args.max_seq_len is not None else env_info['episode_limit']
        self.params = list(self.policy_net.parameters())  # Parameters to optimize
        self.mixer = None
        # QMix
        self.mixer = None
        if args.mixer:
            self.mixer = QMixer(self.state_shape, self.n_agents, args).to(self.device)  # QMixer
            self.target_mixer = deepcopy(self.mixer).to(self.device)
            print(f"mixer = \n{self.mixer}")
            self.params += list(self.mixer.parameters())

        self.gamma = args.gamma  # Discount factor
        self.polyak = args.polyak  # Interpolation factor in polyak averaging for target networks
        self.batch_size = args.batch_size  # Mini-batch size for SGD

        self.buffer = ReplayBuffer(args.replay_size, self.max_seq_len)  # Replay buffer
        self.loss_fn = nn.MSELoss()  # Loss function
        self.optimizer = AdamW(self.params, lr=args.lr)  # Optimizer
        self.anneal_lr = args.anneal_lr  # Whether lr annealing is used.
        if self.anneal_lr:
            lr_lambda = lambda epoch: max(0.4, 1 - epoch / 100)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, verbose=True)
        self.double_q = args.double_q

    def init_hidden(self, batch_size=1):
        """Initializes RNN hidden states for all agents."""
        return self.policy_net.init_hidden().expand(self.n_agents * batch_size, -1)

    def take_actions(self, obs, h, eps_thres):
        """Selects actions following epsilon-greedy strategy"""
        obs_other_features = obs[0].to(self.device)
        obs_gt_features = obs[1].to(self.device)
        h = h.to(self.device)
        # gt_features = np.ones((1, 20, 4)).astype(np.float32)
        # other_features = np.ones((1, 24)).astype(np.float32)
        # hidden_state = np.zeros((1, 256)).astype(np.float32)
        # obs_gt_features = torch.from_numpy(gt_features).to(self.device)
        # obs_other_features = torch.from_numpy(other_features).to(self.device)
        # h = torch.from_numpy(hidden_state).to(self.device)
        with torch.no_grad():
            start_time = time.time()
            move_logits, power_logits, h = self.policy_net(obs_gt_features, obs_other_features, h)
            end_time = time.time()
        # if random.random() > eps_thres:
        if random.random() > eps_thres:
            moves = torch.argmax(move_logits, 1)
            powers = torch.argmax(power_logits, 1)
        else:
            moves = torch.randint(self.n_moves, size=(self.n_agents,), dtype=torch.long)
            powers = torch.randint(self.n_powers, size=(self.n_agents,), dtype=torch.long)

        acts = dict(moves=moves.tolist(), powers=powers.tolist())

        return acts, h, end_time - start_time

    def cache(self, obs, h, state, act, rew, next_obs, next_h, next_state, done, bad_mask):
        if self.args.share_reward:
            rew = rew.mean()
        obs_other_features = obs[0]
        obs_gt_features = obs[1]
        next_obs_other_features = next_obs[0]
        next_obs_gt_features = next_obs[1]

        # When done is True due to reaching episode limit, mute it.
        transition = dict(obs_other_features=obs_other_features,
                          obs_gt_features=obs_gt_features,
                          h=h,
                          state=state,
                          act_moves=torch.tensor(act['moves'], dtype=torch.long).unsqueeze(1),
                          act_powers=torch.tensor(act['powers'], dtype=torch.long).unsqueeze(1),
                          rew=torch.tensor(rew, dtype=torch.float32).reshape(1, -1),
                          next_obs_other_features=next_obs_other_features,
                          next_obs_gt_features=next_obs_gt_features,
                          next_h=(1 - done) * next_h, next_state=next_state,
                          done=torch.tensor((1 - bad_mask) * done, dtype=torch.float32).reshape(1, 1))
        self.buffer.push(transition)

    def update(self):
        """Updates parameters of recurrent agents via BPTT."""

        assert len(self.buffer) >= self.batch_size, "Insufficient samples for update."

        samples = self.buffer.sample(self.batch_size)  # List of sequences
        batch = {k: [] for k in self.buffer.scheme}  # Dict holding batch of samples.

        # Construct input sequences.
        for t in range(self.max_seq_len):
            for k in batch.keys():
                x = [samples[i][k][t].to(self.device) for i in range(self.batch_size)]
                batch[k].append(cat(x))
        # Append next obs/h/state of the last timestep.
        for k in {'obs_other_features', 'obs_gt_features', 'h', 'state'}:
            x = [samples[i][k][self.max_seq_len].to(self.device) for i in range(self.batch_size)]
            batch[k].append(cat(x))

        # acts = torch.stack(batch['act']).to(self.device)
        act_moves = torch.stack(batch['act_moves']).to(self.device)
        act_powers = torch.stack(batch['act_powers']).to(self.device)
        rews = torch.stack(batch['rew']).to(self.device)
        dones = torch.stack(batch['done']).to(self.device)
        h, h_targ = batch['h'][0].to(self.device), batch['h'][1].to(self.device)  # Get initial hidden states.

        agent_out_moves = []
        agent_out_powers = []
        target_out_moves = []
        target_out_powers = []
        # agent_out, target_out = [], []
        obs_other_features = [batch['obs_other_features'][t].to(self.device) for t in range(len(batch['obs_other_features']))]
        obs_gt_features = [batch['obs_gt_features'][t].to(self.device) for t in range(len(batch['obs_gt_features']))]
        # obs = [batch['obs'][t].to(self.device) for t in range(len(batch['obs']))]


        for t in range(self.max_seq_len):
            # Policy network predicts the Q(s_{t},a_{t}) at current timestep.
            # move_logits, power_logits, h = self.policy_net(obs[t], h)
            move_logits, power_logits, h = self.policy_net(obs_gt_features[t],
                                                           obs_other_features[t],
                                                           h)
            agent_out_moves.append(move_logits)
            agent_out_powers.append(power_logits)
            # agent_out.append([move_logits, power_logits])
            # Target network predicts Q(s_{t+1}, a_{t+1}).
            with torch.no_grad():
                next_move_logits, next_power_logits, h_targ = self.target_net(obs_gt_features[t + 1],
                                                                              obs_other_features[t + 1],
                                                                              h_targ)
                # target_out.append([next_move_logits, next_power_logits])
                target_out_moves.append(next_move_logits)
                target_out_powers.append(next_power_logits)

        # Let policy network make predictions for next state of the last timestep in the sequence.
        # move_logits, power_logits, h = self.policy_net(obs[self.max_seq_len], h)
        move_logits, power_logits, h = self.policy_net(obs_gt_features[self.max_seq_len],
                                                       obs_other_features[self.max_seq_len],
                                                       h)

        agent_out_moves.append(move_logits)
        agent_out_powers.append(power_logits)
        # Stack outputs of policy/target networks.
        agent_out_moves, agent_out_powers = torch.stack(agent_out_moves), torch.stack(agent_out_powers)
        target_out_moves, target_out_powers = torch.stack(target_out_moves), torch.stack(target_out_powers)
        # agent_out, target_out = torch.stack(agent_out), torch.stack(target_out)

        # Compute Q_{s_{t}, a_{t}} with policy network.
        q_moves_val = agent_out_moves[:-1].gather(2, act_moves)
        q_powers_val = agent_out_powers[:-1].gather(2, act_powers)
        # qvals = agent_out[:-1].gather(2, acts)
        # Compute V_{s_{t+1}}.
        if not self.double_q:
            next_moves_val = target_out_moves.max(2, keepdim=True)[0]
            next_powers_val = target_out_powers.max(2, keepdim=True)[0]
            # next_vals = target_out.max(2, keepdim=True)[0]
        else:
            next_moves = torch.argmax(agent_out_moves[1:].clone().detach(), dim=2, keepdim=True)
            next_powers = torch.argmax(agent_out_powers[1:].clone().detach(), dim=2, keepdim=True)
            # next_acts = torch.argmax(agent_out[1:].clone().detach(), 2, keepdims=True)
            next_moves_val = target_out_moves.gather(2, next_moves)
            next_powers_val = target_out_powers.gather(2, next_powers)
            # next_vals = target_out.gather(2, next_acts)

        q_moves_val = q_moves_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        q_powers_val = q_powers_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        # qvals = qvals.view(self.max_seq_len, self.batch_size, self.n_agents)
        # next_vals = next_vals.view(self.max_seq_len, self.batch_size, self.n_agents)
        next_moves_val = next_moves_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        next_powers_val = next_powers_val.view(self.max_seq_len, self.batch_size, self.n_agents)

        # Obtain target of update.
        rews, dones = rews.expand_as(next_moves_val), dones.expand_as(next_moves_val)
        target_moves_qvals = rews + self.gamma * (1 - dones) * next_moves_val
        rews, dones = rews.expand_as(next_powers_val), dones.expand_as(next_powers_val)
        target_powers_qvals = rews + self.gamma * (1 - dones) * next_powers_val
        # Compute MSE loss.
        loss_moves = self.loss_fn(q_moves_val, target_moves_qvals)
        loss_powers = self.loss_fn(q_powers_val, target_powers_qvals)
        # loss = self.loss_fn(qvals, target_qvals)
        loss = 0.5 * loss_moves + 0.5 * loss_powers

        # Call one step of gradient descent.
        self.optimizer.zero_grad()
        loss.backward()  # Back propagation
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1)  # Gradient-clipping
        self.optimizer.step()  # Call update.

        # Update the target network via polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.policy_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            if self.mixer is not None:
                for p, p_targ in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        return dict(LossQ=loss.item(),
                    move_QVals=q_moves_val.detach().cpu().numpy(),
                    power_QVals=q_powers_val.detach().cpu().numpy())

    def save_model(self, path, stamp):
        checkpoint = dict()
        checkpoint.update(stamp)
        checkpoint['model_state_dict'] = self.policy_net.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

        print(f"Save checkpoint to {path}.")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.eval()

        print(f"Load checkpoint from {path}.")


if __name__ == "__main__":
    env_info = dict(obs_shape=128, state_shape=128, n_actions=4, n_agents=3)
    args = SN()
    args.state_dim = 128
    args.action_dim = 5
    args.n_agents = 5
    args.gamma = 0.99
    args.polyak = 0.5
    args.batch_size = 32
    args.replay_size = 200
    args.lr = 0.01

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mulAgent = MultiAgentQLearner(env_info=env_info, args=args)

    obs = torch.randn(3, 128)
    h = mulAgent.init_hidden()
    eps_thres = 1
    acts, h = mulAgent.take_actions(obs, h, eps_thres)
    # print(acts, h)
    print(acts)
    print(h.shape)

    print(obs.shape)
