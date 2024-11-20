from mha_drqn_env.environment import UbsRsmaEvn

from algo.mha_multi_drqn.malearner import MultiAgentQLearner

from types import SimpleNamespace as SN

from algo.mha_multi_drqn.config import DEFAULT_CONFIG

import copy

from algo.mha_multi_drqn.utils import *

from tensorboardX import SummaryWriter

import torch

import os

def train(train_kwargs=dict()):
    for i in range(1, 100):
        output_dir = '/home/zlj/uav_rsma-master/mha_drqn_data/exp'
        if not os.path.exists(output_dir + str(i)):
            output_dir = '/home/zlj/uav_rsma-master/mha_drqn_data/exp'
            output_dir = output_dir + str(i)
            os.makedirs(output_dir)
            os.makedirs(output_dir + '/checkpoints')
            os.makedirs(output_dir + '/logs')
            os.makedirs(output_dir + '/vars')
            break

    # Set configuration
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(train_kwargs)
    args = SN(**config)
    args.output_dir = output_dir
    args = check_args_sanity(args)
    save_config(output_dir=args.output_dir, config=args)  #save configuration
    # Set random seeds.
    set_rand_seed(args.seed)
    # Create instances of environment.
    env = UbsRsmaEvn(range_pos=args.range_pos,
                     episode_length=args.episode_length,
                     map=args.map,
                     n_ubs=args.n_ubs,
                     fair_service=args.fair_service,
                     n_powers=args.n_powers,
                     n_moves=args.n_moves,
                     n_gts=args.n_gts,
                     n_eve=args.n_eve,
                     r_cov=args.r_cov,
                     r_sense=args.r_sense,
                     n_community=args.n_community,
                     K_richan=args.K_richan,
                     jamming_power_bound=args.jamming_power_bound,
                     velocity_bound=args.velocity_bound,
                     )  # Train drqn_env
    test_env = UbsRsmaEvn(range_pos=args.range_pos,
                          episode_length=args.episode_length,
                          map=args.map,
                          n_ubs=args.n_ubs,
                          fair_service=args.fair_service,
                          n_powers=args.n_powers,
                          n_moves=args.n_moves,
                          n_gts=args.n_gts,
                          n_eve=args.n_eve,
                          r_cov=args.r_cov,
                          r_sense=args.r_sense,
                          n_community=args.n_community,
                          K_richan=args.K_richan,
                          jamming_power_bound=args.jamming_power_bound,
                          velocity_bound=args.velocity_bound,
                          )  # Test drqn_env
    env_info = env.get_env_info()
    learner = MultiAgentQLearner(env_info, args)
    total_steps = args.steps_per_epoch * args.epochs
    update_after = max(args.update_after, learner.batch_size * learner.max_seq_len)  # Number of steps before updates
    update_every = learner.max_seq_len  # Number of steps between updates
    # Set exploration strategy.
    eps_start, eps_end = 1, 0.05  # Initial/final rate of exploration
    eps_thres = lambda t: max(eps_end, -(eps_start - eps_end) / args.decay_steps * t + eps_start)  # Epsilon scheduler

    test_agents = 0
    test_p_ret = []
    ep_ret_list = []

    def test_agent(test_agents, num_test_episodes):
        with torch.no_grad():
            """Tests the performance of trained agents."""
            returns_mean = []
            returns = np.zeros(args.n_ubs, dtype=np.float32)
            for n in range(args.num_test_episodes):
                reward = 0
                (o, _, init_info), h, d = test_env.reset(), learner.init_hidden(), False  # Reset drqn_env and RNN.
                while not d:  # one episode
                    a, h, inference_time = learner.take_actions(o, h, 0.05)  # Take (quasi) deterministic actions at test time.
                    o, _, _, d, info = test_env.step(a)  # Env step
                returns += info["EpRet"]
                returns_mean.append(info["EpRet"].mean())
            returns /= num_test_episodes
            for agt in range(args.n_ubs):
                writer.add_scalar("evaluate returns/agent{} ep_ret".format(agt), returns[agt], test_agents)

            return returns_mean

    # Start main loop of training.
    episode = 0
    updates = 0
    (o, s, init_info), h = env.reset(), learner.init_hidden()  # Reset drqn_env and RNN hidden states.
    writer = SummaryWriter(log_dir=args.output_dir + '/logs')
    for t in range(total_steps):
        # Select actions following epsilon-greedy strategy.
        a, h2, inference_time = learner.take_actions(o, h, eps_thres(t))
        # a_save.append(a)
        # Call environment step.
        o2, s2, r, d, info = env.step(a)
        # Store experience to replay buffer.
        learner.cache(o, h, s, a, r, o2, h2, s2, d, info.get("BadMask"))
        # Move to next timestep.
        o, s, h = o2, s2, h2
        # Reach the end of an episode.
        if d:
            episode += 1  # On episode completes.
            ep_ret_list.append(info["EpRet"])
            writer.add_scalar("train environment/avg fairness index",
                              info["avg_fair_idx_per_episode"] / args.episode_length, episode)
            writer.add_scalar("train environment/total throughput", info["total_throughput"], episode)
            writer.add_scalar("train returns/mean returns", info["mean_returns"], episode)
            for agt in range(args.n_ubs):
                writer.add_scalar("train returns/agent{} ep_ret".format(agt), info["EpRet"][agt], episode)
            print(
                "智能体与环境交互第{}次, ep_ret = {}, total_throughput={}, average fair_idx = {}, ssr_system_rate = {}".
                format(
                    episode,
                    info['EpRet'],
                    info["total_throughput"],
                    info['avg_fair_idx_per_episode'] / args.episode_length,
                    info['Ssr_Sys']))

            (o, s, init_info), h = env.reset(), learner.init_hidden()  # Reset drqn_env and RNN hidden states.
        if (t >= update_after) and (t % update_every == 0):
            # print("--------------------learner update--------------------")
            updates += 1
            diagnostic = learner.update()
            # End of epoch handling
        if (t + 1) % args.steps_per_epoch == 0:
            epoch = (t + 1) // args.steps_per_epoch
            # Test performance of trained agents.
            returns_mean = test_agent(test_agents, args.num_test_episodes)
            test_p_ret.append(returns_mean)
            test_agents += 1
            # Anneal learning rate.
            if learner.anneal_lr:
                learner.lr_scheduler.step()
            # Save model parameters.
            if (epoch % args.save_freq == 0) or (epoch == args.epochs):
                save_path = args.output_dir + '/checkpoints/checkpoint_epoch{}.pt'.format(epoch)
                learner.save_model(save_path, stamp=dict(epoch=epoch, t=t))
                save_var(var_path=args.output_dir + '/vars/test_p_ret', var=test_p_ret)
                save_var(var_path=args.output_dir + '/vars/ep_ret_list', var=ep_ret_list)

    writer.close()
    print("Complete.")


if __name__ == '__main__':
    import faulthandler

    faulthandler.enable()
    train_kwargs = dict()
    train(train_kwargs)
