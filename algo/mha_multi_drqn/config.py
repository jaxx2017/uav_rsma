import numpy as np
import torch

DEFAULT_CONFIG = {
    # seed
    'seed':10,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cuda_deterministic': False,
    'cuda_index': 0,

    'o': 'mlp',  # Type of observation encoder
    'c': None,  # Protocol for multi-agent communication
    'share_reward': False,

    # Model parameters
    'n_layers': 2,
    'hidden_size': 256,
    'n_heads': 2,

    # Basic training hyperparameters
    'lr': 5e-4,  # Learning rate
    'tau': 0.01,  # Target soft update rate
    'gamma': 0.99,  # Discount factor
    'discrete_action': False,  # whether is discrete action
    'polyak': 0.995,  # Interpolation factor in polyak averaging for target network
    'batch_size': 128,  # Minibatch size for SGD
    'replay_size': int(1e4),  # Capacity of replay buffer
    'decay_steps': int(5e4),  # Number of timesteps for exploration
    'max_seq_len': None,  # seq len
    'mixer': False,
    'anneal_lr': True,
    'double_q': True,
    'fair_service': True,

    'episode_length': 100,
    'steps_per_epoch': 20000,  # Number of timesteps in each epoch
    'epochs': 500,  # Number of epochs to run

    'update_after': 10000,  # Number of drqn_env interactions to collect before starting to do gradient descent updates
    'num_test_episodes': 10,  # Number of episodes in each test
    'save_freq': 10,  # How often (in terms of gap between epochs) to save the current policy and value function

    # enironment parameters
    'range_pos': 400,  # the range of map
    'n_community': 16,
    'n_ubs': 2,  # the number of ubs (agents)
    'n_powers': 10,  # the number of power level
    'n_moves': 16,  # the number of move directions
    'n_gts': 12,  # the number of gts
    'n_eve': 16,  # the number of eve
    'r_cov': 50,  # the number of communication distance
    'velocity_bound': 20,  # the velocity of uav
    'jamming_power_bound': 15,  # jamming power of UAV (dbm)
    'r_sense': np.inf,  # the number of sense distance
    'K_richan': -40,  # richan K dB
    'std': 0.1,  # channel estimate error
    'output_dir': None  # logs record
}

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu')