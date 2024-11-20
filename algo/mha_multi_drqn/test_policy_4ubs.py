import time

import torch

import numpy as np

from mha_drqn_env.environment import UbsRsmaEvn

import copy

from algo.mha_multi_drqn.utils import *

from types import SimpleNamespace as SN

from utils import set_rand_seed

import torch

import matplotlib.pyplot as plt

from algo.mha_multi_drqn.malearner import MultiAgentQLearner

from experiment.experiment1.maps import Rang400MapSpecial
from experiment.experiment2.maps import GeneralMap

import shutil


def plot(init_info, uav_traj, jamming_power, args):
    uav_init_pos = init_info['uav_init_pos']
    eve_init_pos = init_info['eve_init_pos']
    gts_init_pos = init_info['gts_init_pos']
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].axis([0, 400, 0, 400])
    for (x, y) in uav_init_pos:
        ubs_init, = ax[0].plot(x, y, marker='o', color='b')
    for (x, y) in eve_init_pos:
        eve_init, = ax[0].plot(x, y, marker='o', color='r', markersize=5)
    for (x, y) in gts_init_pos:
        gts_init, = ax[0].plot(x, y, marker='o', color='y', markersize=5)
    # plot axis0
    for x in range(0, 400, int(100)):
        if x == 0:
            continue
        ax[0].plot([0, 400], [x, x], linestyle='--', color='b')

    # plot axis1
    for y in range(0, 400, int(100)):
        if y == 0:
            continue
        ax[0].plot([y, y], [0, 400], linestyle='--', color='b')
    uav1_pos, uav2_pos, uav3_pos, uav4_pos = uav_init_pos[0], uav_init_pos[1], uav_init_pos[2], uav_init_pos[3]
    # uav1_pos, uav2_pos = uav_init_pos[0], uav_init_pos[1]
    for i, traj in enumerate(uav_traj):
        uav1_pos_new = traj[0]
        uav2_pos_new = traj[1]
        uav3_pos_new = traj[2]
        uav4_pos_new = traj[3]
        # print("plot:", uav1_pos_new, uav2_pos_new)
        ubs1_trajectory, = ax[0].plot([uav1_pos[0], uav1_pos_new[0]],
                                      [uav1_pos[1], uav1_pos_new[1]], linestyle='-', color='#7158e2', linewidth=2)
        ubs2_trajectory, = ax[0].plot([uav2_pos[0], uav2_pos_new[0]],
                                      [uav2_pos[1], uav2_pos_new[1]], linestyle='-', color='#3ae374', linewidth=2)
        ubs3_trajectory, = ax[0].plot([uav3_pos[0], uav3_pos_new[0]],
                                   [uav3_pos[1], uav3_pos_new[1]], linestyle='-', color='#3d3d3d', linewidth=2)
        ubs4_trajectory, = ax[0].plot([uav4_pos[0], uav4_pos_new[0]],
                                   [uav4_pos[1], uav4_pos_new[1]], linestyle='-', color='#ffb8b8', linewidth=2)
        uav1_pos = uav1_pos_new
        uav2_pos = uav2_pos_new
        uav3_pos = uav3_pos_new
        uav4_pos = uav4_pos_new

    uav1_final_pos, = ax[0].plot(uav1_pos[0], uav1_pos[1], marker='o', color='#7158e2')
    uav2_final_pos, = ax[0].plot(uav2_pos[0], uav2_pos[1], marker='o', color='#3ae374')
    uav3_final_pos, = ax[0].plot(uav3_pos[0], uav3_pos[1], marker='o', color='#3d3d3d')
    uav4_final_pos, = ax[0].plot(uav4_pos[0], uav4_pos[1], marker='o', color='#ffb8b8')
    ax[0].legend(handles=[ubs_init, eve_init, gts_init,
                       ubs1_trajectory, ubs2_trajectory,
                       ubs3_trajectory, ubs4_trajectory,
                       uav1_final_pos, uav2_final_pos,
                       uav3_final_pos, uav4_final_pos],
              labels=['uav_init_pos', 'eve_pos', 'gts_pos',
                      'uav1_traj', 'uav2_traj',
                      'uav3_traj', 'uav4_traj',
                      'uav1_final_pos', 'uav2_final_pos',
                      'uav3_final_pos', 'uav4_final_pos'],
              loc="center left", bbox_to_anchor=(1, 0.5))
    # ax[0].legend(handles=[ubs_init, eve_init, gts_init,
    #                       ubs1_trajectory, ubs2_trajectory,
    #                       uav1_final_pos, uav2_final_pos],
    #              labels=['uav_init_pos', 'eve_pos', 'gts_pos',
    #                      'uav1_traj', 'uav2_traj',
    #                      'uav1_final_pos', 'uav2_final_pos'],
    #              loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0].set_title("UAV trajectory")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    fig.subplots_adjust(wspace=0.5)

    x = np.array(range(0, args.episode_length - 1))
    uav_jamming_power = np.array(jamming_power, dtype=np.float32)
    uav0_jamming_power = uav_jamming_power[:, 0]
    uav1_jamming_power = uav_jamming_power[:, 1]
    uav2_jamming_power = uav_jamming_power[:, 2]
    uav3_jamming_power = uav_jamming_power[:, 3]
    ax[1].plot(x, uav0_jamming_power, linestyle='-', color='#7158e2', linewidth=2, label='uav0 j_power')
    ax[1].plot(x, uav1_jamming_power, linestyle='-', color='#3ae374', linewidth=2, label='uav1 j_power')
    ax[1].plot(x, uav2_jamming_power, linestyle='-', color='#3d3d3d', linewidth=2, label='uav2 j_power')
    ax[1].plot(x, uav3_jamming_power, linestyle='-', color='#ffb8b8', linewidth=2, label='uav3 j_power')
    ax[1].legend(loc="center left",
                 bbox_to_anchor=(1, 0.5))
    ax[1].set_title("UAV jamming power (Watt)")
    ax[1].set_xlabel("Time steps")
    ax[1].set_ylabel("Jamming power")
    plt.show()


def load_and_run_policy(test_kwargs, model_path, config_path, n_episodes=1):
    import json
    with open(config_path) as user_file:
        file_contents = user_file.read()
    parsed_json = json.loads(file_contents)
    config = list(parsed_json.values())[0]
    args = SN(**config)
    # Set random seeds.
    set_rand_seed(10)
    if args.cuda_deterministic:
        # Sacrifice performance to ensure reproducibility.
        # Refer to https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    env = UbsRsmaEvn(map=test_kwargs['map'], fair_service=test_kwargs['fair_service'],
                     range_pos=args.range_pos,
                     episode_length=args.episode_length,
                     n_ubs=args.n_ubs,
                     n_powers=args.n_powers,
                     n_moves=args.n_moves,
                     n_gts=args.n_gts,
                     n_eve=args.n_eve,
                     r_cov=args.r_cov,
                     r_sense=args.r_sense,
                     K_richan=args.K_richan,
                     n_community=args.n_community,
                     jamming_power_bound=args.jamming_power_bound,
                     velocity_bound=args.velocity_bound)  # Train maddpg environment

    env_info = env.get_env_info()
    learner = MultiAgentQLearner(env_info, args)
    learner.load_model(model_path)
    t = 0
    Reward = []
    for i_episode in range(n_episodes):
        with torch.no_grad():
            d = False
            (o, s, init_info), h = env.reset(), learner.init_hidden()
            while not d:
                a, h2, inference_time = learner.take_actions(o, h, 0.05)
                o2, s2, r, d, info = env.step(a)
                Reward.append(r)
                o, s, h = o2, s2, h2
            if d:
                uav_traj = env.get_uav_trajectory()
                jamming_powers = env.get_jamming_power()
                fair_index = env.get_fair_index()
                ssr = env.get_ssr()
                throughput = env.get_throughput()
                isFair = "ltf" if test_kwargs['fair_service'] else "nltf"
                n_uav = str(args.n_ubs)

                data = dict(expname=isFair + '_' + n_uav + "uav",
                            init_info=init_info,
                            uav_traj=uav_traj,
                            jamming_powers=jamming_powers,
                            fair_index=fair_index,
                            ssr=ssr,
                            throughput=throughput,
                            reward=Reward,
                            args=args
                            )

                print("智能体与环境交互第{}次, "
                      "ep_ret = {}, "
                      "ssr_system_rate = {}, "
                      "total_throughput = {}".format(1, info['EpRet'], info['Ssr_Sys'], info['total_throughput']))

    return data, args


if __name__ == '__main__':
    start = 200  # TODO
    config_path = '/home/zlj/uav_rsma-master/mha_drqn_data/exp22/config.json'  # TODO
    test_ret_path = '/home/zlj/uav_rsma-master/mha_drqn_data/exp22/vars/test_p_ret.pickle'  # TODO
    train_ret_path = '/home/zlj/uav_rsma-master/mha_drqn_data/exp22/vars/ep_ret_list.pickle'  # TODO
    model_path = '/home/zlj/uav_rsma-master/mha_drqn_data/exp22/checkpoints/checkpoint_epoch'  # TODO
    test_kwargs = {"map": GeneralMap,
                   "fair_service": True}  # TODO
    saveif = True  # TODO
    for _ in range(10000):
        model_path2 = model_path + str(start) + '.pt'
        print(model_path2)
        n_episodes = 1
        data, args = load_and_run_policy(test_kwargs=test_kwargs,
                                         model_path=model_path2,
                                         config_path=config_path,
                                         n_episodes=n_episodes)
        if saveif:
            shutil.copyfile(model_path2,
                            '/home/zlj/uav_rsma-master/experiment/experiment5/{}_ckp.pt'.format(data['expname']))
            shutil.copyfile(test_ret_path,
                            '/home/zlj/uav_rsma-master/experiment/experiment5/{}_testReturns.pickle'.format(
                                data['expname']))
            shutil.copy(train_ret_path,
                        '/home/zlj/uav_rsma-master/experiment/experiment5/{}_trainReturns.pickle'.format(
                            data['expname']))
            save_var('/home/zlj/uav_rsma-master/experiment/experiment5/{}_data'.format(data['expname']), data)
            break
        plot(init_info=data['init_info'], uav_traj=data['uav_traj'], jamming_power=data['jamming_powers'], args=args)
        start += 10
