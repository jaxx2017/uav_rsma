from algo.mha_multi_drqn.utils import *
import pickle

import matplotlib.pyplot as plt

from experiment.experiment2.main import GeneralMap

import matplotlib.ticker as ticker


def plot_testReturns(returns, expname, save, fair_service):
    plt.cla()
    plt.figure(figsize=(7, 6))
    returns = np.array(returns)
    return_means = returns.mean(axis=1)
    return_stds = returns.std(axis=1)
    # return_means = return_means[:100]
    # return_stds = return_stds[:100]
    # returns_indices = np.arange(1, 101)
    returns_indices = np.arange(1, len(returns) + 1)

    plt.plot(returns_indices, return_means, label='Ruturn', color='#E84393')

    plt.fill_between(returns_indices, return_means - return_stds, return_means + return_stds, color='#E84393', alpha=0.2)

    plt.xlim((0, len(returns_indices)))
    if fair_service:
        plt.ylim((0, 8))
    else:
        plt.ylim((0, 14))

    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    plt.title('Test Returns' + fairstr)
    plt.xlabel('Episode')
    plt.ylabel('Test Return')

    plt.xticks(np.arange(0, len(return_means) + 1, 50))
    plt.legend()
    if save:
        plt.savefig('./pics/{}_testReturns.png'.format(expname), dpi=600)
    # plt.show()


def exp_weight_moving_average(data, beta=0.9):
    smoothed_data = []
    current_value = 0

    for i, x in enumerate(data):
        if i == 0:
            current_value = x
        else:
            current_value = beta * current_value + (1 - beta) * x
        smoothed_data.append(current_value)

    return smoothed_data


def plot_trainReturns(returns, expname, save, fair_service):
    plt.cla()
    plt.figure(figsize=(7, 6))
    returns = np.array(returns)
    returns = returns.mean(axis=1)
    smoothed_data = exp_weight_moving_average(data=returns, beta=0.99)  # beta:平滑因子
    source_data = returns
    indices = np.arange(0, len(returns))

    smoothed_data = smoothed_data[:140000]
    source_data = source_data[:140000]
    indices = np.arange(0, 140000)

    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20000))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4000))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    def format_k(x, pos):
        if x == 0:
            return 0
        else:
            return '{:.0f}k'.format(x / 1000)

    formatter = ticker.FuncFormatter(format_k)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title('Train Returns' + fairstr)
    plt.xlabel('Episode')
    plt.ylabel('Train Return')
    plt.plot(indices, source_data, color='#E84393', alpha=0.3)
    plt.plot(indices, smoothed_data, color='#E84393')
    plt.xlim((0, len(returns)))
    plt.xlim((0, 140000))
    if fair_service:
        plt.ylim((0, 13))
    else:
        plt.ylim((0, 18))
    if save:
        plt.savefig('./pics/{}_trainReturns.png'.format(expname), dpi=600)
    # plt.show()


def plot_traj(args, init_info, uav_traj, expname, save, fair_service):
    plt.cla()
    uav_init_pos = init_info['uav_init_pos']
    eve_init_pos = init_info['eve_init_pos']
    gts_init_pos = init_info['gts_init_pos']
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis([0, args.range_pos, 0, args.range_pos])
    color = ['#5f27cd', '#3ae374', '#3d3d3d', '#ff4757']

    eve_init_pos_x = eve_init_pos[:, 0]
    eve_init_pos_y = eve_init_pos[:, 1]
    eve_init = ax.scatter(eve_init_pos_x, eve_init_pos_y, marker='X', color='r', s=50)  # s：调节点的大小

    gt_init_pos_x = gts_init_pos[:, 0]
    gt_init_pos_y = gts_init_pos[:, 1]
    gts_init = ax.scatter(gt_init_pos_x, gt_init_pos_y, marker='s', color='y', s=30)

    # plot axis0
    for x in range(0, args.range_pos, int(args.range_pos / 4)):
        if x == 0:
            continue
        ax.plot([0, args.range_pos], [x, x], linestyle='--', color='b', linewidth=2)

    # plot axis1
    for y in range(0, args.range_pos, int(args.range_pos / 4)):
        if y == 0:
            continue
        ax.plot([y, y], [0, args.range_pos], linestyle='--', color='b', linewidth=2)
    uav_pos = uav_init_pos
    for i, traj in enumerate(uav_traj):
        uav_new_pos = traj
        uav0_trajectory, = ax.plot([uav_pos[0][0], uav_new_pos[0][0]], [uav_pos[0][1], uav_new_pos[0][1]],
                                   linestyle='--', color=color[0], linewidth=2.5)
        uav1_trajectory, = ax.plot([uav_pos[1][0], uav_new_pos[1][0]], [uav_pos[1][1], uav_new_pos[1][1]],
                                   linestyle='--', color=color[1], linewidth=2.5)
        uav_pos = uav_new_pos
        if i == 3 and fair_service == True:
            plt.annotate(
                "t = 3",
                xy=(uav_pos[1][0], uav_pos[1][1]),
                xytext=(310, 180),
                arrowprops=dict(
                    facecolor='#3d3d3d',
                    shrink=0.05,
                    width=1,
                    headwidth=6,
                    headlength=10,
                    ),
                fontsize=16,
                color='black',
            )
        if i == 6 and fair_service == True:
            plt.annotate(
                "t = 6",
                xy=(uav_pos[1][0], uav_pos[1][1]),
                xytext=(270, 185),
                arrowprops=dict(
                    facecolor='#3d3d3d',
                    shrink=0.05,
                    width=1,
                    headwidth=6,
                    headlength=10,
                    ),
                fontsize=16,
                color='black',
            )


    # print(uav_traj[3])
    uav_init_pos_x = uav_init_pos[:, 0]
    uav_init_pos_y = uav_init_pos[:, 1]
    uav_init = ax.scatter(uav_init_pos_x, uav_init_pos_y, marker='D', color='#FC427B')
    uav_final_pos_x = uav_pos[:, 0]
    uav_final_pos_y = uav_pos[:, 1]
    uav_final = ax.scatter(uav_final_pos_x, uav_final_pos_y, marker='*', color="#0652dd", s=120)
    ax.legend(handles=[uav_init, eve_init, gts_init,
                       uav0_trajectory, uav1_trajectory,
                       uav_final],
              labels=['Initial Positions', 'Eves', 'GTs',
                      'UAV 1', 'UAV 2',
                      'Final Positions'],
              loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)  # frameon:label的阴影框 ncol: label分为3列
    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    # ax.set_title("Trajectories of UAVs" + fairstr)
    # plt.text(x=32, y=4, s="Cell-1")
    # plt.text(x=132, y=4, s="Community-2")
    # plt.text(x=232, y=4, s="Community-3")
    # plt.text(x=332, y=4, s="Community-4")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # plt.subplots_adjust(right=0.75)
    plt.tight_layout(pad=0.2)
    if save:
        plt.savefig('./pics/{}_traj.png'.format(expname), dpi=600)
        plt.savefig('./pics/{}_traj.eps'.format(expname), dpi=600)
    # plt.show()


def plot_jamming_powers(jamming_powers, expname, save, fair_service):
    plt.cla()
    plt.figure(figsize=(7.5, 6))
    episode_length = len(jamming_powers) + 1
    indices = np.array(range(0, episode_length))
    n_ubs = len(jamming_powers[0])
    jamming_powers = np.array(jamming_powers, dtype=np.float32)
    color = ['#5f27cd', '#3ae374', '#3d3d3d', '#ff4757']
    marker = ['x', 's', '*', '3']
    linestyle = ['-', '--', '-.', ':']
    if fair_service:
        plt.annotate(
            "t = 3",
            xy=(3, np.float32(jamming_powers[2][1])),
            xytext=(10, 0.033),
            arrowprops=dict(
                facecolor='black',
                shrink=0.05,
                width=1,
                headwidth=6,
                headlength=10,
            ),
            fontsize=16,
            color='black',
        )
        plt.annotate(
            "t = 6",
            xy=(6, np.float32(jamming_powers[5][1])),
            xytext=(12, 0.015),
            arrowprops=dict(
                facecolor='black',
                shrink=0.05,
                width=1,
                headwidth=6,
                headlength=10,
            ),
            fontsize=16,
            color='black',
        )

    for k in range(n_ubs):
        uav_jamming_power = jamming_powers[:, k]
        uav_jamming_power = np.insert(uav_jamming_power, 0, 0)
        plt.plot(indices, uav_jamming_power, linestyle=linestyle[k], color=color[k], linewidth=2,
                 label='UAV {}'.format(k + 1), marker=marker[k], markersize=8, markevery=5,
                 markerfacecolor='none', markeredgecolor=color[k])

    plt.legend(loc="best")
    fairif = fair_service
    if fairif:
        fairstr = ' (LTF)'
    else:
        fairstr = ' (NLTF)'
    plt.grid(color='#3B3B98', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
    plt.tick_params(axis='x', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='x', which='minor', length=4, width=2, color='#3B3B98')
    plt.tick_params(axis='y', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='y', which='minor', length=4, width=2, color='#3B3B98')
    # plt.title("Jamming Power of UAVs" + fairstr)
    plt.xlabel("Time Step")
    plt.ylabel("Jamming Power (Watt)")
    plt.xlim((0, len(jamming_powers) + 1))
    plt.ylim((0, 0.035))

    if save:
        plt.savefig('./pics/{}_jamming_power.eps'.format(expname), dpi=600, bbox_inches='tight')
        plt.savefig('./pics/{}_jamming_power.png'.format(expname), dpi=600)

    plt.show()


def plot_ssr(ssr, expname, save, fair_service, smooth=False):
    plt.cla()
    plt.figure(figsize=(7, 6))
    if smooth:
        ssr = exp_weight_moving_average(data=ssr, beta=0.3)  # beta:平滑因子
    episode_length = len(ssr)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, ssr, label='Secrecy Rate', color='#E84393')
    plt.legend(loc="best")
    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    plt.title("Secrecy Rate" + fairstr)
    plt.xlabel("Time Step")
    plt.ylabel("Secrecy Rate (Mbits)")
    plt.xlim((0, len(ssr) + 1))
    plt.ylim((0, 2))

    if save:
        plt.savefig('./pics/{}_ssr.png'.format(expname), dpi=600)

    # plt.show()

def plot_fair_idx(fair_idx, expname, save, fair_service, smooth=False):
    plt.cla()
    plt.figure(figsize=(7, 6))
    if smooth:
        fair_idx = exp_weight_moving_average(data=fair_idx, beta=0.3)
    episode_length = len(fair_idx)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, fair_idx, label='Fairness', color='#E84393')
    plt.legend(loc="best")
    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    plt.title("Fairness Index" + fairstr)
    plt.xlabel("Time Step")
    plt.ylabel("Fairness Index")
    plt.xlim((0, len(fair_idx) + 1))
    plt.ylim((0, 1))

    if save:
        plt.savefig('./pics/{}_fair_idx.png'.format(expname), dpi=600)

    # plt.show()


def plot_throughput(throughput, expname, save, fair_service, smooth=False):
    plt.cla()
    plt.figure(figsize=(7, 6))
    if smooth:
        throughput = exp_weight_moving_average(data=throughput, beta=0.3)
    episode_length = len(throughput)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, throughput, label='Throughput', color='#E84393')
    plt.legend(loc="best")
    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    plt.title("Throughput" + fairstr)
    plt.xlabel("Time Step")
    plt.ylabel("Throughput (Mbits)")
    plt.xlim((0, len(throughput) + 1))
    plt.ylim((0, 30))

    if save:
        plt.savefig('./pics/{}_Throughput.png'.format(expname), dpi=600)

    # plt.show()

def plot_reward(reward, expname, save, fair_service, smooth=False):
    plt.cla()
    plt.figure(figsize=(7, 6))
    if smooth:
        reward = exp_weight_moving_average(data=reward, beta=0.3)
    episode_length = len(reward)
    indices = np.array(range(0, episode_length))
    reward = np.array(reward, np.float32)
    reward_means = reward.mean(axis=1)
    plt.plot(indices, reward_means, label='Reward', color='#E84393')
    plt.legend(loc="best")
    fairif = fair_service
    if fairif:
        fairstr = ' (FSF)'
    else:
        fairstr = ' (CQF)'

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    plt.title("Reward" + fairstr)
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.xlim((0, len(reward) + 1))
    if fair_service:
        plt.ylim((0, 0.1))
    else:
        plt.ylim((0, 0.16))

    if save:
        plt.savefig('./pics/{}_Reward.png'.format(expname), dpi=600)

    # plt.show()

if __name__ == '__main__':
    fair_service = True  # TODO
    save = True  # TODO
    path = '/home/zlj/uav_rsma-master/experiment/experiment3/'
    if fair_service:
        path = path + 'ltf_2uav'
    else:
        path = path + 'nltf_2uav'

    # ltf = fsf
    # nltf = cqf

    data = load_var(path + '_data')
    test_returns = load_var(path + '_testReturns')
    train_returns = load_var(path + '_trainReturns')
    expname = data['expname']
    init_info = data['init_info']
    uav_traj = data['uav_traj']
    jamming_powers = data['jamming_powers']
    fair_index = data['fair_index']
    ssr = data['ssr']
    throughput = data['throughput']
    reward = data['reward']
    args = data['args']
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'black'  # 全局字体加粗
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['axes.labelweight'] = 'black'  # 坐标轴标签加粗
    plt.rcParams['axes.titleweight'] = 'black'  # 标题加粗
    plt.rcParams['axes.linewidth'] = 2  # 坐标轴线条加粗
    plt.rcParams['xtick.major.width'] = 2  # X轴主刻度线加粗
    plt.rcParams['ytick.major.width'] = 2  # Y轴主刻度线加粗
    plt.rcParams['xtick.minor.width'] = 1.5  # X轴次刻度线加粗
    plt.rcParams['ytick.minor.width'] = 1.5  # Y轴次刻度线加粗
    plt.rcParams['axes.edgecolor'] = '#3B3B98'  # 坐标轴线颜色


    # plot_testReturns(returns=test_returns, expname=expname, save=save, fair_service=fair_service)
    # plot_trainReturns(returns=train_returns, expname=expname, save=save, fair_service=fair_service)
    plot_jamming_powers(jamming_powers=jamming_powers, expname=expname, save=save, fair_service=fair_service)
    # plot_ssr(ssr=ssr, expname=expname, save=save, fair_service=fair_service)
    # plot_fair_idx(fair_idx=fair_index, expname=expname, save=save, fair_service=fair_service)
    # plot_throughput(throughput=throughput, expname=expname, save=save, fair_service=fair_service)
    # plot_reward(reward=reward, expname=expname, save=save, fair_service=fair_service)
    # plot_traj(args=args, init_info=init_info, uav_traj=uav_traj, expname=expname, save=save, fair_service=fair_service)
