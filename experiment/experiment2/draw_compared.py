from algo.mha_multi_drqn.utils import *
import pickle

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector, zoomed_inset_axes

def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

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


def plot_trainReturns_cmp(train_returns_ltf, train_returns_nltf, save):
    plt.cla()
    plt.figure(figsize=(7, 6))
    returns_ltf = np.array(train_returns_ltf)
    returns_nltf = np.array(train_returns_nltf)
    returns_ltf = returns_ltf.mean(axis=1)
    returns_nltf = returns_nltf.mean(axis=1)
    # returns_ltf = returns_ltf[:30000]
    # returns_nltf = returns_nltf[:30000]
    # flag = True
    # for i, x in enumerate(returns_ltf):
    #     if x != returns_nltf[i]:
    #         flag = False
    # print(flag)
    smoothed_data_ltf = exp_weight_moving_average(data=returns_ltf, beta=0.99)  # 平滑度
    source_data_ltf = returns_ltf
    smoothed_data_nltf = exp_weight_moving_average(data=returns_nltf, beta=0.99)  # 平滑度
    source_data_nltf = returns_nltf
    indices = np.arange(0, len(returns_ltf))
    indices = np.arange(0, 100000)
    smoothed_data_ltf = smoothed_data_ltf[:100000]
    source_data_ltf = source_data_ltf[:100000]
    smoothed_data_nltf = smoothed_data_nltf[:100000]
    source_data_nltf = source_data_nltf[:100000]
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20000))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4000))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')
    def format_k(x, pos):
        if x == 0:
            return 0
        else:
            return '{:.0f}k'.format(x / 1000)

    formatter = ticker.FuncFormatter(format_k)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.xlim((0, len(returns_ltf)))
    plt.xlim((0, 100000))
    plt.ylim((0, 25))
    plt.xlabel('Episode')
    plt.ylabel('Train Return')
    plt.plot(indices, source_data_nltf, color='#ff4757', alpha=0.3)  # 透明度
    plt.plot(indices, smoothed_data_nltf, color='#ff4757', label='CQF')
    plt.plot(indices, source_data_ltf, color='#f0932b', alpha=0.3, )  # 透明度
    plt.plot(indices, smoothed_data_ltf, color='#f0932b', label='FSF')
    plt.legend(loc='best')

    if save:
        plt.savefig('./pics/trainReturnsComp_4uav.png', dpi=600)
        plt.savefig('./pics/trainReturnsComp_4uav.eps', dpi=600)

    # plt.show()


def plot_testReturns_cmp(test_returns_ltf, test_returns_nltf, save):
    plt.cla()
    plt.figure(figsize=(7, 6))
    returns_ltf = np.array(test_returns_ltf)
    return_ltf_means = returns_ltf.mean(axis=1)
    return_ltf_stds = returns_ltf.std(axis=1)
    returns_nltf = np.array(test_returns_nltf)
    return_nltf_means = returns_nltf.mean(axis=1)
    return_nltf_stds = returns_nltf.std(axis=1)

    # return_means = return_means[:100]
    # return_stds = return_stds[:100]
    # returns_indices = np.arange(1, 101)
    returns_indices = np.arange(1, len(return_ltf_means) + 1)

    plt.plot(returns_indices, return_ltf_means, label='FSF', color='#f0932b')
    plt.fill_between(returns_indices, return_ltf_means - return_ltf_stds, return_ltf_means + return_ltf_stds,
                     color='#f0932b', alpha=0.3)
    plt.plot(returns_indices, return_nltf_means, label='CQF', color='#5f27cd')
    plt.fill_between(returns_indices, return_nltf_means - return_nltf_stds, return_nltf_means + return_nltf_stds,
                     color='#5f27cd', alpha=0.3)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2.5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')
    plt.xlim((0, len(return_nltf_means) + 1))
    plt.ylim((0, 20))
    plt.xlabel('Episode')
    plt.ylabel('Test Return')

    # plt.xticks(np.arange(0, len(return_ltf_means), 50))
    plt.legend()

    if save:
        plt.savefig('./pics/testReturnsComp_4uav.png', dpi=600)
        plt.savefig('./pics/testReturnsComp_4uav.eps', dpi=600)

    plt.show()

def plot_test(fair_index_ltf, fair_index_nltf, save):
    plt.cla()
    fig, ax = plt.subplots()
    plt.figure(figsize=(7, 6))
    fair_index_ltf = np.insert(fair_index_ltf, 0, 1)
    fair_index_nltf = np.insert(fair_index_nltf, 0, 1)
    episode_length = len(fair_index_ltf)
    indices = np.array(range(0, episode_length))
    ax.plot(indices, fair_index_ltf, label='FSF', color='black', linestyle=(0, (5, 5)), marker='x', markersize=5)
    ax.plot(indices, fair_index_nltf, label='CQF', color='#ff4757', marker='o', markersize=2)
    ax.set_xlim(0, len(fair_index_ltf))
    ax.set_ylim((0, 1))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Fairness Index')
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    ax.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')


    ax_insert = zoomed_inset_axes(ax, zoom=4, loc='center')
    ax_insert.plot(indices, fair_index_ltf, color='black', linestyle='--', marker='x', markersize=5)
    ax_insert.plot(indices, fair_index_nltf, color='#ff4757', marker='o', markersize=2)
    ax_insert.set_xlim(39, 50)
    ax_insert.set_ylim(0.9, 0.98)
    mark_inset(ax, ax_insert, loc1a=1, loc1b=4, loc2a=2, loc2b=3, fc="none", ec='#3B3B98', lw=2, linestyle='--')

    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    # plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    # plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    # plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    if save:
        plt.savefig('./pics/fair_idxComp_4uav.png', dpi=600)
        plt.savefig('./pics/fair_idxComp_4uav.eps', dpi=600)

    plt.show()



def plot_fair_index_cmp(fair_index_ltf, fair_index_nltf, save):
    plt.cla()
    plt.figure(figsize=(7, 6))
    fair_index_ltf = np.insert(fair_index_ltf, 0, 1)
    fair_index_nltf = np.insert(fair_index_nltf, 0, 1)
    episode_length = len(fair_index_ltf)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, fair_index_ltf, label='FSF', color='black', linestyle='--', marker='x', markersize=8,
             markevery=5, markerfacecolor='none', markeredgecolor='black')
    plt.plot(indices, fair_index_nltf, label='CQF', color='#ff4757', linestyle='-.', marker='o', markersize=8,
                markevery=5, markerfacecolor='none', markeredgecolor='#ff4757')
    plt.grid(color='#3B3B98', linestyle='--', linewidth=0.5)
    plt.xlim((0, len(fair_index_ltf)))
    plt.ylim((0, 1))
    plt.xlabel("Time Step")
    plt.ylabel("Fairness Index")
    plt.legend(loc="best")
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    if save:
        plt.savefig('./pics/fair_idxComp_4uav.png', dpi=600)
        plt.savefig('./pics/fair_idxComp_4uav.eps', dpi=600)

    plt.show()

def plot_ssr_cmp(ssr_ltf, ssr_nltf, save):
    plt.cla()
    plt.figure(figsize=(7, 6))
    ssr_ltf = np.insert(ssr_ltf, 0, 0)
    ssr_nltf = np.insert(ssr_nltf, 0, 0)
    episode_length = len(ssr_ltf)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, ssr_ltf, label='FSF', color='black', linestyle='--', marker='x', markersize=8,
             markevery=4, markerfacecolor='none', markeredgecolor='black')
    plt.plot(indices, ssr_nltf, label='CQF', color='#ff4757', linestyle='-.', marker='o', markersize=8,
                markevery=4, markerfacecolor='none', markeredgecolor='#ff4757')

    plt.grid(color='#3B3B98', linestyle='--', linewidth=0.5)

    plt.xlim((0, len(ssr_nltf)))
    plt.ylim((0, 2.5))
    plt.xlabel("Time Step")
    plt.ylabel("Secrecy Rate (Mbits)")
    plt.legend(loc="best")

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    if save:
        plt.savefig('./pics/ssrComp_4uav.png', dpi=600)
        plt.savefig('./pics/ssrComp_4uav.eps', dpi=600)

    plt.show()


def plot_throughput_cmp(throughput_ltf, throughput_nltf, save, smooth=False):
    plt.cla()
    plt.figure(figsize=(7, 6))
    throughput_ltf = np.insert(throughput_ltf, 0, 0)
    throughput_nltf = np.insert(throughput_nltf, 0, 0)
    if smooth:
        throughput_ltf = exp_weight_moving_average(throughput_ltf, 0.4)
        throughput_nltf = exp_weight_moving_average(throughput_nltf, 0.4)
    episode_length = len(throughput_ltf)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, throughput_ltf, label='FSF', color='black', linestyle='--', marker='x', markersize=8,
             markevery=5, markerfacecolor='none', markeredgecolor='black')
    plt.plot(indices, throughput_nltf, label='CQF', color='#ff4757', linestyle='-.', marker='o', markersize=8,
                markevery=5, markerfacecolor='none', markeredgecolor='#ff4757')
    plt.grid(color='#3B3B98', linestyle='--', linewidth=0.5)
    plt.xlim((0, len(throughput_ltf)))
    plt.ylim((0, 45))
    plt.xlabel("Time Step")
    plt.ylabel("Throughput (Mbits)")
    plt.legend(loc="best")

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    if save:
        plt.savefig('./pics/ThroughputComp_4uav.png', dpi=600)
        plt.savefig('./pics/ThroughputComp_4uav.eps', dpi=600)

    plt.show()

def plot_reward_cmp(reward_ltf, reward_nltf, save):
    plt.cla()
    plt.figure(figsize=(7, 6))
    reward_ltf = np.array(reward_ltf, np.float32)
    reward_ltf_means = reward_ltf.mean(axis=1)
    reward_ltf_means = np.insert(reward_ltf_means, 0, 0)
    reward_nltf = np.array(reward_nltf, np.float32)
    reward_nltf_means = reward_nltf.mean(axis=1)
    reward_nltf_means = np.insert(reward_nltf_means, 0, 0)
    episode_length = len(reward_ltf_means)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, reward_ltf_means, label='FSF', color='#f0932b')
    plt.plot(indices, reward_nltf_means, label='CQF', color='#ff4757')
    plt.legend(loc="best")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.xlim((0, len(reward_ltf) + 1))
    plt.ylim((0, 0.25))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(4))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')

    if save:
        plt.savefig('./pics/RewardComp_4uav.png', dpi=600)
        plt.savefig('./pics/RewardComp_4uav.eps', dpi=600)

    # plt.show()

def plot_throughput_gt(throughput_gt_nltf, throughput_gt_ltf, init_info, save):
    plt.cla()
    x1 = [0, 1, 2, 3, 4]
    x2 = [5, 6, 7, 8, 9]
    x3 = [10, 11, 12, 13, 14]
    x4 = [15, 16, 17, 18, 19]
    x = x3
    throughput_gt_nltf = throughput_gt_nltf[x]
    throughput_gt_ltf = throughput_gt_ltf[x]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.bar(range(len(throughput_gt_ltf)), throughput_gt_ltf, label='FSF', color='#f0932b')
    ax1.set_ylabel("Throughput (Mbits)")
    ax1.set_xlabel("GT")
    ax1.set_xticks(x1)
    ax1.set_title("FSF")
    ax1.set_xticklabels([11, 12, 13, 14, 15])
    ax2.bar(range(len(throughput_gt_nltf)), throughput_gt_nltf, label='CQF', color='#ff4757')
    ax2.set_xticks(x1)
    ax2.set_xlabel("GT")
    ax2.set_title("CQF")
    ax2.set_xticklabels([11, 12, 13, 14, 15])

    plt.tight_layout(pad=0.5)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(50/5))
    ax1.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    ax1.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')
    ax2.yaxis.set_visible(False)
    # ax2.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    # ax2.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')
    plt.ylim((0, 350))
    if save:
        plt.savefig('./pics/throughput_gt_4uav.png', dpi=600)
        plt.savefig('./pics/throughput_gt_4uav.eps', dpi=600)

    plt.show()


if __name__ == '__main__':
    path_root = '/home/zlj/uav_rsma-master/experiment/experiment2/'
    path_ltf = path_root + 'ltf_4uav'
    path_nltf = path_root + 'nltf_4uav'

    data_ltf = load_var(path_ltf + '_data')
    test_returns_ltf = load_var(path_ltf + '_testReturns')
    train_returns_ltf = load_var(path_ltf + '_trainReturns')
    fair_index_ltf = data_ltf['fair_index']
    ssr_ltf = data_ltf['ssr']
    throughput_ltf = data_ltf['throughput']
    # throughput_gt_ltf = data_ltf['throughput_gt']
    reward_ltf = data_ltf['reward']

    data_nltf = load_var(path_nltf + '_data')
    test_returns_nltf = load_var(path_nltf + '_testReturns')
    train_returns_nltf = load_var(path_nltf + '_trainReturns')
    fair_index_nltf = data_nltf['fair_index']
    ssr_nltf = data_nltf['ssr']
    throughput_nltf = data_nltf['throughput']
    # throughput_gt_nltf = data_nltf['throughput_gt']
    reward_nltf = data_nltf['reward']

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

    save = False   # TODO
    # plot_trainReturns_cmp(train_returns_ltf=train_returns_ltf, train_returns_nltf=train_returns_nltf, save=save)
    # plot_testReturns_cmp(test_returns_ltf=test_returns_ltf, test_returns_nltf=test_returns_nltf, save=save)
    # fair_index
    plot_fair_index_cmp(fair_index_ltf=fair_index_ltf, fair_index_nltf=fair_index_nltf, save=save)
    # ssr
    plot_ssr_cmp(ssr_ltf=ssr_ltf, ssr_nltf=ssr_nltf, save=save)
    # throughput
    plot_throughput_cmp(throughput_ltf=throughput_ltf, throughput_nltf=throughput_nltf, save=save)
    # reward
    # plot_reward_cmp(reward_ltf=reward_ltf, reward_nltf=reward_nltf, save=save)

    # plot_throughput_gt(throughput_gt_nltf, throughput_gt_ltf, data_ltf['init_info'], save)
    # plot_test(fair_index_ltf=fair_index_ltf, fair_index_nltf=fair_index_nltf, save=save)
