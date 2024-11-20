from mha_drqn_env.channel_model import *

from mha_drqn_env.utils import *


class UbsRsmaEvn:
    unit = 100  # Length of each unit (m)
    h_ubs = 100  # Elevation of UBSs (m) 海拔高度
    p_tx_c_dbm = 35  # Transmit common info power (dbm)
    p_tx_p_dbm = 10  # Transmit private info power (dbm)
    p_forward_dbm = 10  # forward power (dbm)
    p_tx_c = 1e-3 * np.power(10, p_tx_c_dbm / 10)  # Transmit power (w)
    p_tx_p = 1e-3 * np.power(10, p_tx_p_dbm / 10)  # Transmit power (w)
    p_forward = 1e-3 * np.power(10, p_forward_dbm / 10)  # Transmit power (w)
    n0 = 1e-3 * np.power(10, -200 / 10)  # PSD of noise (w/hz) -200dbm/hz
    bw = 180e3  # Bandwidth of channels (Hz)
    fc = 2.4e9  # Central carrier frequency (Hz) 2.4Ghz
    dt = 10  # Length of timesteps (sec)
    scene = 'urban'  # Scene of channel model

    def __init__(self, map, fair_service=True, range_pos=800, episode_length=50, n_ubs=2, n_powers=10, n_moves=8, n_gts=100, n_eve=16,
                 r_cov=200.,
                 r_sense=np.inf, n_community=4, K_richan=-40, jamming_power_bound=30,
                 velocity_bound=50):  # vels是速度 r_cov是通信范围

        self._fair_service = fair_service
        self.community_common_info_rate_t = None
        self.range_pos = range_pos  # 活动范围
        self.episode_length = episode_length
        self.map = map(range_pos=range_pos, n_eve=n_eve, n_gts=n_gts, n_ubs=n_ubs, n_community=n_community)
        self.n_gts = n_gts  # Number of GTs
        self.n_ubs = n_ubs  # Number of UBSs
        self.n_agents = n_ubs  # Number of Agents
        self.n_powers = n_powers
        self.n_moves = n_moves
        self.n_eve = n_eve  # Number of n_eve
        self.r_sense = r_sense  # Range of sense (感知范围内的UAV信息)
        self.r_cov = r_cov  # Range of wireless coverage (m)  覆盖范围
        self.K_richan = K_richan  # 莱斯因子
        self.ATGChannel = AirToGroundChannel(self.scene, self.fc)  # UAV->GT, Eve
        self.GTGChannel = GroundToGroundChannel(self.fc)  # GT->GT, GT->Eve

        self.n_community = n_community  # the number of community
        self.gts_in_community = None

        g_atg_max = self.ATGChannel.estimate_chan_gain(0, self.h_ubs)
        # max achievable rate for Normalization
        self.snr_c_atg_max = self.p_tx_c * self.n_gts * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)  # UAV->GTs common info
        self.snr_c_gtg_max = self.p_forward * (self.n_gts - 1) * np.power(abs(1), 2) / (self.n0 * self.bw)  # GT->GTs common info
        self.snr_p_ubs_max = self.p_tx_p * self.n_gts * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)  # UAV->GTs private info
        self.snr_p_gts_max = self.p_tx_p * 1 * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)  # UAV->GTs private info
        self.achievable_rate_c_ubs_max = self.bw * np.log(1 + self.snr_c_atg_max) * 1e-6
        self.achievable_rate_c_gts_max = self.bw * (np.log(1 + self.snr_c_atg_max) + np.log(1 + self.snr_c_gtg_max)) * 1e-6
        self.achievable_rate_p_ubs_max = self.bw * np.log(1 + self.snr_p_ubs_max) * 1e-6
        self.achievable_rate_p_gts_max = self.bw * np.log(1 + self.snr_p_gts_max) * 1e-6
        self.achievable_rate_gts_max = self.achievable_rate_c_gts_max + self.achievable_rate_p_gts_max
        self.achievable_rate_ubs_max = self.achievable_rate_c_ubs_max + self.achievable_rate_p_ubs_max
        # self.achievable_rate_ubs_max = self.achievable_rate_gts_max * self.n_gts

        self.gts_served_in_community_by_uav = None
        # Variables
        self.t = 0  # Timer
        self.p_jamming_t = np.zeros(self.n_ubs, dtype=np.float32)
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)  # Positions of UBS (m)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)  # Positions of GTs (m)
        self.pos_eve = np.empty((self.n_eve, 2), dtype=np.float32)  # Positions of Eve (m)
        self.g_recv_gts_t = np.zeros(self.n_gts, dtype=np.float32)  # ATG channel gain of GTs(m)
        self.cominfo_rate_gts_t = np.zeros(self.n_gts, dtype=np.float32)  # receive common info
        self.privateinfo_rate_gts_t = np.zeros(self.n_gts, dtype=np.float32)  # receive private info
        self.cominfo_rate_eves_t = np.zeros(self.n_eve, dtype=np.float32)  # eve receive common info
        self.privateinfo_rate_eves_t = np.zeros((self.n_eve, self.n_gts), dtype=np.float32)
        self.d_u2g_level = np.zeros((self.n_ubs, self.n_gts),
                                    dtype=np.float32)  # level Distance between UBSs and GTs (m)
        self.d_u2u = np.zeros((self.n_ubs, self.n_ubs), dtype=np.float32)  # Distance between UBSs and UBSs (m)
        self.d_u2eve_level = np.zeros((self.n_ubs, self.n_eve),
                                      dtype=np.float32)  # level Distance between UBSs and Eves
        self.d_g2g = np.zeros((self.n_gts, self.n_gts), dtype=np.float32)  # Distance between GTs and Gts (m)
        self.adj = np.empty((self.n_ubs, self.n_ubs), dtype=bool)  # Adjacency matrix of agents
        self.rate_gt = np.zeros(self.n_gts, dtype=np.float32)  # Instant data rate of each GT (Mbps)
        self.rate_gt_t = np.zeros(self.n_gts, dtype=np.float32)
        self.rate_per_ubs_t = np.zeros(self.n_ubs, dtype=np.float32)  # Instant data rate offered by each UBS (Mbps)
        self.throughput_k_t = np.zeros(self.n_ubs, dtype=np.float32)  # Total throughput of the entire episode (Mb)
        self.aver_rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)
        self.fair_idx = None
        self.avg_fair_idx_per_episode = None
        self.total_throughput = None
        self.total_throughput_t = None
        self.global_util = None  # Trade-off between fairness and throughput

        self.eve_wiretap_ubs = None  # Ubs that wiretapped by eve
        self.ubs_serv_gts = None  # ubs serviced gts
        self.gt_served_by_uav = np.array([-1 for _ in range(self.n_gts)])
        self.ssr_community_common_info_rate_t = np.zeros(self.n_community, dtype=np.float32)
        self.ssr_community_private_info_rate_t = np.zeros(self.n_community, dtype=np.float32)
        self.ssr_community_t = np.zeros(self.n_community, dtype=np.float32)
        self.ssr_system_rate = 0  # Secrecy Sum Rate of system until TimeSlot T
        self.ssr_system_avg_rate = 0  # average Secrecy Sum Rate of system until TimeSlot T
        self.ssr_ubsk_t = np.zeros(self.n_ubs, dtype=np.float32)
        self.ssr_community_sum_t = 0.0
        self.ep_ret = 0  # Episode return
        self.eve_cominfo_k_t = np.zeros(self.n_eve, dtype=np.float32)
        self.reward = 0
        self.mean_returns = None
        self.reward_scale = 0.8

        self.zeta_Loc = 0.3
        self.zeta_ssr = 0.7
        self.velocity = velocity_bound
        self.n_dirs = self.n_moves
        # possible actions
        move_amounts = np.array([self.velocity]).reshape(-1, 1)
        ang = 2 * np.pi * np.arange(self.n_dirs) / self.n_dirs
        move_dirs = np.stack([np.cos(ang), np.sin(ang)]).T  # 其分别的正余弦
        self.avail_moves = np.concatenate((np.zeros((1, 2)), np.kron(move_amounts, move_dirs)))
        self.avail_jamming_powers = [0]
        self.jamming_power_bound = jamming_power_bound
        for i in range(self.n_powers - 1):
            self.avail_jamming_powers.append(1 * (self.jamming_power_bound / 1) ** (i / (self.n_powers - 2)))
        self.avail_jamming_powers = np.array(self.avail_jamming_powers, dtype=np.float32)
        self.avail_jamming_powers = 1e-3 * np.power(10, self.avail_jamming_powers / 10)  # to w
        self.n_moves = self.avail_moves.shape[0]

        self.ssr_mask = np.empty(self.n_ubs, dtype=bool)

        self.velocity_bound = velocity_bound
        self.ssr_gt_rate = np.zeros(self.n_gts, np.float32)

        self.uav_traj = []
        self.jamming_power_list = []
        self.ssr_list = []
        self.fair_idx_list = []
        self.throughput_list = []

    def reset(self):
        self.uav_traj = []
        self.jamming_power_list = []
        self.ssr_list = []
        self.throughput_list = []
        self.fair_idx_list = []

        """reset data"""
        env_info = self.map.get_map()
        self.avg_fair_idx_per_episode = 0
        self.t = 1
        self.total_throughput_t = 0
        self.total_throughput = 0
        self.ep_ret = np.zeros(self.n_ubs, dtype=np.float32)
        self.rate_gt = np.zeros(self.n_gts, dtype=np.float32)  # accumulate data rate of each GT (Mbps)
        self.ssr_gt_rate = np.zeros(self.n_gts, np.float32)
        self.gts_in_community = env_info['gts_in_community']
        self.pos_ubs = env_info['pos_ubs']
        self.pos_gts = env_info['pos_gts']

        for i in range(self.n_gts):
            for j in range(self.n_gts):
                self.d_g2g[i][j] = np.linalg.norm(self.pos_gts[i] - self.pos_gts[j])

        self.pos_eve = env_info['pos_eve']
        self.reward = 0
        self.mean_returns = 0
        self.reward_scale = 0.1
        self.ssr_system_rate = 0.0
        self.aver_rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)
        self.rate_per_ubs_t = np.zeros(self.n_ubs, dtype=np.float32)
        self.update_distance()  # update distance
        jamming_power = np.array([0 for _ in range(self.n_ubs)])
        self.transmit_data(jamming_power=jamming_power)  # transmit data
        self.sercurity_model()  # cal ssr

        obs = wrapper_obs(self.get_obs())

        state = wrapper_state(self.get_state())

        init_info = dict(range_pos=self.range_pos,
                         uav_init_pos=self.pos_ubs,
                         eve_init_pos=self.pos_eve,
                         gts_init_pos=self.pos_gts)

        return obs, state, init_info

    def get_env_info(self):
        obs = self.get_obs_size()
        gt_features_dim = obs['gt'][1]
        other_features_dim = obs['agent'] + np.prod(obs['ubs'])
        # obs_shape = obs['agent'] + np.prod(obs['gt']) + np.prod(obs['ubs'])
        env_info = dict(gt_features_dim=gt_features_dim,
                        other_features_dim=other_features_dim,
                        state_shape=self.get_state_size(),
                        n_moves=self.n_moves,
                        n_powers=self.n_powers,
                        n_agents=self.n_agents,
                        episode_limit=self.episode_length)

        return env_info

    def step(self, actions):
        self.t = self.t + 1  # One timeslot
        # Update UBS positions from actions.
        action_moves = actions['moves']
        action_powers = actions['powers']
        moves = self.avail_moves[np.array(action_moves, dtype=int)]  # Moves of all UBSs
        jamming_powers = self.avail_jamming_powers[np.array(action_powers, dtype=int)]  # Jamming powers of all UBSs
        self.pos_ubs = np.clip(self.pos_ubs + moves,
                               a_min=0,
                               a_max=self.range_pos)

        self.update_distance()  # update distance
        self.transmit_data(jamming_power=jamming_powers)  # UBSs provide wireless service for GTs at the new positions
        self.sercurity_model()  # cal ssr
        reward = self.get_reward(self.reward_scale)
        self.ep_ret = self.ep_ret + reward
        self.mean_returns = self.mean_returns + reward.mean()
        self.avg_fair_idx_per_episode = self.avg_fair_idx_per_episode + self.fair_idx
        done = self.get_terminate()
        info = dict(EpRet=self.ep_ret,
                    EpLen=self.t,
                    mean_returns=self.mean_returns,  # episode mean returns
                    total_throughput=self.total_throughput,  # episode total throughput
                    Ssr_Sys=self.ssr_system_rate,  # the ssr of system
                    global_util=self.global_util,
                    avg_fair_idx_per_episode=self.avg_fair_idx_per_episode)  # fair idx of system
        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())
        # Mark whether termination of episode is caused by reaching episode limit.
        info['BadMask'] = True if (self.t == self.episode_length) else False

        self.uav_traj.append(self.pos_ubs)
        self.jamming_power_list.append(jamming_powers)
        self.fair_idx_list.append(self.fair_idx)
        self.ssr_list.append(self.ssr_community_sum_t)
        self.throughput_list.append(self.total_throughput_t)

        return obs, state, reward, done, info

    def get_uav_trajectory(self):
        return self.uav_traj

    def get_jamming_power(self):
        return self.jamming_power_list

    def get_fair_index(self):
        return self.fair_idx_list

    def get_ssr(self):
        return self.ssr_list

    def get_throughput(self):
        return self.throughput_list

    def get_throughput_gt(self):
        return self.rate_gt

    def update_distance(self):
        """Level Distance UAV->GTs"""
        for k in range(self.n_ubs):
            for i in range(self.n_gts):
                self.d_u2g_level[k][i] = np.linalg.norm(self.pos_ubs[k] - self.pos_gts[i])

        """Level Distance UAV->Eves"""
        for k in range(self.n_ubs):
            for e in range(self.n_eve):
                self.d_u2eve_level[k][e] = np.linalg.norm(self.pos_ubs[k] - self.pos_eve[e])

        """UAV->UAV"""
        for k in range(self.n_ubs):
            for j in range(self.n_ubs):
                self.d_u2u[k][j] = np.linalg.norm(self.pos_ubs[k] - self.pos_ubs[j])

    def transmit_data(self, jamming_power):
        # print("func transmit_data jamming_power:", jamming_power)
        """UAV->GTs"""
        self.ubs_serv_gts = [[] for _ in range(self.n_ubs)]
        self.gt_served_by_uav = np.array([-1 for _ in range(self.n_gts)])
        for i in range(self.n_gts):  # each gts is served by UAV with minium distance
            ubs_k = np.argmin(self.d_u2g_level[:, i])
            if self.d_u2g_level[ubs_k][i] <= self.r_cov:
                self.ubs_serv_gts[ubs_k].append(i)
                self.gt_served_by_uav[i] = ubs_k
        # stage 1: UAV->GTs transmit common information and private information
        self.g_recv_gts_t = np.zeros(self.n_gts, dtype=np.float32)
        # cal UAV->GTs channel gain
        for i in range(self.n_gts):
            ubs_k = self.gt_served_by_uav[i]
            if ubs_k != -1:
                g = self.ATGChannel.estimate_chan_gain(self.d_u2g_level[ubs_k][i], self.h_ubs, self.K_richan)
                self.g_recv_gts_t[i] = np.power(abs(g), 2)

        def calGtsInterUavCellNoise(gt, uav):
            interCommonInfoNoise = 0
            interPrivateInfoNoise = 0
            for k in range(self.n_ubs):
                if self.d_u2g_level[k][gt] <= self.r_cov and len(self.ubs_serv_gts[k]) != 0 and (k != uav):
                    g = self.ATGChannel.estimate_chan_gain(self.d_u2g_level[k][gt], self.h_ubs, self.K_richan)
                    g = np.power(abs(g), 2)
                    interPrivateInfoNoise = interPrivateInfoNoise + g * self.p_tx_p * len(
                        self.ubs_serv_gts[k])  # 其他无人机传输的私密信息对他来说也是影响
                    interCommonInfoNoise = interCommonInfoNoise + g * self.p_tx_c * len(self.ubs_serv_gts[k])

            return dict(interCommonInfoNoise=interCommonInfoNoise, interPrivateInfoNoise=interPrivateInfoNoise)

        # cal UAV->GTs common information rate
        self.cominfo_rate_gts_t = np.zeros(self.n_gts, dtype=np.float32)
        for i in range(self.n_gts):
            ubs_k = self.gt_served_by_uav[i]
            if ubs_k != -1:
                s = self.g_recv_gts_t[i] * self.p_tx_c * len(self.ubs_serv_gts[ubs_k])
                interCellNoise = calGtsInterUavCellNoise(gt=i, uav=ubs_k)
                n = (self.bw * self.n0 + interCellNoise['interPrivateInfoNoise'])  # noise + other uav noise
                n = n + self.g_recv_gts_t[i] * self.p_tx_p * len(self.ubs_serv_gts[ubs_k])  # community noise
                self.cominfo_rate_gts_t[i] = self.bw * np.log2(
                    1 + s / n) * 1e-6  # Achievable common info rate of each GT (Mbps)

        # cal UAV->GTs private information achievable rate
        self.privateinfo_rate_gts_t = np.zeros(self.n_gts, dtype=np.float32)
        for i in range(self.n_gts):
            ubs_k = self.gt_served_by_uav[i]
            if ubs_k != -1:
                s = self.g_recv_gts_t[i] * self.p_tx_p
                interCellNoise = calGtsInterUavCellNoise(gt=i, uav=ubs_k)
                n = self.bw * self.n0 + interCellNoise['interPrivateInfoNoise']  # guess noise
                for j in self.ubs_serv_gts[ubs_k]:  # treat others private info as noise
                    if j != i:
                        n = n + self.g_recv_gts_t[i] * self.p_tx_p
                self.privateinfo_rate_gts_t[i] = self.bw * np.log2(
                    1 + s / n) * 1e-6  # Achievable private info rate of each GT (Mbps)

        """UAV->Eve"""
        # stage 1: Eves eavesdrop common and private information
        self.eve_wiretap_ubs = [[] for _ in range(self.n_eve)]
        for e in range(self.n_eve):
            for k in range(self.n_ubs):
                if self.d_u2eve_level[k][e] <= self.r_cov and len(
                        self.ubs_serv_gts[k]) != 0:  # in cover range and ubs are transmit
                    self.eve_wiretap_ubs[e].append(k)

        # cal UAV->Eves channel gain
        g_eve = np.zeros(self.n_eve, dtype=np.float32)
        for e in range(self.n_eve):
            for k in self.eve_wiretap_ubs[e]:
                g = self.ATGChannel.estimate_chan_gain(self.d_u2eve_level[k][e], self.h_ubs, self.K_richan)
                g_eve[e] = np.power(abs(g), 2)

        def calEvesInterUavCellNoise(eve, uav):
            interCommonInfoNoise = 0
            interPrivateInfoNoise = 0
            for k in range(self.n_ubs):
                if self.d_u2eve_level[k][eve] <= self.r_cov and len(self.ubs_serv_gts[k]) != 0 and (k != uav):
                    g = self.ATGChannel.estimate_chan_gain(self.d_u2eve_level[k][eve], self.h_ubs, self.K_richan)
                    g_eve = np.power(abs(g), 2)
                    interCommonInfoNoise = interCommonInfoNoise + g_eve * self.p_tx_c * len(self.ubs_serv_gts[k])
                    interPrivateInfoNoise = interPrivateInfoNoise + g_eve * self.p_tx_p * len(self.ubs_serv_gts[k])

            return dict(interCommonInfoNoise=interCommonInfoNoise, interPrivateInfoNoise=interPrivateInfoNoise)

        # cal UAV->Eves common information rate
        self.cominfo_rate_eves_t = np.zeros(self.n_eve, dtype=np.float32)
        for e in range(self.n_eve):
            for k in self.eve_wiretap_ubs[e]:
                s = g_eve[e] * self.p_tx_c * len(self.ubs_serv_gts[k])  # wiretap common info
                interCellNoise = calEvesInterUavCellNoise(eve=e, uav=k)
                n = (self.bw * self.n0 + interCellNoise['interPrivateInfoNoise'])
                n = n + g_eve[e] * self.p_tx_p * len(self.ubs_serv_gts[k])  # for GTs that serviced by current ubs
                self.cominfo_rate_eves_t[e] = max(self.bw * np.log2(
                    1 + s / n) * 1e-6, self.cominfo_rate_eves_t[e])
                # community eve Achievable wiretap common info (Mbps)

        # cal UAV->Eves private information rate
        self.privateinfo_rate_eves_t = np.zeros((self.n_eve, self.n_gts), dtype=np.float32)
        for e in range(self.n_eve):
            for k in self.eve_wiretap_ubs[e]:
                interCellNoise = calEvesInterUavCellNoise(eve=e, uav=k)
                n = (self.bw * self.n0 + self.p_tx_c * g_eve[e] * len(self.ubs_serv_gts[k])
                     + interCellNoise['interPrivateInfoNoise']
                     + interCellNoise['interCommonInfoNoise'])
                for i in self.ubs_serv_gts[k]:
                    # if gts i in community e, every eve only writap pirvate infomation in his community
                    if i in self.gts_in_community[e]:
                        s = g_eve[e] * self.p_tx_p  # wirtap every gt in community private info
                        n2 = g_eve[e] * self.p_tx_p * (len(self.ubs_serv_gts[k]) - 1)
                        self.privateinfo_rate_eves_t[e][i] = self.bw * np.log2(1 + s / (n + n2)) * 1e-6

        """GTs->GTs GTs->Eve"""
        # stage2: GTs with best channel
        self.gts_served_in_community_by_uav = [[] for _ in range(self.n_community)]
        for community, community_gts in enumerate(self.gts_in_community):
            for i in community_gts:
                if self.gt_served_by_uav[i] != -1:
                    self.gts_served_in_community_by_uav[community].append(i)

        for community, community_gts in enumerate(self.gts_served_in_community_by_uav):
            gt_with_bstch, bstch = None, -1
            for i in community_gts:
                if self.g_recv_gts_t[i] >= bstch:
                    gt_with_bstch = i
                    bstch = self.g_recv_gts_t[i]
            if len(community_gts) > 1:
                """GTs->Eve"""
                # stage2: GTs was wiretapped by Eves
                eve = community  # every community has an eve
                d = np.linalg.norm(self.pos_eve[eve] - self.pos_gts[gt_with_bstch])
                g_forward = self.GTGChannel.estimate_chan_gain(d)
                g_forward = np.power(abs(g_forward), 2)
                s = g_forward * self.p_forward * (len(community_gts) - 1)
                n = self.bw * self.n0
                for k in self.eve_wiretap_ubs[eve]:
                    n = n + 10 * jamming_power[k] * g_eve[eve]
                self.cominfo_rate_eves_t[eve] = (self.cominfo_rate_eves_t[eve] +
                                                 self.bw * np.log2(1 + s / n) * 1e-6)
            for i in community_gts:
                """GTs->GTs"""
                if i != gt_with_bstch:
                    d = np.linalg.norm(self.pos_gts[i] - self.pos_gts[gt_with_bstch])
                    g_forward = self.GTGChannel.estimate_chan_gain(d)
                    g_forward = np.power(abs(g_forward), 2)
                    s = g_forward * self.p_forward * (len(community_gts) - 1)
                    n = self.bw * self.n0
                    if self.gt_served_by_uav[i] != -1:
                        ubs_k = self.gt_served_by_uav[i]
                        n = n + jamming_power[ubs_k] * self.g_recv_gts_t[i]  # stage 2: uav jamming power
                    self.cominfo_rate_gts_t[i] = (self.cominfo_rate_gts_t[i] +
                                                  self.bw * np.log2(1 + s / n) * 1e-6)

        self.rate_gt_t = (
                self.cominfo_rate_gts_t + self.privateinfo_rate_gts_t)  # common info + private info for every gts
        self.total_throughput_t = self.rate_gt_t.sum()
        self.total_throughput = self.total_throughput + self.total_throughput_t
        self.aver_rate_per_gt = (self.aver_rate_per_gt * self.t + self.rate_gt_t) / (self.t + 1)
        self.fair_idx = compute_jain_fairness_index(self.aver_rate_per_gt)  # Current fairness index among GTs
        self.global_util = self.fair_idx * self.rate_gt_t.mean()  # Trade-off between fairness and throughput

        self.rate_per_ubs_t = np.zeros(self.n_ubs, dtype=np.float32)
        for k in range(self.n_ubs):
            self.rate_per_ubs_t[k] = np.sum(self.rate_gt_t[self.ubs_serv_gts[k]])
        self.rate_gt = self.rate_gt + self.rate_gt_t

    def sercurity_model(self):
        """Common info rate (min)"""
        # the comminfo rate of the UAV cell is min comminfo rate for decode
        self.community_common_info_rate_t = np.zeros(self.n_community, dtype=np.float32)
        for community, community_gts in enumerate(self.gts_served_in_community_by_uav):
            if len(community_gts) > 0:
                com_gts = np.array(community_gts)
                self.community_common_info_rate_t[community] = np.min(
                    self.cominfo_rate_gts_t[com_gts])

        """Common info Security"""
        # UAV cell k
        for e in range(self.n_eve):
            ssr_community_common_info_rate = float(self.community_common_info_rate_t[e] - self.cominfo_rate_eves_t[e])
            self.ssr_community_common_info_rate_t[e] = max(0.0, ssr_community_common_info_rate)

        """Private info Security"""
        self.ssr_community_private_info_rate_t = np.zeros(self.n_community, dtype=np.float32)
        for e in range(self.n_eve):
            community_private_info_rate_t = 0
            writap_community_private_info_rate_t = 0
            for i in self.gts_served_in_community_by_uav[e]:
                assert not isinstance(e, list)
                assert not isinstance(i, list)
                community_private_info_rate_t = community_private_info_rate_t + self.privateinfo_rate_gts_t[i]
                writap_community_private_info_rate_t = (writap_community_private_info_rate_t +
                                                        self.privateinfo_rate_eves_t[e][i])
            self.ssr_community_private_info_rate_t[e] = max(0, community_private_info_rate_t
                                                            - writap_community_private_info_rate_t)

        self.ssr_community_t = self.community_common_info_rate_t + self.ssr_community_private_info_rate_t
        self.ssr_community_sum_t = np.sum(self.ssr_community_t)
        self.ssr_community_avg_rate = self.ssr_community_sum_t / self.n_community

        self.ssr_system_rate = self.ssr_system_rate + self.ssr_community_sum_t

        self.ssr_system_avg_rate = self.ssr_system_rate / self.t

        self.ssr_ubsk_t = np.zeros(self.n_ubs, dtype=np.float32)
        self.ssr_gt_rate = np.zeros(self.n_gts, dtype=np.float32)
        for community, community_gts in enumerate(self.gts_served_in_community_by_uav):
            e = community
            for i in community_gts:
                for k in range(self.n_ubs):
                    if i in self.ubs_serv_gts[k]:
                        ssr_comminfo = max(0.0, self.cominfo_rate_gts_t[i] - self.cominfo_rate_eves_t[e])
                        ssr_privateinfo = max(0.0, self.privateinfo_rate_gts_t[i] - self.privateinfo_rate_eves_t[e][i])
                        self.ssr_ubsk_t[k] = self.ssr_ubsk_t[k] + (ssr_comminfo + ssr_privateinfo)
                        self.ssr_gt_rate[i] = ssr_comminfo + ssr_privateinfo

    def get_obs(self) -> list:
        return [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> dict:
        """Returns local observation of specified agent as a dict."""
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ubs_feats = np.zeros(self.obs_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        # own feats
        own_feats[0:2] = self.pos_ubs[agent_id] / self.range_pos
        own_feats[2] = self.ssr_ubsk_t[agent_id] / self.achievable_rate_ubs_max

        # UBS features
        other_ubs = [ubs_id for ubs_id in range(self.n_agents) if ubs_id != agent_id]
        for j, ubs_id in enumerate(other_ubs):
            if self.d_u2u[agent_id][ubs_id] <= self.r_sense:
                ubs_feats[j, 0] = 1  # vis flag
                ubs_feats[j, 1:3] = (self.pos_ubs[ubs_id] - self.pos_ubs[agent_id]) / self.range_pos  # relative pos

        # GTs features
        for m in range(self.n_gts):
            if self.d_u2g_level[agent_id][m] <= self.r_cov:
                gt_feats[m, 0] = 1  # vision flag
                gt_feats[m, 1:3] = (self.pos_gts[m] - self.pos_ubs[agent_id]) / self.range_pos  # relative pos
                # gt_feats[m, 3] = self.rate_gt_t[m] / self.achievable_rate_gts_max  # avg rate
                gt_feats[m, 3] = self.ssr_gt_rate[m] / self.achievable_rate_gts_max

        return dict(agent=own_feats, ubs=ubs_feats, gt=gt_feats)

    def get_obs_size(self) -> dict:
        return dict(agent=self.obs_own_feats_size, ubs=self.obs_ubs_feats_size, gt=self.obs_gt_feats_size)

    @property
    def obs_own_feats_size(self) -> int:
        """
        Features of agent itself include:
        - Normalized position (x, y)
        - Normalized Security Sum Rate(SSR)
        """
        o_fs = 2 + 1
        return o_fs

    @property
    def obs_ubs_feats_size(self) -> tuple:
        """
        Observed features of each UBS include
        - Visibility flag 0 or 1
        - Normalized distance (x, y) when visible
        """
        u_fs = 1 + 2
        return self.n_agents - 1, u_fs

    @property
    def obs_gt_feats_size(self) -> tuple:
        """
        - Visibility flag 1
        - Normalized distance (x, y) when visible 2
        - Normalized instance QoS 1
        # - Normalized instance ssr gt rate 1
        """
        gt_fs = 1 + 2 + 1

        return self.n_gts, gt_fs

    def get_state(self) -> np.ndarray:
        """
        Returns features of all UBSs and GTs as global drqn_env state.
        Note that state is only used for centralized training and should be inaccessible during inference.
        """
        ubs_feats = np.zeros(self.state_ubs_feats_size(), dtype=np.float32)
        gt_feats = np.zeros(self.state_gt_feats_size(), dtype=np.float32)

        # Features of UBSs
        ubs_feats[:, 0:2] = self.pos_ubs / self.range_pos
        ubs_feats[:, 2] = self.ssr_ubsk_t / self.achievable_rate_ubs_max

        # Features of GTs
        gt_feats[:, 0:2] = self.pos_gts / self.range_pos
        gt_feats[:, 2] = self.rate_gt_t / self.achievable_rate_gts_max

        return np.concatenate((ubs_feats.flatten(), gt_feats.flatten()))

    def get_state_size(self) -> int:
        return np.prod(self.state_ubs_feats_size()) + np.prod(self.state_gt_feats_size())

    def state_ubs_feats_size(self) -> tuple:
        """
        State of each UBS includes
        - Normalized distance (x, y)
        - Normalized Security Sum Rate(SSR)
        """
        su_fs = 2 + 1

        return self.n_ubs, su_fs

    def state_gt_feats_size(self) -> tuple:
        """
        tate of each GT includes
        - Normalized position (x, y)
        - Normalized QoS
        """
        sg_fs = 2 + 1

        return self.n_gts, sg_fs

    def get_reward(self, reward_scale_rate) -> float:
        # penlty_Loc_mask = np.empty(self.n_agents, dtype=bool)
        # penlty_Qos_mask = np.empty(self.n_gts, dtype=bool)
        # penlty_Loc = 5
        # for k in range(self.n_agents):
        #     penlty_Loc_mask[k] = True if (self.pos_ubs[k][0] < 0 or self.pos_ubs[k][0] > self.range_pos or
        #                                   self.pos_ubs[k][1] < 0 or self.pos_ubs[k][1] > self.range_pos) else False
        # for i in range(self.n_gts):
        #     penlty_Qos_mask[i] = True if (self.rate_gt_t[i] <= self.min_limQos) else False

        # self.ssr_mask[k] = True if (self.comminfo_k_t[k] > self.eve_cominfo_k_t[k]) else False
        # ubs_rewards = self.fair_idx * self.ssr_community_avg_rate * np.ones(self.n_agents, dtype=np.float32)
        if self._fair_service:
            ubs_rewards = self.fair_idx * self.ssr_gt_rate.mean() * np.ones(self.n_agents, dtype=np.float32)
        else:
            ubs_rewards = self.ssr_gt_rate.mean() * np.ones(self.n_agents, dtype=np.float32)
        # ubs_rewards = self.global_util * np.ones(self.n_agents, dtype=np.float32) # OK
        ubs_rewards = reward_scale_rate * ubs_rewards / self.achievable_rate_ubs_max
        idle_ubs_mask = (self.rate_per_ubs_t == 0)  # Indices of UBSs serving no GT
        ubs_rewards = ubs_rewards * (1 - idle_ubs_mask)  # Idle UBSs do not receive rewards.

        # ubs_rewards = (self.zeta_ssr * ubs_rewards - self.zeta_Loc * penlty_Loc_mask * penlty_Loc)

        return ubs_rewards

    def get_terminate(self) -> bool:
        """Determines the end of an episode."""
        return True if (self.t == self.episode_length) else False


if __name__ == '__main__':
    env = UbsRsmaEvn(range_pos=1000, episode_limit=40, n_ubs=2, n_eve=3, r_cov=200.,
                     r_sense=np.inf, K_richan=5, std=0.1)
    env.transmit_data()
