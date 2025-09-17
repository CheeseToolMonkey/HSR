import numpy as np
import pdb
import scipy
import scipy.linalg
import scipy.io
import control  # This import can likely be removed if control is not used elsewhere.
import gym
import math
import random

from matplotlib.cbook import flatten
from scipy.special import j0

from AdHoc.config_downlinkconstraint import num_users, satellite_enabled, max_delay ,p
from scipy.stats import bernoulli
from gym import spaces
from gym.utils import seeding
from WirelessNets import *


class LQR_Env(gym.Env):  # Renamed LQR_Env for consistency, although LQR features are removed
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, mu=1,
                 # hsr
                 t=1, v=138, R=3000, d=400, choice=1, A_b=30, A_m=3, f=930000000, link_all=num_users, bandwidth=20e6,
                 alpha=0.2, beta=0.7,
                 HVFT_block_size=500 * 8 * 1024 * 1024, Data_chunk_size=10 * 1024 * 1024,
                 # satellite
                 satellite_enabled=satellite_enabled, leo1_altitude=800e3, leo2_altitude=1200e3,
                 satellite_freq_hz=30e9, satellite1_bandwidth=400e6, satellite2_bandwidth=400e6,
                 c_light=3e8, k_boltzmann=1.38e-23,
                 satellite_noise_figure_db=1.2, satellite_tx_power_init=2.0,  # This now could be system-level TST power
                 satellite_antenna_gain_tx=10 ** (43.3 / 10), satellite_g_over_t=18.5,
                 satellite_subchannel_num=num_users,
                 # wcs (parameters for wireless network, not LQR)
                 T=40, gamma=0.99, pl=2., pp=5., p0=1., num_features=1, scaling=True,
                 ideal_comm=False):

        super(LQR_Env, self).__init__()
        # ------------------ Initialize Satellite Parameters -----------------------
        self.satellite_enabled = satellite_enabled
        if self.satellite_enabled:
            self.leo1_altitude = leo1_altitude  # meters
            self.leo2_altitude = leo2_altitude  # meters
            self.satellite_freq_hz = satellite_freq_hz  # Hz
            self.satellite1_bandwidth = satellite1_bandwidth  # Hz
            self.satellite2_bandwidth = satellite2_bandwidth  # Hz
            self.satellite_noise_figure = satellite_noise_figure_db
            # Satellite Tx power here refers to the TST's transmit power for satellite links,
            # not per-user ground power for satellite assistance.
            self.satellite_tx_power = satellite_tx_power_init
            self.satellite_antenna_gain_tx = satellite_antenna_gain_tx
            self.satellite_g_over_t = satellite_g_over_t
            self.satellite_subchannel_num = satellite_subchannel_num
            self.k_boltzmann = k_boltzmann
            self.c_light = c_light
            T0 = 290  # Kelvin (standard noise temperature)
            self.satellite_noise_temperature = T0 * (
                    10 ** (self.satellite_noise_figure / 10) - 1)

        # Network parameters
        self.hvft_ratio = 0.2
#         self.others_load_min_bits = 1 * 1024 * 1024
#         self.others_load_max_bits = 1 * 1024 * 1024
#         self.hvft_load_mb = 1 * 1024 * 1024
        
        self.others_load_min_bits = 3000 * 8
        self.others_load_max_bits = 3000 * 8
        self.hvft_load_mb = 3000 * 8
        
        self.users = {}

        # HSR related parameters
        self.t = t
        self.v = v
        self.R = R
        self.d = d
        self.A_b = A_b
        self.A_m = A_m
        self.f = f
        self.choice = choice
        self.bandwidth = bandwidth
        self.link_all = link_all
        self.alpha = alpha
        self.beta = beta
        self.factor = 0.5
        self.hvft_link_indices = []

        # Dimensions
        self.constraint_dim = 1
        self.num_users = num_users
        self.num_features = num_features  # This num_features is passed to GNN, should be 3 for observation features
        self.channel_state_dim = num_users ** 2
        # Observation space dimension: channel state + user traffic types + transmission delays + RTT delays + satellite status
        # 将延迟分解为传输延迟和RTT延迟，让智能体更好地理解功率分配的影响
        self.state_dim_dnn = self.channel_state_dim + num_users*p

        self.np_random = []
        self.seed()

        # Control system parameters (REMOVED: p, q, Ao, r, a0, Bo, A, B, W, Wobs, Wobs_channels, fb_gain_ind, fb_gain)
        self.T = T  # Total timesteps
        self.max_pwr_perplant = pp  # Max power per plant (ground)
        self.p0 = p0  # Base power for ground links

        # Wireless network parameters
        self.mu = mu
        self.sigma = 1.
        self.n_transmitting = np.rint(num_users / 3).astype(np.int32)
        self.gamma = gamma
        self.upperbound = upperbound  # Total ground power constraint
        self.pl = pl
        self.L = L
        self.assign = assign

        self.batch_size = 1
        self.H = 0  # Interference matrix
        self.control_actions = []  # This list now holds agent's actions (power)

        # Open AI gym structure: action_space defined in subclasses
        self.action_space = []
        # Observation space defined to be unbounded for flexibility, normalization handled separately
        self.observation_space = spaces.Box(low=-max_delay * np.ones(self.state_dim_dnn),
                                            high=max_delay * np.ones(self.state_dim_dnn))
        self.scaling = scaling

        self.current_state = self.sample(batch_size=1)  # current_state is just channel observations now

        # Training data tracking (REMOVED LQR-specific cost/Lagrangian history)
        self.current_episode = 0
        self.ep_cost_hist = []
        self.constraint_hist = []
        self.constraint_violation = 0
        self.ep_constraint = []
        self.Lagrangian_hist = []
        self.ep_Lagrangian = []
        self.downlink_constraint_dualvar = 0

        self.time_step = 0
        self.downlinkap_dnn = []
        self.downlinkap_gnn = []

        # Delay tolerance parameters
        self.delay_tolerance_params = {
            'T_HVFT_max': 6,
            'T_others_max': 0.5,
            'K_HVFT': 5.0,
            'K_others': 50.0
        }
        self.user_traffic_types = np.zeros(num_users)
        # 分解延迟为传输延迟和RTT延迟，让智能体更好地理解功率分配的影响
        self.transmission_delays = np.zeros(num_users)  # 传输延迟（受功率影响）
        self.rtt_delays = np.zeros(num_users)          # RTT延迟（固定，不受功率影响）
        self.current_link_delays = np.zeros(num_users)  # 总延迟（用于兼容性）
        self.link_satellite_status = np.zeros(num_users)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def normalize_gso(S):
        norm = np.linalg.norm(S, ord=2, axis=None)
        Snorm = S / norm
        return Snorm

    @staticmethod
    def normalize_inputs(inputs):
        input2 = inputs - inputs.mean(axis=1).reshape(-1, 1)
        return input2

    def normalize_obs(self, obs: np.ndarray, mean, var, epsilon=1e-8, clip_obs=10.) -> np.ndarray:
        return np.clip((obs - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs)

    @staticmethod
    def packet_delivery_rate(snr_value):
        pdr = 1 - np.exp(-snr_value)
        pdr = np.nan_to_num(pdr)
        return pdr

    def doppler_shift_effect(self, t, v, d, R):
        c = 3 * 10 ** 8
        med = R ** 2 - d ** 2
        b = math.sqrt(med)
        b -= v * t
        r = math.sqrt(b ** 2 + d ** 2)
        if r < 1e-6:
            cos_theta = 1.0
        else:
            cos_theta = b / r
        doppler_effect_num = self.f * (v * cos_theta) / c
        return doppler_effect_num, r

    def interference_packet_delivery_rate(self, H, actions):
        """
        计算考虑干扰的数据包传输率
        H: 信道矩阵，H[i,j]表示从发送端j到接收端i的信道增益
        actions: 每个发送端的发送功率
        """
        actions_vec = actions[:, None]  # 转换为列向量
        bandwidth = self.bandwidth
        
        # 计算有用信号功率（直接链路）
        H_diag = np.diag(H)  # 直接链路的信道增益
        desired_signal = np.multiply(H_diag, actions)  # 有用信号功率
        
        # 计算多普勒效应
        doppler_effect_coef, _ = self.doppler_shift_effect(self.t, self.v, self.d, self.R)
        
        # 计算ICI（载波间干扰）
        x_values = np.linspace(-1, 1, 1000)
        integral_result = np.trapz((1 - abs(x_values)) * j0(2 * np.pi * doppler_effect_coef * x_values), x_values)
        W_ICI = 1 - integral_result
        W_ICI = max(0.0, W_ICI)  # 确保ICI系数非负
        
        # 计算干扰
        # H[i,j]表示从发送端j到接收端i的信道增益，不需要转置
        H_interference = H - np.diag(np.diag(H))  # 去除对角线元素，保留干扰信道
        
        # 计算总干扰功率：每个接收端收到的其他发送端的干扰之和
        # np.dot(H_interference, actions_vec)计算每个接收端收到的所有干扰信号的和
        interference_power = np.dot(H_interference, actions_vec).flatten()
        
        # 考虑ICI影响的总干扰加噪声功率
        total_interference = interference_power * (1 + W_ICI) + self.sigma ** 2
        total_interference[total_interference < 1e-9] = 1e-9  # 数值稳定性
        
        # 计算SINR
        SINR = desired_signal / total_interference
        
        # 计算PDR（数据包传输率）
        pdr = 1 - np.exp(-SINR)
        pdr = np.nan_to_num(pdr)  # 处理可能的nan值
        
        # 计算信道容量和有效吞吐量
        channel_capacity_rate = bandwidth * np.log2(1 + SINR)
        effective_throughput_rate = channel_capacity_rate * pdr
        throughput = effective_throughput_rate * self.t
        
        return SINR, pdr, throughput

    def _init_link_state(self, link_all):
        link_state = np.ones(link_all, dtype=int)
        num_hvft_links = int(link_all * self.hvft_ratio)
        all_indices = list(range(link_all))
        random.shuffle(all_indices)
        indices_for_hvft = all_indices[:num_hvft_links]
        if indices_for_hvft:
            link_state[indices_for_hvft] = 1
        return link_state

    def hsr_rtt_delay(self, num_users):
        d_min = 0.065
        sigma = 0.0075
        p_spike = 0.018
        p_retransmission = 0.24
        max_spike_delay = 0.2
        RTT_delay = [0] * num_users
        for i in range(num_users):
            d_vol = np.random.normal(loc=0, scale=sigma)
            d_spike = 0.0
            if np.random.rand() < p_spike:
                d_spike += np.random.uniform(0, max_spike_delay)
                if np.random.rand() < p_retransmission:
                    d_spike += np.random.uniform(0, max_spike_delay)
                    if np.random.rand() < p_retransmission:
                        d_spike += np.random.uniform(0, max_spike_delay)

            total_delay = d_min + d_vol + d_spike
            RTT_delay[i] = max(0, total_delay)

        return RTT_delay

    def link_throughput_and_delay(self, throughput_per_link_rate, user_traffic_types):
        if len(throughput_per_link_rate) != len(user_traffic_types):
            raise ValueError("Arrays throughput_per_link_rate and user_traffic_types must have same length.")

        HVFT_block_size = self.hvft_load_mb

        throughput_HVFT_total = 0.0
        throughput_others_total = 0.0

        traffic_loads_per_link = np.zeros(self.num_users)
        deliver_times_per_chunk = np.full(self.num_users, float('inf'))

        for index, traffic_type in enumerate(user_traffic_types):
            current_link_rate = throughput_per_link_rate[index]

            if traffic_type == 1:
                traffic_loads_per_link[index] = HVFT_block_size
            else:
                traffic_loads_per_link[index] = np.random.uniform(self.others_load_min_bits, self.others_load_max_bits)

            if current_link_rate < 1e-9:
                deliver_times_per_chunk[index] = max_delay
            else:
                deliver_times_per_chunk[index] = traffic_loads_per_link[index] / current_link_rate

            if deliver_times_per_chunk[index] <= self.t:
                delivered_data_this_step = traffic_loads_per_link[index]
            else:
                delivered_data_this_step = current_link_rate * self.t

            if traffic_type == 1:
                throughput_HVFT_total += delivered_data_this_step
            else:
                throughput_others_total += delivered_data_this_step

        throughput_all_total = throughput_HVFT_total + throughput_others_total

        return throughput_HVFT_total, throughput_others_total, throughput_all_total, deliver_times_per_chunk, traffic_loads_per_link

    # Modified calculate_satellite_throughput to calculate aggregate system capacity
    def calculate_satellite_throughput(self, satellite_tx_power_system, satellite_bandwidth_system):
        if not self.satellite_enabled:
            return 0.0
        distance = self.leo1_altitude
        fspl = (4 * np.pi * distance * self.satellite_freq_hz / self.c_light) ** 2
        if fspl < 1e-9:
            fspl = 1e-9
        received_power = (satellite_tx_power_system * self.satellite_antenna_gain_tx) / fspl
        noise_power = self.k_boltzmann * self.satellite_noise_temperature * satellite_bandwidth_system

        satellite_snr = received_power / noise_power
        if satellite_snr < 1e-9:
            satellite_snr = 1e-9

        C_m_n_total_capacity = satellite_bandwidth_system * np.log2(1 + satellite_snr)

        return C_m_n_total_capacity

    def calculate_satellite_delays_with_dov(self, satellite_mask, deliver_times_per_chunk, 
                                           rtt_delays_all_links, traffic_loads_per_link):
        final_delays_per_link = np.zeros(self.num_users)
        link_satellite_status = np.zeros(self.num_users, dtype=int)
        
        if not self.satellite_enabled or not np.any(satellite_mask):
            return final_delays_per_link, link_satellite_status
            
        T_trip_n_satellite = 2 * self.leo1_altitude / self.c_light
        
        # 智能体选择的卫星用户
        satellite_users = np.where(satellite_mask)[0]
        
        # DOV选择策略：计算每个卫星用户的优先级
        satellite_priorities = []
        for idx in satellite_users:
            ground_delay = deliver_times_per_chunk[idx] + rtt_delays_all_links[idx]
            traffic_load = traffic_loads_per_link[idx]
            traffic_type = self.user_traffic_types[idx]
            
            # 优先级计算：考虑延迟、流量、业务类型
            if traffic_type == 1:  # HVFT业务
                priority = ground_delay * traffic_load * 2.0  # HVFT优先级更高
            else:  # 普通业务
                priority = ground_delay * traffic_load
            
            satellite_priorities.append((priority, idx))
        
        # 按优先级排序
        satellite_priorities.sort(reverse=True)
        
        # 选择前satellite_subchannel_num个用户（如果超过限制）
        if len(satellite_users) > self.satellite_subchannel_num:
            selected_satellite_users = [idx for _, idx in satellite_priorities[:self.satellite_subchannel_num]]
            # 更新卫星状态：只有被选中的用户才真正使用卫星
            link_satellite_status.fill(0)
            link_satellite_status[selected_satellite_users] = 1
            final_satellite_users = selected_satellite_users
        else:
            # 用户数量在限制内，全部使用卫星
            link_satellite_status[satellite_users] = 1
            final_satellite_users = satellite_users
        
        # 计算等效容量（基于最终选中的卫星用户）
        total_satellite_traffic = np.sum(traffic_loads_per_link[final_satellite_users])
        
        if total_satellite_traffic > 0:
            # 计算卫星等效容量
            satellite_equivalent_capacity = self.calculate_satellite_throughput(
                self.satellite_tx_power, self.satellite1_bandwidth
            )
            
            # 计算等效传输时间
            if satellite_equivalent_capacity > 0:
                equivalent_transmission_time = total_satellite_traffic / satellite_equivalent_capacity
            else:
                equivalent_transmission_time = max_delay
            
            # 为最终选中的卫星用户分配延迟
            for idx in final_satellite_users:
                traffic = traffic_loads_per_link[idx]
                if total_satellite_traffic > 0:
                    user_throughput = satellite_equivalent_capacity * (traffic / total_satellite_traffic)
                else:
                    user_throughput = 1e-9
                
                if user_throughput < 1e-9:
                    sat_delay = max_delay
                else:
                    # 卫星延迟 = 传播延迟 + 等效传输时间
                    sat_delay = T_trip_n_satellite + equivalent_transmission_time
                
                final_delays_per_link[idx] = min(sat_delay, max_delay)
            
            # 对于智能体选择但未被DOV选中的用户，保持地面延迟
            rejected_users = set(satellite_users) - set(final_satellite_users)
            for idx in rejected_users:
                # 这些用户保持地面延迟，不更新final_delays_per_link
                pass
        else:
            # 没有卫星流量
            for idx in final_satellite_users:
                final_delays_per_link[idx] = max_delay
        
        return final_delays_per_link, link_satellite_status

    def Hata_PL(self, choice, f, A_b, A_m, t, v, d, R):
        num, r = self.doppler_shift_effect(t, v, d, R)
        if r <= 0:
            r = 1e-9

        model1 = 5.74 * np.log10(A_b) - 30.42 + 26.16 * np.log10(f) - 13.82 * np.log10(A_b) - 3.2 * (
                np.log10(11.75 * A_m) ** 2) + (
                         44.9 - 6.55 * np.log10(A_b) - 6.72) * np.log10(r)
        model2 = -21.42 + 26.16 * np.log10(f) - 13.82 * np.log10(A_b) - 3.2 * (np.log10(11.75 * A_m) ** 2) + (
                44.9 - 6.55 * np.log10(A_b) - 9.62) * np.log10(r)
        if choice == 1:
            PL_ii = model1
        else:
            PL_ii = model2
        return PL_ii

    def sample(self, batch_size):
        self.H, samples = self.sample_graph()
        return samples

    def sample_graph(self):
        mu = self.mu
        PL = mu * self.Hata_PL(self.choice, self.f, self.A_b, self.A_m, self.t, self.v, self.d, self.R)
        print("PL:",PL)
        if PL <= 0:
            PL = 1e-9
        samples = self.np_random.rayleigh(PL, size=(self.L.shape[0], self.L.shape[1]))
        PP = samples[None, :, :] * self.L
        A = PP[0]
        A[A < 0.001] = 0.0
        
        # 保存原始信道增益（未归一化的）用于记录
        self.H = A.copy()
        
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()
        return A, A_flat

    def sample_graph_uplink(self):
        mu = self.mu
        PL = mu * self.Hata_PL(self.choice, self.f, self.A_b, self.A_m, self.t, self.v, self.d, self.R)
        if PL <= 0:
            PL = 1e-9
        samples = self.np_random.rayleigh(PL, size=(self.L.shape[0], self.L.shape[1]))
        PP = samples[None, :, :] * self.L
        A = PP[0]
        A[A < 0.001] = 0.0
        A = A.T
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()
        return A, A_flat

    def scale_power(self, power_action):
        power_action = np.clip(power_action, -1., 1.)
        power_action += 1.
        power_action /= 2
        power_action *= self.max_pwr_perplant
        return power_action

    def normalize_scale_power(self, power_action):
        power_action = np.clip(power_action, -1., 1.)
        power_action += 1.
        power_action = power_action / (power_action.sum() + 1e-8)
        power_action *= self.upperbound
        return power_action

    def _reset(self):
        channel_obs_flat = self.sample(batch_size=1)
        self.current_state = channel_obs_flat
        self.time_step = 0

        num_hvft_users = int(self.num_users * self.hvft_ratio)
        shuffled_indices = self.np_random.permutation(self.num_users)
        self.user_traffic_types = np.zeros(self.num_users)
        self.user_traffic_types[shuffled_indices[:num_hvft_users]] = 1

        self.current_link_delays = np.array(self.hsr_rtt_delay(self.num_users))

        self.link_satellite_status = np.zeros(self.num_users)

        if self.ep_cost_hist:
            pass

        self.current_episode += 1
        self.cost_hist = []
        self.constraint_hist = []
        self.Lagrangian_hist = []
        self.others_link_indices = []

        agent_obs = self._get_downlink_obs(channel_obs_flat, self.user_traffic_types,
                                           self.transmission_delays, self.rtt_delays, self.link_satellite_status, np.zeros(self.num_users))
        return agent_obs

    # Modified _update_control_states_downlink
    def _update_control_states_downlink(self, actual_ground_tx_power, lambd,
                                        action_penalty):

        self.time_step += 1
        SINR_per_link, pdr_per_link, throughput_per_link_rate = self.interference_packet_delivery_rate(
            self.H, actual_ground_tx_power)

        throughput_HVFT_total, throughput_others_total, throughput_all_total, deliver_times_per_chunk, traffic_loads_per_link \
            = self.link_throughput_and_delay(throughput_per_link_rate, self.user_traffic_types)
        rtt_delays_all_links = np.array(self.hsr_rtt_delay(self.num_users))
        
        # 存储分解的延迟信息，让智能体更好地理解功率分配的影响
        self.transmission_delays = np.copy(deliver_times_per_chunk)  # 传输延迟（受功率影响）
        self.rtt_delays = np.copy(rtt_delays_all_links)             # RTT延迟（固定，不受功率影响）
        
        # 初始化最终延迟数组
        final_delays_per_link = np.zeros(self.num_users)
        self.link_satellite_status = np.zeros(self.num_users, dtype=int)  # Reset status
        
        # 地面用户延迟：传输延迟 + RTT延迟
        # ground_mask = ~satellite_mask
        # ground_total_delays = np.copy(deliver_times_per_chunk[ground_mask]) + rtt_delays_all_links[ground_mask]
        # ground_total_delays = np.minimum(ground_total_delays, max_delay)
        # final_delays_per_link[ground_mask] = ground_total_delays
        
        # # 卫星用户延迟：调用专门的卫星计算函数
        # satellite_delays, satellite_status = self.calculate_satellite_delays_with_dov(
        #     satellite_mask, deliver_times_per_chunk, rtt_delays_all_links, traffic_loads_per_link
        # )
        # final_delays_per_link += satellite_delays
        # self.link_satellite_status = satellite_status

        final_delays_per_link = np.minimum(final_delays_per_link, max_delay)
        self.current_link_delays = final_delays_per_link  # Update for observation space

        # # 卫星资源使用效率（DOV选择策略）
        # satellite_efficiency = 0
        # if np.any(satellite_mask):
        #     satellite_users_count = np.sum(satellite_mask)
        #     final_satellite_users_count = np.sum(self.link_satellite_status)
        #     # 智能体选择的用户数量
        #     if satellite_users_count > 0:
        #         # DOV选择成功率（被选中的用户比例）
        #         dov_success_rate = final_satellite_users_count / satellite_users_count
        #         # 如果智能体选择合理（大部分被DOV接受），给予奖励
        #         if dov_success_rate >= 0.8:
        #             satellite_efficiency = 0.1 * final_satellite_users_count
        #         elif dov_success_rate >= 0.5:
        #             satellite_efficiency = 0.05 * final_satellite_users_count
        #     else:
        #         # 选择不合理，给予轻微惩罚
        #         satellite_efficiency = -0.05 * (satellite_users_count - final_satellite_users_count)
        #         # 额外奖励：HVFT业务优先使用卫星
        #         hvft_satellite_users = np.sum((self.user_traffic_types == 1) & (self.link_satellite_status == 1))
        #         satellite_efficiency += 0.02 * hvft_satellite_users
    
        # 延迟惩罚，归一化
        delay_sum = np.sum(final_delays_per_link)
        delay_mean = np.mean(final_delays_per_link)
        
        # # 功率效率，归一化到[0,1]
        # total_power = np.sum(actual_ground_tx_power)
        # power_efficiency = 1.0 - (total_power / (self.num_users * self.max_pwr_perplant))
        
        # 拉格朗日乘子法控制功耗
        # 只有当功率超出约束时才产生惩罚
        if action_penalty > 0:  # 功率超出约束
            power_penalty = float(np.dot(lambd, action_penalty) / self.upperbound)
        else:  # 功率满足约束
            power_penalty = 0.0  # 无惩罚
       
        # 延迟本身跟吞吐量挂钩，不再计算吞吐量
        reward = -delay_mean -power_penalty*0.05
        
        reward /= 10
        
        # 记录训练历史
        self.cost_hist.append(-reward)

        done = False
        if self.time_step > self.T - 1:
            done = True

        return (final_delays_per_link, SINR_per_link, pdr_per_link, reward,
                done, throughput_HVFT_total, throughput_others_total, throughput_all_total)

    def _get_downlink_obs(self, channel_obs, user_traffic_types, transmission_delays, rtt_delays, link_satellite_status,log2_sinr):
        if channel_obs.ndim > 1:
            channel_obs = channel_obs.flatten()
        if user_traffic_types.ndim > 1:
            user_traffic_types = user_traffic_types.flatten()
        if transmission_delays.ndim > 1:
            transmission_delays = transmission_delays.flatten()
        if rtt_delays.ndim > 1:
            rtt_delays = rtt_delays.flatten()
        if link_satellite_status.ndim > 1:
            link_satellite_status = link_satellite_status.flatten()
        if log2_sinr.ndim > 1:
            log2_sinr = log2_sinr.flatten()

        # 观测空间：信道状态 + 用户业务类型 + 传输延迟 + RTT延迟 + 卫星状态
        # 这样智能体可以分别理解功率分配对传输延迟的影响，以及固定的RTT延迟
        obs = np.hstack(
            (channel_obs, user_traffic_types, transmission_delays, rtt_delays, link_satellite_status,log2_sinr))

        return obs

    def _test_step_uplink(self, states, states_obs, H, action, estimator):

        control_states = states[:, self.channel_state_dim:]  # [None, :]

        # uplink delivery rate
        qq = self.interference_packet_delivery_rate(H, action.flatten())
        qq = np.nan_to_num(qq)
        trials = np.transpose(bernoulli.rvs(qq))

        # updates state estimate
        control_states_obs = np.multiply(trials, control_states)

        # control actions computed at the remote controller
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        # cost / reward -> plant states
        cost_aux = np.multiply(control_states, control_states)

        # cost / reward -> control input
        cost_aux2 = np.multiply(control_actions, control_actions)

        # total reward
        one_step_cost = cost_aux.sum(axis=1)

        return one_step_cost, control_states, control_actions

    def _test_step_downlink(self, control_states, control_actions, H, downlink_action, zerovec):

        # downlink delivery rate
        qq = self.interference_packet_delivery_rate(H, downlink_action)
        qq = np.nan_to_num(qq)
        trials_aux = np.transpose(bernoulli.rvs(qq))
        trials = np.repeat(trials_aux, self.q, axis=0)[:, None]

        control_estimate = np.multiply(trials, control_actions)
        control_states = control_states.transpose()

        # new control states
        control_states = (np.dot(self.A, control_states) + np.dot(self.B, control_estimate) +
                          np.transpose(self.np_random.multivariate_normal(zerovec, self.W, size=self.batch_size)))
        control_states = np.transpose(control_states)
        control_states = np.clip(control_states, -self.max_control_state, self.max_control_state)
        control_states_obs = control_states + self.np_random.multivariate_normal(zerovec, self.Wobs,
                                                                                 size=self.batch_size)
        control_states_obs = np.clip(control_states_obs, -self.max_control_state, self.max_control_state)

        # new channel states
        H, channel_states = self.sample_graph_uplink()

        states = np.hstack((channel_states, control_states.flatten()))
        states_obs = np.hstack((channel_states, control_states_obs.flatten()))

        return states, states_obs, H

    # 注释掉不兼容的LQR test函数，因为它们依赖于LQR特有的属性
    def test_equal_power(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_round_robin(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1,
                         last_idx=0):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_control_aware(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1,
                           last_idx=-1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_wmmse(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1,
                   last_idx=-1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_random_access(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1,
                           last_idx=-1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    # 注释掉不兼容的LQR test函数，因为它们依赖于LQR特有的属性
    def test_mlp_inst_constraint(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                 cost_mtx, batch_size=1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_gnn_inst_constraint(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                 cost_mtx, batch_size=1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_mlp_hor_constraint(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                cost_mtx, batch_size=1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test_gnn_hor_constraint(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                cost_mtx, batch_size=1):
        # 此函数使用旧的LQR格式，与AdHoc环境不兼容
        # 返回空结果以避免错误
        return (np.zeros(T), np.zeros((T, batch_size, self.num_users)), np.zeros((T, batch_size, self.state_dim_dnn)))

    def test(self, upper_bound, T, allocation_dnn, allocation_gnn, batch_size=1, test_type='output_constraint'):

        (cost_matrices, current_states, interference_matrices, states_matrices, allocation_decisions, observations,
         zerovec, eq_power, last_idx) = \
            self._test_init(T, batch_size=batch_size)

        [dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, capwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx,
         rapwr_cost_mtx] = cost_matrices
        [dnn_state, gnn_state, eq_state, ca_state, wmmse_state, rr_state, ra_state] = current_states
        [dnn_H, gnn_H, eq_H, ca_H, wmmse_H, rr_H, ra_H] = interference_matrices
        [dnn_states, gnn_states, eq_states, ca_states, wmmse_states, rr_states, ra_states] = states_matrices
        [dnn_power, gnn_power, equal_power, ca_power, wmmse_power, rr_power, ra_power] = allocation_decisions
        [dnn_obs, gnn_obs, eq_obs, ca_obs, wmmse_obs, rr_obs, ra_obs] = observations

        # Heuristics
        # Equal power
        eqpwr_cost_mtx, equal_power, eq_states = self.test_equal_power(upper_bound, T, eq_state, eq_obs, eq_states,
                                                                       equal_power, eq_H, eqpwr_cost_mtx)
        # WMMSE
        wmmsepwr_cost_mtx, wmmse_power, wmmse_states = self.test_wmmse(upper_bound, T, wmmse_state, wmmse_obs,
                                                                       wmmse_states,
                                                                       wmmse_power, wmmse_H, wmmsepwr_cost_mtx)
        # Control-Aware
        capwr_cost_mtx, ca_power, ca_states = self.test_control_aware(upper_bound, T, ca_state, ca_obs, ca_states,
                                                                      ca_power, ca_H, capwr_cost_mtx)
        # Round Robin
        rrpwr_cost_mtx, rr_power, rr_states = self.test_round_robin(upper_bound, T, rr_state, rr_obs, rr_states,
                                                                    rr_power, rr_H, rrpwr_cost_mtx)
        # Random Access
        rapwr_cost_mtx, ra_power, ra_states = self.test_random_access(upper_bound, T, ra_state, ra_obs, ra_states,
                                                                      ra_power, ra_H, rapwr_cost_mtx)

        # Learned Policies
        if test_type == 'output_constraint':
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_inst_constraint(allocation_dnn, upper_bound, T,
                                                                                dnn_state, dnn_obs, dnn_states,
                                                                                dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_inst_constraint(allocation_gnn, upper_bound, T,
                                                                                gnn_state, gnn_obs, gnn_states,
                                                                                gnn_power, gnn_H, gnn_cost_mtx)
        else:
            # TODO: enforce constraint over horizon during test
            overall_constraint = T * upper_bound
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_hor_constraint(allocation_dnn, upper_bound, T,
                                                                               dnn_state, dnn_obs, dnn_states,
                                                                               dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_hor_constraint(allocation_gnn, upper_bound, T,
                                                                               gnn_state, gnn_obs, gnn_states,
                                                                               gnn_power, gnn_H, gnn_cost_mtx)

        return (
        dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, capwr_cost_mtx, rapwr_cost_mtx,
        dnn_power, gnn_power, equal_power, wmmse_power, rr_power, ca_power, ra_power,
        dnn_states, gnn_states, eq_states, wmmse_states, rr_states, ca_states, ra_states)

    def test_transf(self, upper_bound, T, allocation_gnn, batch_size=1, test_type='output_constraint'):

        (cost_matrices, current_states, interference_matrices, states_matrices, allocation_decisions, observations,
         zerovec, eq_power, last_idx) = \
            self._test_init(T, batch_size=batch_size)

        [dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, capwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx,
         rapwr_cost_mtx] = cost_matrices
        [dnn_state, gnn_state, eq_state, ca_state, wmmse_state, rr_state, ra_state] = current_states
        [dnn_H, gnn_H, eq_H, ca_H, wmmse_H, rr_H, ra_H] = interference_matrices
        [dnn_states, gnn_states, eq_states, ca_states, wmmse_states, rr_states, ra_states] = states_matrices
        [dnn_power, gnn_power, equal_power, ca_power, wmmse_power, rr_power, ra_power] = allocation_decisions
        [dnn_obs, gnn_obs, eq_obs, ca_obs, wmmse_obs, rr_obs, ra_obs] = observations

        # Heuristics
        # Equal power
        eqpwr_cost_mtx, equal_power, eq_states = self.test_equal_power(upper_bound, T, eq_state, eq_obs, eq_states,
                                                                       equal_power, eq_H, eqpwr_cost_mtx)
        # WMMSE
        wmmsepwr_cost_mtx, wmmse_power, wmmse_states = self.test_wmmse(upper_bound, T, wmmse_state, wmmse_obs,
                                                                       wmmse_states,
                                                                       wmmse_power, wmmse_H, wmmsepwr_cost_mtx)
        # Control-Aware
        capwr_cost_mtx, ca_power, ca_states = self.test_control_aware(upper_bound, T, ca_state, ca_obs, ca_states,
                                                                      ca_power, ca_H, capwr_cost_mtx)
        # Round Robin
        rrpwr_cost_mtx, rr_power, rr_states = self.test_round_robin(upper_bound, T, rr_state, rr_obs, rr_states,
                                                                    rr_power, rr_H, rrpwr_cost_mtx)
        # Random Access
        rapwr_cost_mtx, ra_power, ra_states = self.test_random_access(upper_bound, T, ra_state, ra_obs, ra_states,
                                                                      ra_power, ra_H, rapwr_cost_mtx)

        # Learned Policies
        if test_type == 'output_constraint':
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_inst_constraint(allocation_gnn, upper_bound, T,
                                                                                gnn_state, gnn_obs, gnn_states,
                                                                                gnn_power, gnn_H, gnn_cost_mtx)
        else:
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_hor_constraint(allocation_gnn, upper_bound, T,
                                                                               gnn_state, gnn_obs, gnn_states,
                                                                               gnn_power, gnn_H, gnn_cost_mtx)

        return (gnn_cost_mtx, eqpwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, capwr_cost_mtx, rapwr_cost_mtx)

    def _test_init(self, T, batch_size=1):
        """
        初始化测试环境 - 适配新的AdHoc环境
        """
        # 初始化环境状态
        self.reset()
        
        # 初始化测试数据结构
        states = self.current_state.reshape(1, -1)
        states_obs = self._get_downlink_obs(
            self.H.flatten(), 
            self.user_traffic_types, 
            self.transmission_delays, 
            self.rtt_delays, 
            self.link_satellite_status,
            np.zeros(self.num_users)
        ).reshape(1, -1)
        
        # 初始化矩阵
        states_mtx = np.zeros((T, batch_size, self.observation_space.shape[0]))
        power_mtx = np.zeros((T, batch_size, self.num_users))
        H = self.H.copy()
        cost_mtx = np.zeros(T)
        
        return (states, states_obs, states_mtx, power_mtx, H, cost_mtx)
    
    def _build_action_from_power_and_satellite(self, powers, satellite_mask):
        """从功率和卫星选择构建动作"""
        # 将功率归一化到[-1, 1]
        normalized_powers = (powers / self.max_pwr_perplant) * 2 - 1
        normalized_powers = np.clip(normalized_powers, -1, 1)
        
        # 构建动作 [功率, 是否用卫星]
        action = np.zeros((self.num_users, 2))
        action[:, 0] = normalized_powers  # 功率
        action[:, 1] = satellite_mask.astype(float)  # 卫星选择
        
        # 如果有约束维度，添加lambda
        if hasattr(self, 'constraint_dim') and self.constraint_dim > 0:
            lambda_values = np.zeros(self.constraint_dim)
            action = np.concatenate([action.flatten(), lambda_values])
        
        return action


# ------------------------------------------- Downlink Environments ------------------------------------------ #

# LQRAdHocDownlink class - completely removed LQR parameters from its init and super() call
class LQRAdHocDownlink(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1,  # Removed p, q, Ao from here
                 T=40, gamma=0.99, pl=2., pp=5., p0=1., num_features=1,
                 scaling=False):  # Removed W, Wobs, Wobs_channels, a0, r
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu,
                         T=T, gamma=gamma, pl=pl, pp=pp, p0=p0,
                         # Removed p, q, Ao, W, Wobs, Wobs_channels, a0, r from super() call
                         num_features=num_features, scaling=scaling)

        # Action space for each user: (num_users, 2)
        # 第一列为功率（-1~1），第二列为是否用卫星（-1~1，step中round到0/1）
        self.action_space = spaces.Box(low=np.array([[-1, -1]] * num_users), high=np.array([[1, 1]] * num_users))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs_flat = obs[:self.channel_state_dim]
        log2_sinr = np.zeros(self.num_users)
        agent_obs = self._get_downlink_obs(channel_obs_flat, self.user_traffic_types,
                                           self.transmission_delays, self.rtt_delays, self.link_satellite_status, log2_sinr)
        return agent_obs

    def step(self, action):
        # 动作解析
        action = np.asarray(action)
        
        # 处理约束维度
        if len(action) > self.num_users * 2:
            # 如果有约束维度，提取lambd
            lambd = action[-self.constraint_dim:]
            action = action[:-self.constraint_dim]
        else:
            lambd = np.array([0.0])  # 默认约束值
            
        # 确保action是1D数组，然后重塑为(num_users, 2)
        if action.ndim == 1:
            if len(action) != self.num_users * 2:
                raise ValueError(f"Action length {len(action)} does not match expected {self.num_users * 2}")
            action = action.reshape((self.num_users, 2))
        elif action.ndim == 2:
            if action.shape != (self.num_users, 2):
                raise ValueError(f"Action shape {action.shape} does not match expected ({self.num_users}, 2)")
        else:
            raise ValueError(f"Action has unexpected dimensions: {action.ndim}")
            
        power_raw = action[:, 0]
        is_satellite_raw = action[:, 1]
        # 缩放功率
        power_scaled = self.scale_power(power_raw)
        # 是否用卫星，强制为0或1
        is_satellite = np.clip(np.round(is_satellite_raw), 0, 1).astype(int)

        # 分配功率到地面/卫星
        satellite_mask = is_satellite == 1
        ground_mask = ~satellite_mask
        # 所有用户都参与地面功率分配优化
        # 卫星用户的选择只影响最终链路选择，不影响功率优化
        actual_ground_tx_power = power_scaled.copy()
        
        # 统计总功率（所有用户的功率）
        total_power = np.sum(power_scaled)

        # 计算功率约束惩罚
        current_ground_power_sum = np.sum(actual_ground_tx_power)
        action_penalty = (current_ground_power_sum - self.upperbound)
                
        # 记录约束历史
        self.constraint_violation = action_penalty
        self.constraint_hist.append(action_penalty)
        self.Lagrangian_hist.append(np.dot(lambd, action_penalty))
        self.downlink_constraint_dualvar = np.dot(lambd, action_penalty)

        # 调用状态转移函数计算延迟和奖励
        delays_per_link, SINR_all_links, pdr_all_links, reward, done, \
            throughput_HVFT, throughput_others, throughput_all = \
            self._update_control_states_downlink(actual_ground_tx_power, satellite_mask, lambd, action_penalty)

        # 更新状态
        self.H, channel_states = self.sample_graph()
        self.current_state = channel_states
        log2_sinr = np.log2(1 + np.maximum(SINR_all_links, 1e-6))
        states_obs = self._get_downlink_obs(channel_states, self.user_traffic_types,
                                            self.transmission_delays, self.rtt_delays, self.link_satellite_status, log2_sinr)
        done = False
        self.time_step += 1
        if self.time_step > self.T - 1:
            done = True
        # 计算info字段
        sinr_sum = np.sum(SINR_all_links)
        sinr_mean = np.mean(SINR_all_links)
        sinr_min = np.min(SINR_all_links)
        log_sinr_sum = np.sum(np.log2(1 + np.maximum(SINR_all_links, 1e-6)))
        delay_sum = np.sum(delays_per_link)
        delay_mean = np.mean(delays_per_link)
        power_efficiency = 1.0 - (total_power / (self.num_users * self.max_pwr_perplant))
        
        info = {
            "sinr_sum": float(sinr_sum),
            "sinr_mean": float(sinr_mean),
            "sinr_min": float(sinr_min),
            "log_sinr_sum": float(log_sinr_sum),
            "delay_sum": float(delay_sum),
            "delay_mean": float(delay_mean),
            "total_power": float(total_power),
            "ground_users_count": int(np.sum(ground_mask)),
            "satellite_users_count": int(np.sum(satellite_mask)),
            # 添加custom_callback需要的字段
            "throughput_hvft": float(throughput_HVFT),
            "throughput_others": float(throughput_others),
            "throughput_all_total": float(throughput_all),
            "link_pdr_avg": float(np.mean(pdr_all_links))
        }
        return states_obs, reward, bool(done), info


# LQRAdHocDownlinkOutputConstraint class - completely removed LQR parameters from its init and super() call
class LQRAdHocDownlinkOutputConstraint(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1,
                 T=40, gamma=0.99, pl=2., pp=5., p0=1, num_features=1, scaling=False,
                 t=1, v=138, R=3000, d=400, choice=1, A_b=30, A_m=3, f=930, link_all=100, bandwidth=1000, beta=0.3,
                 alpha=0.3):
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu,
                         T=T, gamma=gamma, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling,
                         t=t, v=v, R=R, d=d, choice=choice, A_b=A_b, A_m=A_m, f=f, link_all=link_all,
                         bandwidth=bandwidth, alpha=alpha, beta=beta)

        # (num_users, 2)
        self.action_space = spaces.Box(low=-np.ones((num_users, 2)), high=np.ones((num_users, 2)))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs_flat = obs[:self.channel_state_dim]
        log2_sinr = np.zeros(self.num_users)
        agent_obs = self._get_downlink_obs(channel_obs_flat, self.user_traffic_types,
                                           self.transmission_delays, self.rtt_delays, self.link_satellite_status,log2_sinr)
        return agent_obs

    def step(self, action):
        # 动作解析
        action = np.asarray(action)
        
        # 处理约束维度
        if len(action) > self.num_users * 2:
            # 如果有约束维度，提取lambd
            lambd = action[-self.constraint_dim:]
            action = action[:-self.constraint_dim]
        else:
            lambd = np.array([0.0])  # 默认约束值
            
        # # 确保action是1D数组，然后重塑为(num_users, 2)
        # if action.ndim == 1:
        #     if len(action) != self.num_users * 2:
        #         raise ValueError(f"Action length {len(action)} does not match expected {self.num_users * 2}")
        #     action = action.reshape((self.num_users, 2))
        # elif action.ndim == 2:
        #     if action.shape != (self.num_users, 2):
        #         raise ValueError(f"Action shape {action.shape} does not match expected ({self.num_users}, 2)")
        #     else:
        #         raise ValueError(f"Action has unexpected dimensions: {action.ndim}")
            
        power_raw = action[:, 0]
        is_satellite_raw = action[:, 1]
        
        # 缩放功率
        power_scaled = self.scale_power(power_raw)
        # 是否用卫星，强制为0或1
        # is_satellite = np.clip(np.round(is_satellite_raw), 0, 1).astype(int)

        # 分配功率到地面/卫星
        # satellite_mask = is_satellite == 1
        # ground_mask = ~satellite_mask
        
        # 所有用户都参与地面功率分配优化
        # 卫星用户的选择只影响最终链路选择，不影响功率优化
        actual_ground_tx_power = power_scaled.copy()
        
        # 统计总功率（所有用户的功率）
        total_power = np.sum(power_scaled)

        # 计算功率约束惩罚
        current_ground_power_sum = np.sum(actual_ground_tx_power)
        action_penalty = (current_ground_power_sum - self.upperbound)
        
        # 记录约束历史
        self.constraint_violation = action_penalty
        self.constraint_hist.append(action_penalty)
        self.Lagrangian_hist.append(np.dot(lambd, action_penalty))
        self.downlink_constraint_dualvar = np.dot(lambd, action_penalty)

        # 调用状态转移函数计算延迟和奖励
        delays_per_link, SINR_all_links, pdr_all_links, reward, done, \
            throughput_HVFT, throughput_others, throughput_all = \
            self._update_control_states_downlink(actual_ground_tx_power, lambd, action_penalty)

        # 更新状态
        self.H, channel_states = self.sample_graph()
        self.current_state = channel_states
        log2_sinr = np.log2(1 + np.maximum(SINR_all_links, 1e-6))
        states_obs = self._get_downlink_obs(channel_states, self.user_traffic_types,
                                            self.transmission_delays, self.rtt_delays, self.link_satellite_status, log2_sinr)

        done = False
        self.time_step += 1
        if self.time_step > self.T - 1:
            done = True
            
        # 计算info字段
        sinr_sum = np.sum(SINR_all_links)
        sinr_mean = np.mean(SINR_all_links)
        sinr_min = np.min(SINR_all_links)
        log_sinr_sum = np.sum(np.log2(1 + np.maximum(SINR_all_links, 1e-6)))
        delay_sum = np.sum(delays_per_link)
        delay_mean = np.mean(delays_per_link)
        power_efficiency = 1.0 - (total_power / (self.num_users * self.max_pwr_perplant))

        info = {
            "sinr_sum": float(sinr_sum),
            "sinr_mean": float(sinr_mean),
            "sinr_min": float(sinr_min),
            "log_sinr_sum": float(log_sinr_sum),
            "delay_sum": float(delay_sum),
            "delay_mean": float(delay_mean),
            "total_power": float(total_power),
            "satellite_users_count": int(np.sum(satellite_mask)),
            # 添加custom_callback需要的字段
            "throughput_hvft": float(throughput_HVFT),
            "throughput_others": float(throughput_others),
            "throughput_all_total": float(throughput_all),
            "link_pdr_avg": float(np.mean(pdr_all_links))
        }
        
        return states_obs, reward, bool(done), info

    def test_benchmark_algorithms(self, T=40, batch_size=1):
        """
        测试基准算法：Round Robin, WMMSE, Random Power, Max Power
        适配新的动作空间和奖励函数
        """
        # 测试结果存储
        benchmark_results = {
            'round_robin': {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []},
            'wmmse': {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []},
            'random_power': {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []},
            'max_power': {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []}
        }
        
        # 1. Round Robin 测试
        print("Testing Round Robin...")
        self.reset()  # 重置环境
        rr_results = self._test_round_robin_new(T)
        benchmark_results['round_robin'] = rr_results
        
        # 2. WMMSE 测试
        print("Testing WMMSE...")
        self.reset()  # 重置环境
        wmmse_results = self._test_wmmse_new(T)
        benchmark_results['wmmse'] = wmmse_results
        
        # 3. Random Power 测试
        print("Testing Random Power...")
        self.reset()  # 重置环境
        random_results = self._test_random_power_new(T)
        benchmark_results['random_power'] = random_results
        
        # 4. Max Power 测试
        print("Testing Max Power...")
        self.reset()  # 重置环境
        max_results = self._test_max_power_new(T)
        benchmark_results['max_power'] = max_results
        
        return benchmark_results
    
    def _test_round_robin_new(self, T):
        """Round Robin算法 - 适配新动作空间"""
        results = {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []}
        last_idx = 0
        
        for tt in range(T):
            # 生成Round Robin功率分配
            rr_powers = self._generate_round_robin_powers(last_idx)
            last_idx = (last_idx + 1) % self.num_users
            
            # 生成卫星选择（随机选择部分用户使用卫星）
            satellite_mask = self._generate_satellite_selection('round_robin')
            
            # 构建动作
            action = self._build_action_from_power_and_satellite(rr_powers, satellite_mask)
            
            # 执行环境步骤
            obs, reward, done, info = self.step(action)
            
            # 记录结果
            results['rewards'].append(reward)
            results['sinr_sum'].append(info.get('sinr_sum', 0))
            results['delay_mean'].append(info.get('delay_mean', 0))
            results['power_efficiency'].append(info.get('power_efficiency', 0))
            
            if done:
                break
        
        return results
    
    def _test_wmmse_new(self, T):
        """WMMSE算法 - 适配新动作空间"""
        results = {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []}
        
        for tt in range(T):
            # 生成WMMSE功率分配
            wmmse_powers = self._generate_wmmse_powers(self.H)
            
            # 生成卫星选择（基于信道质量）
            satellite_mask = self._generate_satellite_selection('wmmse', self.H)
            
            # 构建动作
            action = self._build_action_from_power_and_satellite(wmmse_powers, satellite_mask)
            
            # 执行环境步骤
            obs, reward, done, info = self.step(action)
            
            # 记录结果
            results['rewards'].append(reward)
            results['sinr_sum'].append(info.get('sinr_sum', 0))
            results['delay_mean'].append(info.get('delay_mean', 0))
            results['power_efficiency'].append(info.get('power_efficiency', 0))
            
            if done:
                break
        
        return results
    
    def _test_random_power_new(self, T):
        """Random Power算法 - 适配新动作空间"""
        results = {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []}
        
        for tt in range(T):
            # 生成随机功率分配
            random_powers = self._generate_random_powers()
            
            # 生成随机卫星选择
            satellite_mask = self._generate_satellite_selection('random')
            
            # 构建动作
            action = self._build_action_from_power_and_satellite(random_powers, satellite_mask)
            
            # 执行环境步骤
            obs, reward, done, info = self.step(action)
            
            # 记录结果
            results['rewards'].append(reward)
            results['sinr_sum'].append(info.get('sinr_sum', 0))
            results['delay_mean'].append(info.get('delay_mean', 0))
            results['power_efficiency'].append(info.get('power_efficiency', 0))
            
            if done:
                break
        
        return results
    
    def _test_max_power_new(self, T):
        """Max Power算法 - 适配新动作空间"""
        results = {'rewards': [], 'sinr_sum': [], 'delay_mean': [], 'power_efficiency': []}
        
        for tt in range(T):
            # 生成最大功率分配
            max_powers = self._generate_max_powers()
            
            # 生成卫星选择（基于延迟需求）
            satellite_mask = self._generate_satellite_selection('max_power')
            
            # 构建动作
            action = self._build_action_from_power_and_satellite(max_powers, satellite_mask)
            
            # 执行环境步骤
            obs, reward, done, info = self.step(action)
            
            # 记录结果
            results['rewards'].append(reward)
            results['sinr_sum'].append(info.get('sinr_sum', 0))
            results['delay_mean'].append(info.get('delay_mean', 0))
            results['power_efficiency'].append(info.get('power_efficiency', 0))
            
            if done:
                break
        
        return results
    
    def _generate_round_robin_powers(self, last_idx):
        """生成Round Robin功率分配"""
        powers = np.zeros(self.num_users)
        powers[last_idx] = self.max_pwr_perplant
        return powers
    
    def _generate_wmmse_powers(self, H):
        """生成WMMSE功率分配"""
        try:
            # 使用WMMSE算法计算功率
            if hasattr(self, 'wmmse'):
                wmmse_powers = self.wmmse(H[None, :]).flatten()
            else:
                # 如果没有WMMSE函数，使用基于信道质量的功率分配
                channel_qualities = np.diag(H)
                wmmse_powers = channel_qualities / np.sum(channel_qualities) * self.max_pwr_perplant
            # 归一化到功率约束
            wmmse_powers = np.clip(wmmse_powers, 0, self.max_pwr_perplant)
            return wmmse_powers
        except:
            # 如果WMMSE失败，使用均匀功率分配
            return np.ones(self.num_users) * self.max_pwr_perplant / self.num_users
    
    def _generate_random_powers(self):
        """生成随机功率分配"""
        return np.random.uniform(0, self.max_pwr_perplant, self.num_users)
    
    def _generate_max_powers(self):
        """生成最大功率分配"""
        return np.ones(self.num_users) * self.max_pwr_perplant
    
    def _generate_satellite_selection(self, algorithm_type, H=None):
        """生成卫星选择策略"""
        if algorithm_type == 'round_robin':
            # Round Robin: 轮流选择用户使用卫星
            satellite_mask = np.zeros(self.num_users, dtype=bool)
            num_satellite_users = min(3, self.num_users // 3)  # 选择1/3的用户
            indices = np.random.choice(self.num_users, num_satellite_users, replace=False)
            satellite_mask[indices] = True
            
        elif algorithm_type == 'wmmse':
            # WMMSE: 基于信道质量选择卫星用户
            if H is not None:
                # 选择信道质量较差的用户使用卫星
                channel_qualities = np.diag(H)
                worst_indices = np.argsort(channel_qualities)[:self.num_users//4]  # 选择1/4最差的用户
                satellite_mask = np.zeros(self.num_users, dtype=bool)
                satellite_mask[worst_indices] = True
            else:
                satellite_mask = np.random.choice([True, False], self.num_users, p=[0.2, 0.8])
                
        elif algorithm_type == 'random':
            # Random: 随机选择卫星用户
            satellite_mask = np.random.choice([True, False], self.num_users, p=[0.3, 0.7])
            
        elif algorithm_type == 'max_power':
            # Max Power: 基于延迟需求选择卫星用户
            # 假设HVFT用户优先使用卫星
            satellite_mask = np.zeros(self.num_users, dtype=bool)
            hvft_users = (self.user_traffic_types == 1)
            satellite_mask[hvft_users] = True
            # 再随机选择一些其他用户
            other_users = ~hvft_users
            if np.any(other_users):
                other_indices = np.where(other_users)[0]
                num_additional = min(2, len(other_indices))
                additional_indices = np.random.choice(other_indices, num_additional, replace=False)
                satellite_mask[additional_indices] = True
        
        else:
            # 默认策略
            satellite_mask = np.random.choice([True, False], self.num_users, p=[0.25, 0.75])
        
        return satellite_mask
    
    def print_benchmark_results(self, benchmark_results):
        """打印基准算法测试结果"""
        print("\n" + "="*60)
        print("基准算法测试结果对比")
        print("="*60)
        
        for algorithm, results in benchmark_results.items():
            print(f"\n{algorithm.upper()} 算法:")
            print(f"  平均奖励: {np.mean(results['rewards']):.4f}")
            print(f"  平均SINR和: {np.mean(results['sinr_sum']):.4f}")
            print(f"  平均延迟: {np.mean(results['delay_mean']):.4f}")
            print(f"  平均功率效率: {np.mean(results['power_efficiency']):.4f}")
        
        # 找出最佳算法
        best_algorithm = max(benchmark_results.keys(), 
                           key=lambda x: np.mean(benchmark_results[x]['rewards']))
        print(f"\n最佳算法: {best_algorithm.upper()}")
        print("="*60)