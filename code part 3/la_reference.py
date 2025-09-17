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

from AdHoc.config_downlinkconstraint import num_users, satellite_enabled, max_delay ,p,satellite_subchannel_num
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
                 satellite_freq_hz=30e9, satellite1_bandwidth=10e6, satellite2_bandwidth=10e6,
                 c_light=3e8, k_boltzmann=1.38e-23,
                 satellite_noise_figure_db=1.2, satellite_tx_power_init=2.0,  # This now could be system-level TST power
                 satellite_antenna_gain_tx=10 ** (43.3 / 10), satellite_g_over_t=18.5,
                 satellite_subchannel_num=satellite_subchannel_num,
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
        self.others_load_min_bits = 3000 * 8  # 3000 bytes/s = 24000 bits
        self.others_load_max_bits = 3000 * 8  # 3000 bytes/s = 24000 bits
        self.hvft_load_mb = 3000 * 8  # 3000 bytes/s = 24000 bits
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
        # Observation space dimension: channel state + user traffic types + transmission delays + RTT delays + satellite status + HVFT accumulation info + power
        # 实际obs结构：[channel_obs, user_traffic_types, transmission_delays, rtt_delays, link_satellite_status, log2_sinr, hvft_accumulation_info, normalized_updated_power]
        # 维度计算：num_users**2 + num_users*10 = 900 + 300 = 1200
        self.state_dim_dnn = self.channel_state_dim + num_users * 10

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

        # Jakes衰落模型参数
        self.time_slot_duration = 1.0  # 时间槽持续时间（秒）
        self.c_light = 3e8  # 光速
        
        # 初始化信道状态变量
        self.h_small_scale = None  # 小尺度衰落分量
        self.h_small_scale_prev = None  # 前一时刻的小尺度衰落
        self.alpha_large_scale = None  # 大尺度衰落分量（路径损耗）
        self.channel_initialized = False  # 信道是否已初始化

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
        
        # HVFT延迟累积机制参数
        self.hvft_delay_tolerance = 5  # HVFT延迟容忍度：5个回合
        self.hvft_accumulated_data = np.zeros(num_users)  # 每个用户累积的HVFT数据
        self.hvft_accumulation_rounds = np.zeros(num_users)  # 每个用户累积的回合数
        self.hvft_transmission_decisions = np.zeros(num_users)  # HVFT传输决策（0=不传输，1=传输）
        self.hvft_satellite_transmission = np.zeros(num_users)  # HVFT卫星传输标记
        
        # Delay tolerance parameters
        self.delay_tolerance_params = {
            'T_HVFT_max': 5,      # HVFT最大延迟容忍度：5个回合
            'T_others_max': 0.2,  # Others最大延迟容忍度：0.2秒（更敏感）
            'K_HVFT': 2.0,        # HVFT延迟惩罚系数（更低，更容忍）
            'K_others': 100.0     # Others延迟惩罚系数（更高，更敏感）
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

    def calculate_correlation_coefficient(self):
        """
        计算相关系数 ρ = J_0(2 * π * f_d * T)
        其中：J_0是零阶贝塞尔函数，f_d是多普勒频率，T是时间槽长度
        """
        from scipy.special import j0
        
        # 计算多普勒频率 f_d = v * f_c / c
        # 其中v是相对速度，f_c是载波频率，c是光速
        doppler_frequency = self.v * self.f / self.c_light
        
        # 计算相关系数：ρ = J_0(2 * π * f_d * T)
        rho = j0(2 * np.pi * doppler_frequency * self.time_slot_duration)
        
        return rho

    def initialize_jakes_channel_model(self):
        """
        初始化Jakes衰落模型
        设置大尺度衰落分量和小尺度衰落分量的初始值
        """
        # 计算大尺度衰落分量（路径损耗）
        self.alpha_large_scale = self.calculate_large_scale_fading()
        
        # 初始化小尺度衰落分量
        # 生成复高斯随机变量：实部和虚部都是均值为0，方差为1/2的正态分布
        real_part = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_users))
        imag_part = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_users))
        self.h_small_scale = real_part + 1j * imag_part
        
        # 保存前一时刻的小尺度衰落（初始时等于当前值）
        self.h_small_scale_prev = self.h_small_scale.copy()
        
        # 标记信道已初始化
        self.channel_initialized = True

    def calculate_large_scale_fading(self):
        """
        计算大尺度衰落分量 α_i->j
        基于路径损耗模型，使用固定的几何距离
        """
        # 计算固定的几何距离（基于初始位置）
        _, r = self.doppler_shift_effect(0, self.v, self.d, self.R)  # t=0时的距离
        if r <= 0:
            r = 1e-9
        
        # 使用Hata路径损耗模型计算路径损耗
        model1 = 5.74 * np.log10(self.A_b) - 30.42 + 26.16 * np.log10(self.f) - 13.82 * np.log10(self.A_b) - 3.2 * (
                np.log10(11.75 * self.A_m) ** 2) + (
                         44.9 - 6.55 * np.log10(self.A_b) - 6.72) * np.log10(r)
        model2 = -21.42 + 26.16 * np.log10(self.f) - 13.82 * np.log10(self.A_b) - 3.2 * (np.log10(11.75 * self.A_m) ** 2) + (
                44.9 - 6.55 * np.log10(self.A_b) - 9.62) * np.log10(r)
        
        if self.choice == 1:
            PL = model1
        else:
            PL = model2
        
        # 将路径损耗转换为信道增益（大尺度衰落）
        # α_i->j = 10^(-PL/10)
        alpha = 10 ** (-PL / 10)
        
        # 考虑网络拓扑结构L（如果L为None，则使用全连接矩阵）
        alpha_matrix = alpha * np.ones((self.num_users, self.num_users))
        if self.L is not None:
            alpha_matrix = alpha_matrix * self.L
        else:
            # 如果L为None，创建一个默认的全连接矩阵（除了对角线为0）
            default_L = np.ones((self.num_users, self.num_users))
            np.fill_diagonal(default_L, 0)  # 对角线设为0（无自环）
            alpha_matrix = alpha_matrix * default_L
        
        return alpha_matrix

    def update_jakes_channel_model(self):
        """
        根据Jakes衰落模型更新信道增益
        g_i->j^(t) = |h_i->j^(t)|^2 * α_i->j
        h_i->j^(t) = ρ * h_i->j^(t-1) + √(1 - ρ^2) * e_i->j^(t)
        """
        if not self.channel_initialized:
            self.initialize_jakes_channel_model()
        
        # 计算相关系数
        rho = self.calculate_correlation_coefficient()
        
        # 生成信道创新过程 e_i->j^(t)
        # 创新过程是i.i.d.的圆对称复高斯随机变量，单位方差
        real_innovation = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_users))
        imag_innovation = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_users))
        innovation = real_innovation + 1j * imag_innovation
        
        # 更新小尺度衰落分量
        # h_i->j^(t) = ρ * h_i->j^(t-1) + √(1 - ρ^2) * e_i->j^(t)
        self.h_small_scale = rho * self.h_small_scale_prev + np.sqrt(1 - rho**2) * innovation
        
        # 计算信道增益
        # g_i->j^(t) = |h_i->j^(t)|^2 * α_i->j
        channel_gains = np.abs(self.h_small_scale)**2 * self.alpha_large_scale
        
        # 保存当前小尺度衰落作为下一时刻的前一时刻值
        self.h_small_scale_prev = self.h_small_scale.copy()
        
        return channel_gains

    def interference_packet_delivery_rate(self, H, actions):
        actions_vec = actions[:, None]
        bandwidth = self.bandwidth
        H_diag = np.diag(H)
        num = np.multiply(H_diag, actions)

        doppler_effect_coefficent, _ = self.doppler_shift_effect(self.t, self.v, self.d, self.R)

        x_values = np.linspace(-1, 1, 1000)
        integral_result = np.trapz((1 - abs(x_values)) * j0(2 * np.pi * doppler_effect_coefficent * x_values), x_values)
        W_ICI = 1 - integral_result
        W_ICI = max(0.0, W_ICI)

        H_interference = (H - np.diag(np.diag(H))).transpose()
        den = (np.dot(H_interference * (1 + W_ICI), actions_vec) + self.sigma ** 2).flatten()

        den[den < 1e-9] = 1e-9
        SINR = num / den
        pdr = 1 - np.exp(- SINR)
        pdr = np.nan_to_num(pdr)
        # channel_bandwidth = bandwidth / self.num_users
        channel_capacity_rate = bandwidth * np.log2(1 + SINR)
        effective_throughput_rate = channel_capacity_rate
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

    def link_throughput_and_delay(self, throughput_per_link_rate, user_traffic_types, hvft_agent_decisions=None, satellite_mask=None):
        if len(throughput_per_link_rate) != len(user_traffic_types):
            raise ValueError("Arrays throughput_per_link_rate and user_traffic_types must have same length.")

        HVFT_block_size = self.hvft_load_mb

        throughput_HVFT_total = 0.0
        throughput_others_total = 0.0

        traffic_loads_per_link = np.zeros(self.num_users)
        deliver_times_per_chunk = np.full(self.num_users, float('inf'))

        for index, traffic_type in enumerate(user_traffic_types):
            current_link_rate = throughput_per_link_rate[index]

            if traffic_type == 1:  # HVFT流量 - 只有HVFT用户才应用HVFT传输决策
                # 获取智能体的HVFT决策（如果有的话）
                agent_decision = None
                if hvft_agent_decisions is not None:
                    agent_decision = hvft_agent_decisions[index]
                
                # HVFT智能传输决策 - 只对HVFT用户有效
                # 检查是否选择了卫星传输
                satellite_selected = satellite_mask is not None and satellite_mask[index]
                hvft_decision = self._make_hvft_transmission_decision(index, current_link_rate, agent_decision, satellite_selected)
                self.hvft_transmission_decisions[index] = hvft_decision
                
                if hvft_decision == 1:  # 决定传输HVFT
                    # 计算需要传输的数据量（包括累积的数据）
                    total_hvft_data = HVFT_block_size + self.hvft_accumulated_data[index]
                    traffic_loads_per_link[index] = total_hvft_data
                    
                    if current_link_rate < 1e-9:
                        deliver_times_per_chunk[index] = max_delay
                    else:
                        deliver_times_per_chunk[index] = total_hvft_data / current_link_rate
                    
                    # 传输成功后，清空累积数据
                    if deliver_times_per_chunk[index] <= self.t:
                        delivered_data_this_step = total_hvft_data
                        self.hvft_accumulated_data[index] = 0
                        self.hvft_accumulation_rounds[index] = 0
                        self.hvft_satellite_transmission[index] = 0
                    else:
                        delivered_data_this_step = current_link_rate * self.t
                        # 传输失败，数据重新累积
                        self.hvft_accumulated_data[index] = total_hvft_data - delivered_data_this_step
                        self.hvft_accumulation_rounds[index] += 1
                else:  # 决定不传输HVFT
                    # 累积HVFT数据
                    self.hvft_accumulated_data[index] += HVFT_block_size
                    self.hvft_accumulation_rounds[index] += 1
                    deliver_times_per_chunk[index] = 0  # HVFT延迟归0
                    delivered_data_this_step = 0
                    
                    # 检查是否需要卫星传输
                    if self.hvft_accumulation_rounds[index] >= self.hvft_delay_tolerance:
                        self.hvft_satellite_transmission[index] = 1
                        # 卫星传输逻辑将在后续处理
                    
                    traffic_loads_per_link[index] = 0  # 不传输，负载为0
            else:  # Others流量
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
        
        base_trip_delay = 2 * self.leo1_altitude / self.c_light
        # 添加0.08-0.12秒之间的随机波动延迟
        fluctuation_delay = np.random.uniform(0.08, 0.12)
        T_trip_n_satellite = base_trip_delay + fluctuation_delay    
        # T_trip_n_satellite = 2 * self.leo1_altitude / self.c_light + 0.2
        
        # 智能体选择的卫星用户
        satellite_users = np.where(satellite_mask)[0]
        
        # 过滤掉流量负载为0的用户
        valid_satellite_users = []
        for idx in satellite_users:
            if traffic_loads_per_link[idx] > 0:  # 只考虑有流量的用户
                valid_satellite_users.append(idx)

        # DOV选择策略：计算每个卫星用户的优先级
        satellite_priorities = []
        for idx in satellite_users:
            ground_delay = deliver_times_per_chunk[idx] + rtt_delays_all_links[idx]
            traffic_load = traffic_loads_per_link[idx]
            traffic_type = self.user_traffic_types[idx]
            
            # 优先级计算：考虑延迟、流量、业务类型
            if traffic_type == 1:  # HVFT业务
                priority = ground_delay * traffic_load * 1.0  # HVFT优先级降低
            else:  # Others业务（优先级更高）
                priority = ground_delay * traffic_load * 3.0  # Others优先级更高（3倍权重）
            
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
                ground_delay = deliver_times_per_chunk[idx] + rtt_delays_all_links[idx]
                final_delays_per_link[idx] = min(ground_delay, max_delay)
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
        """
        使用Jakes衰落模型采样信道增益
        """
        # 使用Jakes模型更新信道增益
        A = self.update_jakes_channel_model()
        
        # 应用阈值处理，但确保不会全为零
        A[A < 0.001] = 0.0
        # 如果矩阵全为零，设置一个最小非零值
        if np.all(A == 0):
            A[0, 0] = 0.001  # 设置一个最小非零值
        
        # 归一化处理
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()
        
        return A, A_flat

    def sample_graph_uplink(self):
        """
        使用Jakes衰落模型采样上行链路信道增益
        """
        # 使用Jakes模型更新信道增益
        A = self.update_jakes_channel_model()
        
        # 应用阈值处理，但确保不会全为零
        A[A < 0.001] = 0.0
        # 如果矩阵全为零，设置一个最小非零值
        if np.all(A == 0):
            A[0, 0] = 0.001  # 设置一个最小非零值
        
        # 转置矩阵（上行链路）
        A = A.T
        
        # 归一化处理
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
        # 初始化Jakes信道模型（如果还没有初始化）
        if not self.channel_initialized:
            self.initialize_jakes_channel_model()
        
        channel_obs_flat = self.sample(batch_size=1)
        self.current_state = channel_obs_flat
        self.time_step = 0

        num_hvft_users = int(self.num_users * self.hvft_ratio)
        shuffled_indices = self.np_random.permutation(self.num_users)
        self.user_traffic_types = np.zeros(self.num_users)
        self.user_traffic_types[shuffled_indices[:num_hvft_users]] = 1

        self.current_link_delays = np.array(self.hsr_rtt_delay(self.num_users))

        self.link_satellite_status = np.zeros(self.num_users)
        
        # 重置HVFT累积机制
        self.hvft_accumulated_data = np.zeros(self.num_users)
        self.hvft_accumulation_rounds = np.zeros(self.num_users)
        self.hvft_transmission_decisions = np.zeros(self.num_users)
        self.hvft_satellite_transmission = np.zeros(self.num_users)
        
        # 初始化功率分配
        self.current_power_allocation = np.zeros(self.num_users)

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


    def _make_hvft_transmission_decision(self, user_index, current_link_rate, agent_decision=None, satellite_selected=False):
        # 如果智能体提供了决策，直接使用
        if agent_decision is not None:
            return agent_decision
        
        # 如果选择了卫星传输，强制传输（通过卫星）
        if satellite_selected:
            return 1
        
        # 否则使用环境自动决策逻辑
        # 获取当前用户的HVFT累积状态
        accumulated_rounds = self.hvft_accumulation_rounds[user_index]

        # 如果已经累积了最大容忍回合数，强制传输
        if accumulated_rounds >= self.hvft_delay_tolerance:
            return 1
        
        # 如果链路质量很差（吞吐量很低），选择不传输
        if current_link_rate < self.hvft_load_mb * 0.5:  # 吞吐量低于HVFT负载的50%
            return 0

        # 如果链路质量良好，选择传输
        if current_link_rate > self.hvft_load_mb * 1.2:  # 吞吐量超过HVFT负载的120%
            return 1
        
        # 默认选择不传输，继续累积
        return 0

    # Modified _update_control_states_downlink
    def _update_control_states_downlink(self, actual_ground_tx_power, satellite_mask, lambd,
                                        action_penalty, hvft_agent_decisions=None):
        self.time_step += 1
        # 根据HVFT传输决策调整功率分配
        adjusted_ground_tx_power = actual_ground_tx_power.copy()


        # 对不传输的HVFT用户，将功率归零
        for i in range(self.num_users):
            if self.user_traffic_types[i] == 1 and self.hvft_transmission_decisions[i] == 0:
                # 如果选择了卫星传输，保持原始功率（因为会通过卫星传输）
                if not satellite_mask[i]:
                    adjusted_ground_tx_power[i] = 0.0
        
        # 使用调整后的功率计算SINR和吞吐量
        SINR_per_link, pdr_per_link, throughput_per_link_rate = self.interference_packet_delivery_rate(
            self.H, adjusted_ground_tx_power)
        
        throughput_HVFT_total, throughput_others_total, throughput_all_total, deliver_times_per_chunk, traffic_loads_per_link \
            = self.link_throughput_and_delay(throughput_per_link_rate, self.user_traffic_types, hvft_agent_decisions, satellite_mask)
        rtt_delays_all_links = np.array(self.hsr_rtt_delay(self.num_users))
        
        # 存储分解的延迟信息，让智能体更好地理解功率分配的影响
        self.transmission_delays = np.copy(deliver_times_per_chunk)  # 传输延迟（受功率影响）
        self.rtt_delays = np.copy(rtt_delays_all_links)             # RTT延迟（固定，不受功率影响）
        
        # 初始化最终延迟数组
        final_delays_per_link = np.zeros(self.num_users)
        self.link_satellite_status = np.zeros(self.num_users, dtype=int)  # Reset status
        
        # 地面用户延迟：传输延迟 + RTT延迟
        ground_mask = ~satellite_mask
        ground_total_delays = np.copy(deliver_times_per_chunk[ground_mask]) + rtt_delays_all_links[ground_mask]
        ground_total_delays = np.minimum(ground_total_delays, max_delay)
        # HVFT等待传输的用户，其延迟归0
        
        final_delays_per_link[ground_mask] = ground_total_delays
        
        # 卫星用户延迟：调用专门的卫星计算函数
        satellite_delays, satellite_status = self.calculate_satellite_delays_with_dov(
            satellite_mask, deliver_times_per_chunk, rtt_delays_all_links, traffic_loads_per_link
        )
        final_delays_per_link += satellite_delays
        self.link_satellite_status = satellite_status
        
        # 处理HVFT卫星传输
        hvft_satellite_users = np.where(self.hvft_satellite_transmission == 1)[0]
        for user_idx in hvft_satellite_users:
            if self.user_traffic_types[user_idx] == 1:  # 确认是HVFT用户
                # 使用卫星传输HVFT累积数据
                satellite_capacity = self.calculate_satellite_throughput(
                    self.satellite_tx_power, self.satellite1_bandwidth
                )
                
                if satellite_capacity > 0:
                    # 计算卫星传输延迟
                    base_trip_delay = 2 * self.leo1_altitude / self.c_light
                    # 添加0.08-0.12秒之间的随机波动延迟
                    fluctuation_delay = np.random.uniform(0.08, 0.12)
                    T_trip_satellite = base_trip_delay + fluctuation_delay 
                    
                    satellite_transmission_time = self.hvft_accumulated_data[user_idx] / satellite_capacity
                    satellite_delay = T_trip_satellite + satellite_transmission_time
                    
                    # 更新最终延迟
                    final_delays_per_link[user_idx] = min(satellite_delay, max_delay)
                    
                    # 清空累积数据
                    self.hvft_accumulated_data[user_idx] = 0
                    self.hvft_accumulation_rounds[user_idx] = 0
                    self.hvft_satellite_transmission[user_idx] = 0
                    
                    # 更新卫星状态
                    self.link_satellite_status[user_idx] = 1
                else:
                    # 卫星容量不足，保持地面延迟
                    pass

        final_delays_per_link = np.minimum(final_delays_per_link, max_delay)
        self.current_link_delays = final_delays_per_link  # Update for observation space

        # 延迟惩罚，考虑Others流量优先级
        delay_sum = np.sum(final_delays_per_link)
        delay_mean = np.mean(final_delays_per_link)
        
        # 分别计算HVFT和Others的延迟
        hvft_mask = (self.user_traffic_types == 1)
        others_mask = (self.user_traffic_types == 0)
        
        hvft_delays = final_delays_per_link[hvft_mask]
        others_delays = final_delays_per_link[others_mask]
        
        # 计算加权延迟，Others流量权重更高
        if np.any(others_mask):
            others_delay_mean = np.mean(others_delays)
            others_weight = 0.6  # Others流量权重60%
        else:
            others_delay_mean = 0
            others_weight = 0
            
        if np.any(hvft_mask):
            hvft_delay_mean = np.mean(hvft_delays)
            hvft_weight = 0.4  # HVFT流量权重40%
        else:
            hvft_delay_mean = 0
            hvft_weight = 0
            
        # 加权平均延迟，Others流量影响更大
        weighted_delay_mean = others_weight * others_delay_mean + hvft_weight * hvft_delay_mean
        
        # 基于图片1和2设计的奖励函数：r_i^(t+1) = w_i^(t) C_i^(t) - Σ_{k∈O_i^(t+1)} π_{i→k}^(t)
        
        # 计算每个用户的信道容量（频谱效率）
        bandwidth = self.bandwidth
        noise_power = self.sigma ** 2
        
        # 计算当前状态下的信道容量
        signal_powers = np.diag(self.H) * adjusted_ground_tx_power
        H_interference = self.H - np.diag(np.diag(self.H))
        interference_powers = np.dot(H_interference, adjusted_ground_tx_power)
        SINR = signal_powers / (interference_powers + noise_power)
        channel_capacities = bandwidth * np.log2(1 + SINR)
        

        
        # 计算每个用户的奖励
        user_rewards = np.zeros(self.num_users)
        
        for i in range(self.num_users):
            # 第一部分：直接贡献 C_i^(t)（去掉权重）
            direct_contribution = channel_capacities[i]
            
            # 第二部分：惩罚项 Σ_{k∈O_i^(t+1)} π_{i→k}^(t)
            penalty_sum = 0.0
            
            # 找到被智能体i干扰的邻居 O_i^(t+1)
            for k in range(self.num_users):
                if k != i and self.H[k, i] > 0:  # 如果k被i干扰
                    # 计算无干扰时的信道容量 C_{k\i}^(t)
                    interference_without_i = interference_powers[k] - self.H[k, i] * adjusted_ground_tx_power[i]
                    SINR_without_i = signal_powers[k] / (interference_without_i + noise_power)
                    capacity_without_i = bandwidth * np.log2(1 + SINR_without_i)
                    
                    # 计算外部效应价格 π_{i→k}^(t) = C_{k\i}^(t) - C_k^(t)（去掉权重）
                    external_effect_price = capacity_without_i - channel_capacities[k]
                    penalty_sum += external_effect_price

            # 最终奖励：直接贡献 - 惩罚项
            user_rewards[i] = direct_contribution - penalty_sum
        
        # 系统级奖励：所有用户奖励的平均值
        reward = np.mean(user_rewards)
        
        # 使用bandwidth归一化以提高数值稳定性
        reward /= bandwidth
        # 记录训练历史
        self.cost_hist.append(reward)

        done = False
        if self.time_step > self.T - 1:
            done = True

        # 保存更新后的功率分配用于观察空间
        self.current_power_allocation = adjusted_ground_tx_power.copy()

        return (final_delays_per_link, SINR_per_link, pdr_per_link, reward,
                done, throughput_HVFT_total, throughput_others_total, throughput_all_total)

    def _get_downlink_obs(self, channel_obs, user_traffic_types, transmission_delays, rtt_delays, link_satellite_status, log2_sinr):
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

        # 构建HVFT累积信息
        hvft_accumulation_info = []
        for i in range(self.num_users):
            if user_traffic_types[i] == 1:  # HVFT用户
                # 归一化累积数据量
                normalized_accumulated_data = self.hvft_accumulated_data[i] / (self.hvft_load_mb * self.hvft_delay_tolerance)
                # 归一化累积回合数
                normalized_accumulation_rounds = self.hvft_accumulation_rounds[i] / self.hvft_delay_tolerance
                # 传输决策
                transmission_decision = self.hvft_transmission_decisions[i]
                # 卫星传输标记
                satellite_transmission = self.hvft_satellite_transmission[i]
                
                hvft_accumulation_info.extend([normalized_accumulated_data, normalized_accumulation_rounds, 
                                             transmission_decision, satellite_transmission])
            else:  # Others用户
                hvft_accumulation_info.extend([0, 0, 0, 0])  # 非HVFT用户，HVFT信息为0
        
        hvft_accumulation_info = np.array(hvft_accumulation_info)

        # 添加更新后的功率分配信息
        updated_power_allocation = []
        for i in range(self.num_users):
            if self.user_traffic_types[i] == 1 and self.hvft_transmission_decisions[i] == 0:
                # HVFT用户且决定不传输
                if self.link_satellite_status[i] == 1:
                    # 如果选择了卫星传输，保持原始功率（因为会通过卫星传输）
                    updated_power_allocation.append(self.current_power_allocation[i])
                else:
                    # 没有选择卫星传输，功率为0
                    updated_power_allocation.append(0.0)
            else:
                # 其他用户保持原始功率分配
                updated_power_allocation.append(self.current_power_allocation[i])
        
        # 归一化功率值
        normalized_updated_power = np.array(updated_power_allocation) / self.max_pwr_perplant
        
        # 构建增强的观察空间
        enhanced_obs = np.hstack((
            channel_obs, user_traffic_types, transmission_delays, rtt_delays, 
            link_satellite_status, log2_sinr, hvft_accumulation_info,
            normalized_updated_power  # 新增：更新后的功率分配
        ))
        
        return enhanced_obs

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
    
    def _build_action_from_power_and_satellite(self, powers, satellite_mask, hvft_decisions=None):
        """从功率、卫星选择和HVFT决策构建动作"""
        # 将功率归一化到[-1, 1]
        normalized_powers = (powers / self.max_pwr_perplant) * 2 - 1
        normalized_powers = np.clip(normalized_powers, -1, 1)
        
        # 构建动作 [功率, 是否用卫星, HVFT决策]
        action = np.zeros((self.num_users, 3))
        action[:, 0] = normalized_powers  # 功率
        action[:, 1] = satellite_mask.astype(float)  # 卫星选择
        
        # HVFT决策（如果提供）
        if hvft_decisions is not None:
            action[:, 2] = hvft_decisions.astype(float)  # HVFT决策
        else:
            # 默认HVFT决策（环境自动决策）
            action[:, 2] = np.zeros(self.num_users)
        
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

        # Action space for each user: (num_users, 3)
        # 第一列为功率（-1~1），第二列为是否用卫星（-1~1，step中round到0/1），第三列为HVFT传输决策（-1~1，step中round到0/1）
        self.action_space = spaces.Box(low=np.array([[-1, -1, -1]] * num_users), high=np.array([[1, 1, 1]] * num_users))
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
        if len(action) > self.num_users * 3:
            # 如果有约束维度，提取lambd
            lambd = action[-self.constraint_dim:]
            action = action[:-self.constraint_dim]
        else:
            lambd = np.array([0.0])  # 默认约束值
            
        # 确保action是1D数组，然后重塑为(num_users, 3)
        if action.ndim == 1:
            if len(action) != self.num_users * 3:
                raise ValueError(f"Action length {len(action)} does not match expected {self.num_users * 3}")
            action = action.reshape((self.num_users, 3))
        elif action.ndim == 2:
            if action.shape != (self.num_users, 3):
                raise ValueError(f"Action shape {action.shape} does not match expected ({self.num_users}, 3)")
        else:
            raise ValueError(f"Action has unexpected dimensions: {action.ndim}")
            
        power_raw = action[:, 0]
        is_satellite_raw = action[:, 1]
        hvft_decision_raw = action[:, 2]
        # 缩放功率
        power_scaled = self.scale_power(power_raw)
        # 是否用卫星，强制为0或1
        is_satellite = np.clip(np.round(is_satellite_raw), 0, 1).astype(int)
        hvft_decision = np.clip(np.round(hvft_decision_raw), 0, 1).astype(int)

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
            self._update_control_states_downlink(actual_ground_tx_power, satellite_mask, lambd, action_penalty, hvft_decision)

        # 更新状态
        self.H, channel_states = self.sample_graph()
        self.current_state = channel_states
        log2_sinr = np.log2(1 + np.maximum(SINR_all_links, 1e-6))
        states_obs = self._get_downlink_obs(channel_states, self.user_traffic_types,
                                            self.transmission_delays, self.rtt_delays, self.link_satellite_status, log2_sinr)
        
        # 记录数据到CSV文件（如果logger存在）
        if hasattr(self, 'logger') and self.logger is not None:
            # 记录所有数据到单个CSV文件，使用obs中的真实数据
            self.logger.log_step_data(
                self.time_step,
                self,
                power_scaled,
                is_satellite,
                obs_data=states_obs
            )
            
        # 计算info字段
        sinr_sum = np.sum(SINR_all_links)
        sinr_mean = np.mean(SINR_all_links)
        sinr_min = np.min(SINR_all_links)
        log_sinr_sum = np.sum(np.log2(1 + np.maximum(SINR_all_links, 1e-6)))
        delay_sum = np.sum(delays_per_link)
        delay_mean = np.mean(delays_per_link)
        power_efficiency = 1.0 - (total_power / (self.num_users * self.max_pwr_perplant))
        
        # 计算Others流量延迟统计
        others_mask = (self.user_traffic_types == 0)
        hvft_mask = (self.user_traffic_types == 1)
        
        if np.any(others_mask):
            others_delays = delays_per_link[others_mask]
            others_delay_sum = np.sum(others_delays)
            others_delay_mean = np.mean(others_delays)
            others_delay_min = np.min(others_delays)
            others_delay_max = np.max(others_delays)
            others_delay_std = np.std(others_delays)
            others_count = int(np.sum(others_mask))
        else:
            others_delay_sum = 0.0
            others_delay_mean = 0.0
            others_delay_min = 0.0
            others_delay_max = 0.0
            others_delay_std = 0.0
            others_count = 0
            
        # 计算HVFT流量延迟统计 - 排除不传输的HVFT链路
        if np.any(hvft_mask):
            hvft_delays = delays_per_link[hvft_mask]
            hvft_transmission_mask = (self.hvft_transmission_decisions == 1)[hvft_mask]
            
            # 只考虑实际传输的HVFT链路
            if np.any(hvft_transmission_mask):
                hvft_transmitted_delays = hvft_delays[hvft_transmission_mask]
                hvft_delay_sum = np.sum(hvft_delays)  # 总延迟包括所有HVFT
                hvft_delay_mean = np.mean(hvft_delays)  # 平均延迟包括所有HVFT
                hvft_delay_min = np.min(hvft_transmitted_delays)  # 最小延迟只考虑传输的
                hvft_delay_max = np.max(hvft_delays)  # 最大延迟包括所有HVFT
                hvft_delay_std = np.std(hvft_delays)  # 标准差包括所有HVFT
                hvft_count = int(np.sum(hvft_mask))
            else:
                # 没有HVFT传输，最小延迟设为0
                hvft_delay_sum = np.sum(hvft_delays)
                hvft_delay_mean = np.mean(hvft_delays)
                hvft_delay_min = 0.0
                hvft_delay_max = np.max(hvft_delays)
                hvft_delay_std = np.std(hvft_delays)
                hvft_count = int(np.sum(hvft_mask))
        else:
            hvft_delay_sum = 0.0
            hvft_delay_mean = 0.0
            hvft_delay_min = 0.0
            hvft_delay_max = 0.0
            hvft_delay_std = 0.0
            hvft_count = 0
        
        info = {
            "sinr_sum": float(sinr_sum),
            "sinr_mean": float(sinr_mean),
            "sinr_min": float(sinr_min),
            "log_sinr_sum": float(log_sinr_sum),
            "delay_sum": float(delay_sum),
            "delay_mean": float(delay_mean),
            "total_power": float(total_power),
            "power_efficiency": float(power_efficiency),  # 添加功率效率
            "ground_users_count": int(np.sum(ground_mask)),
            "satellite_users_count": int(np.sum(satellite_mask)),
            # 添加custom_callback需要的字段
            "throughput_hvft": float(throughput_HVFT),
            "throughput_others": float(throughput_others),
            "throughput_all_total": float(throughput_all),
            "link_pdr_avg": float(np.mean(pdr_all_links)),
            # 添加HVFT相关统计信息
            "hvft_accumulated_data_total": float(np.sum(self.hvft_accumulated_data)),
            "hvft_accumulation_rounds_avg": float(np.mean(self.hvft_accumulation_rounds)),
            "hvft_transmission_decisions_sum": int(np.sum(self.hvft_transmission_decisions)),
            "hvft_satellite_transmission_sum": int(np.sum(self.hvft_satellite_transmission)),
            # 添加Others流量延迟详细统计
            "others_delay_sum": float(others_delay_sum),
            "others_delay_mean": float(others_delay_mean),
            "others_delay_min": float(others_delay_min),
            "others_delay_max": float(others_delay_max),
            "others_delay_std": float(others_delay_std),
            "others_count": int(others_count),
            # 添加HVFT流量延迟详细统计
            "hvft_delay_sum": float(hvft_delay_sum),
            "hvft_delay_mean": float(hvft_delay_mean),
            "hvft_delay_min": float(hvft_delay_min),
            "hvft_delay_max": float(hvft_delay_max),
            "hvft_delay_std": float(hvft_delay_std),
            "hvft_count": int(hvft_count)
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

        # (num_users, 3)
        self.action_space = spaces.Box(low=-np.ones((num_users, 3)), high=np.ones((num_users, 3)))
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
        if len(action) > self.num_users * 3:
            # 如果有约束维度，提取lambd
            lambd = action[-self.constraint_dim:]
            action = action[:-self.constraint_dim]
        else:
            lambd = np.array([0.0])  # 默认约束值
            
        # 确保action是1D数组，然后重塑为(num_users, 3)
        if action.ndim == 1:
            if len(action) != self.num_users * 3:
                raise ValueError(f"Action length {len(action)} does not match expected {self.num_users * 3}")
            action = action.reshape((self.num_users, 3))
        elif action.ndim == 2:
            if action.shape != (self.num_users, 3):
                raise ValueError(f"Action shape {action.shape} does not match expected ({self.num_users}, 3)")
            else:
                raise ValueError(f"Action has unexpected dimensions: {action.ndim}")
            
        power_raw = action[:, 0]
        is_satellite_raw = action[:, 1]
        hvft_decision_raw = action[:, 2]
        
        # 缩放功率
        power_scaled = self.scale_power(power_raw)
        # 是否用卫星，强制为0或1
        is_satellite = np.clip(np.round(is_satellite_raw), 0, 1).astype(int)
        hvft_decision = np.clip(np.round(hvft_decision_raw), 0, 1).astype(int)

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
            self._update_control_states_downlink(actual_ground_tx_power, satellite_mask, lambd, action_penalty, hvft_decision)

        # 更新状态
        self.H, channel_states = self.sample_graph()
        self.current_state = channel_states
        log2_sinr = np.log2(1 + np.maximum(SINR_all_links, 1e-6))
        states_obs = self._get_downlink_obs(channel_states, self.user_traffic_types,
                                            self.transmission_delays, self.rtt_delays, self.link_satellite_status, log2_sinr)
        
        # 记录数据到CSV文件（如果logger存在）
        if hasattr(self, 'logger') and self.logger is not None:
            # 记录所有数据到单个CSV文件，使用obs中的真实数据
            self.logger.log_step_data(
                self.time_step,
                self,
                power_scaled,
                is_satellite,
                obs_data=states_obs
            )
            
        # 计算info字段
        sinr_sum = np.sum(SINR_all_links)
        sinr_mean = np.mean(SINR_all_links)
        sinr_min = np.min(SINR_all_links)
        log_sinr_sum = np.sum(np.log2(1 + np.maximum(SINR_all_links, 1e-6)))
        delay_sum = np.sum(delays_per_link)
        delay_mean = np.mean(delays_per_link)
        power_efficiency = 1.0 - (total_power / (self.num_users * self.max_pwr_perplant))
        
        # 计算Others流量延迟统计
        others_mask = (self.user_traffic_types == 0)
        hvft_mask = (self.user_traffic_types == 1)
        
        if np.any(others_mask):
            others_delays = delays_per_link[others_mask]
            others_delay_sum = np.sum(others_delays)
            others_delay_mean = np.mean(others_delays)
            others_delay_min = np.min(others_delays)
            others_delay_max = np.max(others_delays)
            others_delay_std = np.std(others_delays)
            others_count = int(np.sum(others_mask))
        else:
            others_delay_sum = 0.0
            others_delay_mean = 0.0
            others_delay_min = 0.0
            others_delay_max = 0.0
            others_delay_std = 0.0
            others_count = 0
            
        # 计算HVFT流量延迟统计 - 排除不传输的HVFT链路
        if np.any(hvft_mask):
            hvft_delays = delays_per_link[hvft_mask]
            hvft_transmission_mask = (self.hvft_transmission_decisions == 1)[hvft_mask]
            
            # 只考虑实际传输的HVFT链路
            if np.any(hvft_transmission_mask):
                hvft_transmitted_delays = hvft_delays[hvft_transmission_mask]
                hvft_delay_sum = np.sum(hvft_delays)  # 总延迟包括所有HVFT
                hvft_delay_mean = np.mean(hvft_delays)  # 平均延迟包括所有HVFT
                hvft_delay_min = np.min(hvft_transmitted_delays)  # 最小延迟只考虑传输的
                hvft_delay_max = np.max(hvft_delays)  # 最大延迟包括所有HVFT
                hvft_delay_std = np.std(hvft_delays)  # 标准差包括所有HVFT
                hvft_count = int(np.sum(hvft_mask))
            else:
                # 没有HVFT传输，最小延迟设为0
                hvft_delay_sum = np.sum(hvft_delays)
                hvft_delay_mean = np.mean(hvft_delays)
                hvft_delay_min = 0.0
                hvft_delay_max = np.max(hvft_delays)
                hvft_delay_std = np.std(hvft_delays)
                hvft_count = int(np.sum(hvft_mask))
        else:
            hvft_delay_sum = 0.0
            hvft_delay_mean = 0.0
            hvft_delay_min = 0.0
            hvft_delay_max = 0.0
            hvft_delay_std = 0.0
            hvft_count = 0

        info = {
            "sinr_sum": float(sinr_sum),
            "sinr_mean": float(sinr_mean),
            "sinr_min": float(sinr_min),
            "log_sinr_sum": float(log_sinr_sum),
            "delay_sum": float(delay_sum),
            "delay_mean": float(delay_mean),
            "total_power": float(total_power),
            "power_efficiency": float(power_efficiency),  # 添加功率效率
            "satellite_users_count": int(np.sum(satellite_mask)),
            # 添加custom_callback需要的字段
            "throughput_hvft": float(throughput_HVFT),
            "throughput_others": float(throughput_others),
            "throughput_all_total": float(throughput_all),
            "link_pdr_avg": float(np.mean(pdr_all_links)),
            # 添加HVFT相关统计信息
            "hvft_accumulated_data_total": float(np.sum(self.hvft_accumulated_data)),
            "hvft_accumulation_rounds_avg": float(np.mean(self.hvft_accumulation_rounds)),
            "hvft_transmission_decisions_sum": int(np.sum(self.hvft_transmission_decisions)),
            "hvft_satellite_transmission_sum": int(np.sum(self.hvft_satellite_transmission)),
            # 添加Others流量延迟详细统计
            "others_delay_sum": float(others_delay_sum),
            "others_delay_mean": float(others_delay_mean),
            "others_delay_min": float(others_delay_min),
            "others_delay_max": float(others_delay_max),
            "others_delay_std": float(others_delay_std),
            "others_count": int(others_count),
            # 添加HVFT流量延迟详细统计
            "hvft_delay_sum": float(hvft_delay_sum),
            "hvft_delay_mean": float(hvft_delay_mean),
            "hvft_delay_min": float(hvft_delay_min),
            "hvft_delay_max": float(hvft_delay_max),
            "hvft_delay_std": float(hvft_delay_std),
            "hvft_count": int(hvft_count)
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
            # 假设Others用户优先使用卫星（优先级更高）
            satellite_mask = np.zeros(self.num_users, dtype=bool)
            others_users = (self.user_traffic_types == 0)  # Others用户优先
            satellite_mask[others_users] = True
            # 再随机选择一些HVFT用户
            hvft_users = ~others_users
            if np.any(hvft_users):
                hvft_indices = np.where(hvft_users)[0]
                num_additional = min(2, len(hvft_indices))
                additional_indices = np.random.choice(hvft_indices, num_additional, replace=False)
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