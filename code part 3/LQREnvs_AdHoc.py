###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################
# 
# 修改说明：
# 1. 实现了基于Jakes衰落模型的时变信道状态定义
#    - 信道增益 g_i->j^(t) = |h_i->j^(t)|^2 * α_i->j
#    - 小尺度衰落 h_i->j^(t) = ρ * h_i->j^(t-1) + √(1 - ρ^2) * e_i->j^(t)
#    - 相关系数 ρ = J_0(2π * f_d * T)，其中f_d = v * f_c / c
# 
# 2. 重新设计了奖励函数，基于图片1和2的算法原理：
#    - 奖励公式：r_i^(t+1) = C_i^(t) - Σ_{k∈O_i^(t+1)} π_{i→k}^(t)
#    - 直接贡献：C_i^(t) = B * log2(1 + SINR_i)
#    - 惩罚项：π_{i→k}^(t) = C_{k\i}^(t) - C_k^(t)
#    - 其中C_{k\i}^(t)表示没有智能体i干扰时链路k的容量
# 
# 3. 移除了权重相关的代码，专注于直接贡献和惩罚机制
#

import numpy as np
import pdb
import scipy
import scipy.linalg
import scipy.io
import control
import gym
import math
import random

from scipy.stats import bernoulli
from scipy.special import j0
from gym import spaces
from gym.utils import seeding
from WirelessNets import *
from link_data_logger import LinkDataLogger


class LQR_Env(gym.Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, mu=1, T=40,
                 gamma=0.99, pl=2., pp=5., p0=1., num_features=1, scaling=True,
                 ideal_comm=False):
        super(LQR_Env, self).__init__()

         # dimensions
        self.num_features = num_features
        self.state_dim = num_users**2  # 只保留信道状态维度
        self.action_dim = num_users
        self.constraint_dim = constraint_dim
        self.channel_state_dim = num_users**2
        # 移除控制状态相关维度

        # 更新观察状态维度：信道矩阵 + 功率分配 + 接收干扰 + 造成干扰 + 信道速率
        # = num_users^2 + num_users + num_users + num_users + num_users = num_users^2 + 4*num_users
        self.enhanced_state_dim = self.channel_state_dim + num_features * num_users
        # using different seeds across different realizations of the WCS
        self.np_random = []
        self.seed()
        
        # system parameters
        self.num_users = num_users
        self.T = T
        self.max_pwr_perplant = pp
        self.p0 = p0

        # wireless network parameters
        self.mu = mu  # parameter for distribution of channel states (fast fading)
        self.sigma = 1.
        self.n_transmitting = np.rint(num_users/3).astype(np.int32)  # number of plants transmitting at a given time
        self.gamma = gamma
        self.upperbound = upperbound
        self.pl = pl  # path loss
        self.L = L  # build_adhoc_network(num_users, pl)
        self.assign = assign

        self.batch_size = 1
        self.cost_hist = []
        self.H = 0  # interference matrix

        self.max_control_state = 50.
        self.max_cost = self.max_control_state**2
        self.control_actions = []

        # open AI gym structure: separate between AdHoc, MultiCell, UpLink, Downlink envs!
        self.action_space = []
        # 使用新的增强观察状态维度，值范围设为[0, 1]因为我们会进行归一化
        self.observation_space = spaces.Box(low=np.zeros(self.enhanced_state_dim),
                                            high=np.ones(self.enhanced_state_dim))
        self.scaling = scaling

        # 移除所有LQR相关的矩阵和参数
        
        # HSR相关参数（高铁场景）
        self.f = 930 * 1e6  # 载波频率 (Hz) - 930 MHz
        self.A_b = 30  # 基站天线高度 (m)
        self.A_m = 3   # 移动台天线高度 (m)
        self.t = 1     # 时间参数
        self.v = 138   # 速度 (km/h)
        self.d = 400   # 距离参数 (m)
        self.R = 3000  # 半径参数 (m)
        self.choice = 1  # Hata模型选择
        self.bandwidth = 10e6  # 带宽 (Hz)
        
        # Jakes衰落模型参数
        self.time_slot_duration = 1.0  # 时间槽持续时间（秒）
        self.c_light = 3e8  # 光速
        
        # 初始化信道状态变量
        self.h_small_scale = None  # 小尺度衰落分量
        self.h_small_scale_prev = None  # 前一时刻的小尺度衰落
        self.alpha_large_scale = None  # 大尺度衰落分量（路径损耗）
        self.channel_initialized = False  # 信道是否已初始化

        self.Data_chunk_size = 3000 * 8

        self.current_state = self.sample(batch_size=1)

        # to save training data
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
        
        # 数据记录相关属性
        self.logger = None  # 数据记录器
        self.env_id = 0     # 子环境ID
        self.current_power_allocation = np.zeros(self.num_users)  # 当前功率分配
        
        # 新的观察状态相关属性
        self.channel_rates = np.zeros(self.num_users)  # 信道速率
        self.channel_gains = np.zeros(self.num_users)  # 信道增益
        
        # 延迟和吞吐量相关属性
        self.traffic_loads = np.full(self.num_users, 3000*8)  # 流量负载 (Mbps) - 每个用户固定为3000*8
        self.rtt_delays = np.zeros(self.num_users)  # RTT延迟
        self.link_delays = np.zeros(self.num_users)  # 链路延迟
        self.throughput_per_user = np.zeros(self.num_users)  # 每个用户的吞吐量
        
    def disc_cost(self, cost_vec):
        T = np.size(cost_vec)
        cost_discounted = np.zeros(T)
        cost_discounted[-1] = cost_vec[-1]

        # calculating discounted return backwards (step by step)
        for k in range(1, T):
            cost_discounted[T - 1 - k] = cost_vec[T - 1 - k] + self.gamma * cost_discounted[T - 1 - k + 1]

        return cost_discounted

    def disc_constraint(self, cost_vec):
        T = np.size(cost_vec)
        if cost_vec.ndim > 1:
            T, constraint_dim = cost_vec.shape
            cost_discounted = np.zeros((T, constraint_dim))
        else:
            cost_discounted = np.zeros(T)
        cost_discounted[-1] = cost_vec[-1]

        # calculating discounted return backwards (step by step)
        for k in range(1, T):
            cost_discounted[T - 1 - k, :] = cost_vec[T - 1 - k, :] + self.gamma * cost_discounted[T - 1 - k + 1, :]

        return cost_discounted

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Normalizing H_t / GSO
    @staticmethod
    def normalize_gso(S):
        # norms = np.linalg.norm(S, ord=2, axis=(1, 2))
        norm = np.linalg.norm(S, ord=2, axis=None)
        Snorm = S / norm  # norms[:, None, None]
        return Snorm

    @staticmethod
    def normalize_inputs(inputs):
        input2 = inputs - inputs.mean(axis=1).reshape(-1, 1)
        return input2

    def normalize_obs(self, obs: np.ndarray, mean, var, epsilon=1e-8, clip_obs=10.) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        channel_mean = mean[0]
        plants_mean = mean[1]
        interval_mean = mean[2]
        channel_var = var[0]
        plants_var = var[1]
        interval_var = var[2]
        # obs = np.clip((obs - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs)
        channel_obs = obs[:self.num_users ** 2]
        channel_obs = np.clip((channel_obs - channel_mean) / np.sqrt(channel_var + epsilon), -clip_obs, clip_obs)
        obs_aux = obs[self.num_users ** 2:].reshape(-1, self.num_features)
        plants_obs = obs_aux[:, 0]
        plants_obs = np.clip((plants_obs - plants_mean) / np.sqrt(plants_var + epsilon), -clip_obs, clip_obs)
        interval_obs = obs_aux[:, 1]
        interval_obs = np.clip((interval_obs - interval_mean) / np.sqrt(interval_var + epsilon), -clip_obs, clip_obs)
        obs_aux[:, 0] = plants_obs
        obs_aux[:, 1] = interval_obs
        obs_aux = obs_aux.reshape(-1)
        obs = np.hstack((channel_obs.flatten(), obs_aux.flatten()))
        return obs

    # 移除控制状态归一化方法，因为不再需要

    # packet delivery rate: no interference (已去除PDR约束)
    @staticmethod
    def packet_delivery_rate(snr_value):
        # 去除PDR约束，返回固定值1.0表示100%传输成功率
        return 1.0

    def doppler_shift_effect(self, t, v, d, R):
        """计算多普勒频移效应"""
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
            # 如果L为None，创建一个默认的全连接矩阵
            default_L = np.ones((self.num_users, self.num_users))
            np.fill_diagonal(default_L, 0)
            alpha_matrix = alpha_matrix * default_L
        
        return alpha_matrix

    def update_jakes_channel_model(self):
        if not self.channel_initialized:
            self.initialize_jakes_channel_model()

        rho = self.calculate_correlation_coefficient()

        real_innovation = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_users))
        imag_innovation = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_users))
        innovation = real_innovation + 1j * imag_innovation

        self.h_small_scale = rho * self.h_small_scale_prev + np.sqrt(1 - rho**2) * innovation
        channel_gains = np.abs(self.h_small_scale)**2 * self.alpha_large_scale
        self.h_small_scale_prev = self.h_small_scale.copy()
        
        return channel_gains

    # interference with Doppler effect
    def interference_packet_delivery_rate(self, H, actions, traffic_loads=None):
        actions_vec = actions[:, None]
        # 假设使用默认带宽值
        bandwidth = getattr(self, 'bandwidth', 10e6)
        
        # numerator: diagonal elements (hii) - 信号功率 S
        H_diag = np.diag(H)
        signal_powers = np.multiply(H_diag, actions)

        # 计算多普勒效应
        doppler_effect_coefficent, _ = self.doppler_shift_effect(
            getattr(self, 't', 1), 
            getattr(self, 'v', 138), 
            getattr(self, 'd', 400), 
            getattr(self, 'R', 3000)
        )

        # 计算载波间干扰(ICI)
        x_values = np.linspace(-1, 1, 1000)
        integral_result = np.trapz((1 - abs(x_values)) * j0(2 * np.pi * doppler_effect_coefficent * x_values), x_values)
        W_ICI = 1 - integral_result
        W_ICI = max(0.0, W_ICI)
        # denominator: off-diagonal elements with Doppler effect - 干扰功率 I + 噪声功率 N
        H_interference = (H - np.diag(np.diag(H))).transpose()
        interference_plus_noise = (np.dot(H_interference * (1 + W_ICI), actions_vec) + self.sigma ** 2).flatten()

        # 确保分母不为0
        interference_plus_noise[interference_plus_noise < 1e-9] = 1e-9
        # SINR = S / (N + I)
        SINR = signal_powers / interference_plus_noise
        # 计算干扰信息：其他设备对本设备的干扰和本设备对所有其他设备的干扰
        num_users = len(actions)
        interference_received = np.zeros(num_users)
        interference_caused_to_all = np.zeros(num_users)
        
        for i in range(num_users):
            # 计算其他设备对设备i的干扰
            for j in range(num_users):
                if i != j and actions[j] > 0:  # 只考虑有发送功率的设备
                    interference_received[i] += H[i, j] * actions[j] * (1 + W_ICI)
            
            # 计算设备i对所有其他设备的干扰（全体干扰）
            if actions[i] > 0:  # 只有当设备i有发送功率时
                for j in range(num_users):
                    if i != j:  # 对所有其他设备
                        interference_caused_to_all[i] += H[j, i] * actions[i] * (1 + W_ICI)
        
        # 根据香农公式计算信道容量：C = B * log2(1 + SINR)
        channel_rate = bandwidth * np.log2(1 + SINR)

        rate_loss_suffered = np.zeros(self.num_users)
        delay_loss_suffered = np.zeros(self.num_users)
        
        # 噪声功率 N
        noise_power = self.sigma ** 2

        if traffic_loads is None:
            traffic_loads = np.full(self.num_users, 3000 * 8)
        
        # 对每个用户计算其造成的干扰和遭受的干扰
        for i in range(self.num_users):
            if actions[i] > 0:  # 只考虑有发送功率的用户
                signal_power_i = signal_powers[i]
                
                # 用户j在没有用户i干扰时的SINR = S/N
                sinr_without_interference = signal_power_i / noise_power
                
                # 计算容量损失 = B log2(1 + S/N) - B log2(1 + S/(N+I))
                capacity_without_interference = bandwidth * np.log2(1 + sinr_without_interference)
                capacity_with_interference = channel_rate[i]
                capacity_loss = max(capacity_without_interference - capacity_with_interference, 0)
                
                # 计算延迟损失 = traffic_load/(channel_rate - capacity_loss) - traffic_load/channel_rate
                if capacity_with_interference > 0:
                    delay_without_interference = traffic_loads[i] / capacity_without_interference
                    delay_with_interference = traffic_loads[i] / capacity_with_interference
                    # 有干扰的延迟 - 无干扰的延迟  
                    delay_loss = max(delay_with_interference - delay_without_interference, 0)
                else:
                    delay_loss = 0
                
                # 累计每个用户遭受的容量损失和延迟损失
                rate_loss_suffered[i] = capacity_loss
                delay_loss_suffered[i] = delay_loss
                

        self.rate_loss_suffered_per_user = rate_loss_suffered
        self.delay_loss_suffered_per_user = delay_loss_suffered

        pdr = np.ones(self.num_users)
        return SINR, pdr, channel_rate, delay_loss_suffered

    def set_logger(self, logger):
        """设置数据记录器"""
        self.logger = logger

    def set_env_id(self, env_id):
        """设置子环境ID"""
        self.env_id = env_id

    def hsr_rtt_delay(self, num_users):
        """计算高铁RTT延迟"""
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

    def link_delay(self, throughput_per_link_rate, traffic_loads_per_link, rtt_delays_current=None):
        """计算链路延迟"""
        max_delay = 3.0  # 延迟值上限
        deliver_times_per_chunk = np.full(self.num_users, max_delay)
        
        for index in range(self.num_users):
            current_link_rate = throughput_per_link_rate[index]
            traffic_load = traffic_loads_per_link[index]
            
            if current_link_rate < 1e-9:
                deliver_times_per_chunk[index] = max_delay  # 使用最大延迟值
            else:
                # 简化计算：直接使用传输延迟，不计入RTT
                # 传输延迟 = 流量负载 / 信道速率
                transmission_delay = traffic_load / max(current_link_rate, 1e-9)
                
                # 限制延迟值不超过上限
                deliver_times_per_chunk[index] = min(transmission_delay, max_delay)
        
        return deliver_times_per_chunk

    def Hata_PL(self, choice, f, A_b, A_m, t, v, d, R):
        """计算Hata路径损耗模型"""
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



    def get_observation_space_info(self):
        """获取观察空间的详细信息"""
        info = {
            'total_dim': self.enhanced_state_dim,
            'components': {
                'channel_matrix': self.channel_state_dim,  # num_users^2 (Jakes模型)
                'power_allocation': self.num_users,
                'delay_loss_suffered': self.num_users,
                'channel_rates': self.num_users
            },
            'is_one_dimensional': True,
            'value_range': [0.0, 1.0],
            'normalization': 'All components are normalized to [0,1] range',
            'channel_model': 'Jakes衰落模型 - 基于多普勒效应的时变信道'
        }
        return info



    def greedy_control_aware_scheduling(self, n_transmitting, control_states_obs_ca_pwr):
        # 简化版本：基于信道状态进行调度
        ca_pwr = np.zeros(self.num_users)
        # 选择信道状态最好的n_transmitting个用户
        channel_gains = np.diag(self.H)
        ind = np.argpartition(channel_gains, -n_transmitting)[-n_transmitting:]
        ca_pwr[ind] = 1.

        return ca_pwr

    def round_robin(self, n, last_idx):
        transmitting_plants = np.zeros(self.num_users)
        if (last_idx + n) >= self.num_users:
            n_under = self.num_users - last_idx
            n_over = last_idx + n - self.num_users
            transmitting_plants[-n_under:] = 1.
            transmitting_plants[:n_over] = 1.
            last_idx = n_over
        else:
            transmitting_plants[last_idx:last_idx + n] = 1.
            last_idx += n
        rr_pwr = transmitting_plants

        return rr_pwr, last_idx

    def wmmse(self, S):
        Pmax = self.p0
        h2 = np.copy(S)
        h = np.sqrt(h2)
        m = S.shape[1]
        N = S.shape[0]
        v = np.ones((N, m)) * np.sqrt(Pmax) / 2
        T = 100
        v2 = np.expand_dims(v ** 2, axis=2)

        u = (np.diagonal(h, axis1=1, axis2=2) * v) / (np.matmul(h2, v2)[:, :, 0] + self.sigma)
        w = 1 / (1 - u * np.diagonal(h, axis1=1, axis2=2) * v)
        N = 1000
        for n in np.arange(T):
            u2 = np.expand_dims(u ** 2, axis=2)
            w2 = np.expand_dims(w, axis=2)
            v = (w * u * np.diagonal(h, axis1=1, axis2=2)) / (np.matmul(np.transpose(h2, (0, 2, 1)), (w2 * u2)))[:, :, 0]
            v = np.minimum(np.sqrt(Pmax), np.maximum(0, v))
            v2 = np.expand_dims(v ** 2, axis=2)
            u = (np.diagonal(h, axis1=1, axis2=2) * v) / (np.matmul(h2, v2)[:, :, 0] + self.sigma)
            w = 1 / (1 - u * np.diagonal(h, axis1=1, axis2=2) * v)
        p = v ** 2
        return p

    # samples initial state and channel conditions
    def sample(self, batch_size):
        # graph, flat observation
        self.H, samples = self.sample_graph()
        return samples

    def sample_graph(self):  # downlink
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

    def sample_graph_uplink(self):  # uplink
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
        power_action /= 2  # [0, 1.]
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
        
        obs = self.sample(batch_size=1)
        self.current_state = obs
        self.time_step = 0

        # to save training data
        if self.cost_hist:
            cost_hist = np.array(self.cost_hist)
            cost_disc = self.disc_cost(cost_hist)
            ep_cost = cost_disc[0]
            self.ep_cost_hist.append(ep_cost)

        if self.Lagrangian_hist:
            cost_hist = np.array(self.Lagrangian_hist)
            cost_disc = self.disc_cost(cost_hist)
            ep_lagrangian = cost_disc[0]
            self.ep_Lagrangian.append(ep_lagrangian)

        if self.constraint_hist:
            constraint_hist = np.array(self.constraint_hist)
            constraint_disc = self.disc_constraint(constraint_hist)
            ep_constraint = constraint_disc[0]
            self.ep_constraint.append(ep_constraint)

        self.current_episode += 1
        self.cost_hist = []
        self.constraint_hist = []
        self.Lagrangian_hist = []

        return obs

    def _update_control_states_downlink(self, downlink_power, H, lambd, action_penalty):
        self.time_step += 1
        
        # 计算信道相关信息（包含延迟损失信息）
        SINR_per_link, pdr_per_link, channel_rates, delay_loss_suffered = self.interference_packet_delivery_rate(H, downlink_power, self.traffic_loads)
        
        # 计算延迟和吞吐量
        # 计算链路延迟（简化版本，不计入RTT）
        self.link_delays = self.link_delay(channel_rates, self.traffic_loads)
        
        # RTT延迟暂时设为0（不计入总延迟）
        self.rtt_delays = np.zeros(self.num_users)
        
        # 计算吞吐量（不考虑PDR约束，直接使用信道速率）
        self.throughput_per_user = channel_rates
        
        # 存储信道速率和其他信息用于观察状态
        self.channel_rates = channel_rates
        self.current_power_allocation = downlink_power.copy()
        self.channel_gains = np.diag(H)
        # 存储SINR和PDR供logger使用
        self.last_sinr_per_link = SINR_per_link
        self.last_pdr_per_link = pdr_per_link
        
        # 基于图片1和2设计的奖励函数：r_i^(t+1) = C_i^(t) - Σ_{k∈O_i^(t+1)} π_{i→k}^(t)
        
        # 计算每个用户的信道容量（频谱效率）
        bandwidth = self.bandwidth
        noise_power = self.sigma ** 2
        
        # 计算当前状态下的信道容量
        signal_powers = np.diag(self.H) * downlink_power
        H_interference = self.H - np.diag(np.diag(self.H))
        interference_powers = np.dot(H_interference, downlink_power)
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
                    interference_without_i = interference_powers[k] - self.H[k, i] * downlink_power[i]
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
        # 保存系统级指标
        self.system_avg_rate = float(np.mean(channel_rates))
        self.total_delay_loss_suffered = float(np.sum(delay_loss_suffered))
        self.current_reward = reward
        
        # 记录训练历史
        self.cost_hist.append(-reward)  # 记录负奖励作为cost
        self.constraint_hist.append(np.atleast_1d(action_penalty))
        self.Lagrangian_hist.append(-reward + np.dot(lambd, action_penalty))
        self.downlink_constraint_dualvar = np.dot(lambd, action_penalty)

        done = False
        if self.time_step > self.T - 1:
            done = True

        return SINR_per_link, pdr_per_link, reward, done

    def _update_control_states_uplink(self, control_actions, lambd, action_penalty):
        self.time_step += 1
        zerovec = np.zeros(self.num_users * self.p)

        # cost / reward -> plant states
        control_states = self.current_state[self.channel_state_dim:][None, :]
        cost_aux = np.multiply(control_states, control_states)

        # cost / reward -> control actions
        cost_aux2 = np.multiply(control_actions, control_actions)

        # in this case we assume ideal communications during downlink transmission
        control_estimate = control_actions 
        control_states = control_states.transpose()

        # new control states
        control_states = (np.dot(self.A, control_states) + np.dot(self.B, control_estimate) +
                          np.transpose(self.np_random.multivariate_normal(zerovec, self.W, size=self.batch_size)))
        control_states = np.transpose(control_states)
        control_states = np.clip(control_states, -self.max_control_state, self.max_control_state)
        control_states_obs = control_states + self.np_random.multivariate_normal(zerovec, self.Wobs,
                                                                                 size=self.batch_size)
        control_states_obs = np.clip(control_states_obs, -self.max_control_state, self.max_control_state)

        # total reward; Q and R are diagonal -> reward should be computed with ''previous'' control state, not current!
        one_step_cost = cost_aux.sum(axis=1)   # objective function
        one_step_cost2 = cost_aux.sum(axis=1)  + np.dot(lambd, action_penalty)

        self.cost_hist.append(one_step_cost)  # save cost, Lagrangian during training
        self.Lagrangian_hist.append(one_step_cost2)

        done = False
        if self.time_step > self.T - 1:
            done = True
            one_step_cost = cost_aux.sum(axis=1)

        one_step_reward = -1 * one_step_cost2[0] / (self.num_users*self.max_control_state)

        return control_states, control_states_obs, one_step_reward, done

    def _update_control_actions_uplink(self, control_states, uplink_power, H):

        # uplink delivery rate
        qq, _, channel_rate, _, _, _, _ = self.interference_packet_delivery_rate(H, uplink_power.flatten())
        qq = np.nan_to_num(qq)
        trials = np.transpose(bernoulli.rvs(qq))

        # updates state estimate
        control_states_obs = np.multiply(trials, control_states)

        # control actions computed at the remote controller
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        return control_actions

    def _update_control_actions_downlink(self, control_states_obs):

        # control actions computed at the remote controller
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        return control_actions

    def _test_init(self, T, batch_size=1):
        n_comps = 7  # GNN, MLP, Equal, Control-Aware, WMMSE, Round Robin, Random Access

        # cost per time step
        cost_matrices = [np.zeros(T) for _ in np.arange(n_comps)]

        # states
        dnn_state = self.sample(batch_size=1)[np.newaxis]
        init_states = [dnn_state for _ in np.arange(n_comps)]

        # interference matrix
        H, _ = self.sample_graph()
        init_interference = [H for _ in np.arange(n_comps)]

        # saving trajectory - 移除控制状态相关
        trajectories = [np.zeros((T, batch_size, self.channel_state_dim)) for _ in np.arange(n_comps)]

        # saving power allocation
        allocation_decisions = [np.zeros((T, batch_size, self.num_users)) for _ in np.arange(n_comps)]

        # 简化观察状态，只使用信道状态
        channel_states = dnn_state[:, :self.channel_state_dim]
        channel_states_obs = channel_states
        # 使用增强观察状态
        dnn_obs = self._get_downlink_obs(channel_states_obs.flatten())
        observations = [dnn_obs for _ in np.arange(n_comps)]

        # baselines
        eq_power = np.ones(self.num_users)
        eq_power *= self.p0
        last_idx = -1  # round robin

        return (cost_matrices, init_states, init_interference, trajectories, allocation_decisions, observations,
                eq_power, last_idx)

    def _get_uplink_obs(self, channel_obs, control_estimates):

        control_estimates_obs = self.control_plant_norm(control_estimates)
        obs = np.hstack((channel_obs, control_estimates_obs))

        return control_estimates, control_estimates_obs, obs

    def _get_downlink_obs(self, channel_obs, additional_features=None):
        """
        构建下行链路观察数据，包含信道增益、功率分配、干扰信息等
        注意：返回的是一维观察向量
        """
        obs_components = []
        
        # 1. 信道状态矩阵（完整信道信息）
        channel_flat = channel_obs.flatten()
        # 归一化信道状态
        max_channel = np.max(np.abs(channel_flat)) + 1e-8
        normalized_channel = channel_flat / max_channel
        obs_components.append(normalized_channel)
        
        # 2. 当前功率分配
        if hasattr(self, 'current_power_allocation'):
            normalized_power = self.current_power_allocation / (self.max_pwr_perplant + 1e-8)
            obs_components.append(normalized_power)
        else:
            obs_components.append(np.zeros(self.num_users))
        
        # 3. 延迟损失信息（干扰造成的延迟损失）
        if hasattr(self, 'delay_loss_suffered_per_user'):
            max_delay_loss = np.max(np.abs(self.delay_loss_suffered_per_user)) + 1e-8
            normalized_delay_loss = self.delay_loss_suffered_per_user / max_delay_loss
            obs_components.append(normalized_delay_loss)
        else:
            obs_components.append(np.zeros(self.num_users))
        
        # 5. 信道速率信息
        if hasattr(self, 'channel_rates'):
            bandwidth = getattr(self, 'bandwidth', 10e6)
            normalized_rates = self.channel_rates / (bandwidth + 1e-8)
            obs_components.append(normalized_rates)
        else:
            obs_components.append(np.zeros(self.num_users))
        
        # 如果有额外特征，添加进来
        if additional_features is not None:
            obs_components.append(additional_features.flatten())

        # 合并所有观察组件，返回一维向量
        obs = np.hstack(obs_components)
        
        # 验证观察向量维度
        expected_dim = self.enhanced_state_dim
        actual_dim = len(obs)
        if actual_dim != expected_dim:
            print(f"Warning: Observation dimension mismatch. Expected: {expected_dim}, Got: {actual_dim}")
            print(f"Components: channel({len(normalized_channel)}), power({self.num_users}), interference_recv({self.num_users}), interference_caused({self.num_users}), rates({self.num_users})")
        
        return obs

    def _test_step_uplink(self, states, states_obs, H, action, estimator):

        control_states = states[:, self.channel_state_dim:]  # [None, :]

        # uplink delivery rate
        qq, _, channel_rate, _, _, _, _ = self.interference_packet_delivery_rate(H, action.flatten())
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
        qq, _, channel_rate, _, _, _, _ = self.interference_packet_delivery_rate(H, downlink_action)
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

    # heuristics always satisfy instantaneous power constraints
    def test_equal_power(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        eq_power_downlink = np.ones(self.num_users)
        eq_power_downlink *= (upper_bound / self.num_users)

        for tt in range(T):

            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1) 

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, eq_power_downlink, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = eq_power_downlink
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_round_robin(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1,
                         last_idx=0):
        zerovec = np.zeros(self.num_users * self.p)

        for tt in range(T):
            rr_pwr, last_idx = self.round_robin(self.n_transmitting, last_idx)
            rr_pwr /= rr_pwr.sum()
            rr_pwr *= upper_bound
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1) 

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, rr_pwr, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = rr_pwr
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_control_aware(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)
        n_transmitting = self.n_transmitting

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            downlink_action = (self.greedy_control_aware_scheduling(n_transmitting, control_states_obs)).flatten()
            downlink_action /= downlink_action.sum()
            downlink_action *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, downlink_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = downlink_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_wmmse(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            downlink_action = (self.wmmse(H[None, :])).flatten()
            downlink_action /= downlink_action.sum()
            downlink_action *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1) 

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, downlink_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = downlink_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)
    
    def test_random_access(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)
        transmitting_plants = np.hstack((np.ones(self.n_transmitting), np.zeros(self.num_users - self.n_transmitting)))

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            ra_pwr = np.random.permutation(transmitting_plants)
            ra_pwr /= (ra_pwr.sum() + 1e-8)
            ra_pwr *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1) 
            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, ra_pwr, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = ra_pwr
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_mlp_inst_constraint(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1) 

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            dnn_obs = self._get_downlink_obs(channel_states_obs)
            allocation_action, _ = allocation_dnn.predict(dnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            allocation_action = np.clip(allocation_action, -1., 1.)
            allocation_action += 1.
            allocation_action = allocation_action / (allocation_action.sum() + 1e-8)
            allocation_action *= upper_bound

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, allocation_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = allocation_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_gnn_inst_constraint(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1) 

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            gnn_obs = self._get_downlink_obs(channel_states_obs)
            allocation_action, _ = allocation_gnn.predict(gnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            allocation_action = np.clip(allocation_action, -1., 1.)
            allocation_action += 1.
            allocation_action = allocation_action / (allocation_action.sum() + 1e-8)
            allocation_action *= upper_bound

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, allocation_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = allocation_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_mlp_hor_constraint(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        overall_constraint = upper_bound*T
        current_budget = 0.
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            dnn_obs = self._get_downlink_obs(channel_states_obs)
            allocation_action, _ = allocation_dnn.predict(dnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            allocation_action = np.clip(allocation_action, -1., 1.)
            allocation_action += 1.
            allocation_action *= 0.5  # now decisions are scaled to [0, 1]
            allocation_action *= self.max_pwr_perplant

            # Enforcing constraint over simulation horizon
            if current_budget + allocation_action.sum() > overall_constraint:
                remaining_budget = max(overall_constraint - current_budget, 0.)
                allocation_action /= (allocation_action.sum() + 1e-8)
                allocation_action *= remaining_budget
            current_budget += allocation_action.sum()

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, allocation_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = allocation_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_gnn_hor_constraint(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        overall_constraint = upper_bound * T
        current_budget = 0.
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            gnn_obs = self._get_downlink_obs(channel_states_obs)
            allocation_action, _ = allocation_gnn.predict(gnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            allocation_action = np.clip(allocation_action, -1., 1.)
            allocation_action += 1.
            allocation_action *= 0.5  # now decisions are scaled to [0, 1]
            allocation_action *= self.max_pwr_perplant

            # Enforcing constraint over simulation horizon
            if current_budget + allocation_action.sum() > overall_constraint:
                remaining_budget = max(overall_constraint - current_budget, 0.)
                allocation_action /= (allocation_action.sum() + 1e-8)
                allocation_action *= remaining_budget
            current_budget += allocation_action.sum()

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, allocation_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = allocation_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)


    def test(self, upper_bound, T, allocation_dnn, allocation_gnn, batch_size=1, test_type='output_constraint'):

        (cost_matrices, current_states, interference_matrices, states_matrices, allocation_decisions, observations,
         zerovec, eq_power, last_idx) = \
            self._test_init(T, batch_size=batch_size)

        [dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, capwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, rapwr_cost_mtx] = cost_matrices
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
        wmmsepwr_cost_mtx, wmmse_power, wmmse_states = self.test_wmmse(upper_bound, T, wmmse_state, wmmse_obs, wmmse_states,
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
            overall_constraint = T*upper_bound
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_hor_constraint(allocation_dnn, upper_bound, T,
                                                                                dnn_state, dnn_obs, dnn_states,
                                                                                dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_hor_constraint(allocation_gnn, upper_bound, T,
                                                                                gnn_state, gnn_obs, gnn_states,
                                                                                gnn_power, gnn_H, gnn_cost_mtx)


        return (dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, capwr_cost_mtx, rapwr_cost_mtx,
                dnn_power, gnn_power, equal_power, wmmse_power, rr_power, ca_power, ra_power,
                dnn_states, gnn_states, eq_states, wmmse_states, rr_states, ca_states, ra_states)

    def test_transf(self, upper_bound, T, allocation_gnn, batch_size=1, test_type='output_constraint'):

        (cost_matrices, current_states, interference_matrices, states_matrices, allocation_decisions, observations,
         zerovec, eq_power, last_idx) = \
            self._test_init(T, batch_size=batch_size)

        [dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, capwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, rapwr_cost_mtx] = cost_matrices
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
        wmmsepwr_cost_mtx, wmmse_power, wmmse_states = self.test_wmmse(upper_bound, T, wmmse_state, wmmse_obs, wmmse_states,
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


# ------------------------------------------- Downlink Environments ------------------------------------------ #
class LQRAdHocDownlink(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, T=40,
                 gamma=0.99, pl=2., pp=5., p0=1., num_features=1, scaling=False):

        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, T=T,
                         gamma=gamma, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        # Downlink: continuous allocation decisions
        self.action_space = spaces.Box(low=-np.ones(num_users), high=np.ones(num_users))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        # 初始化观察状态属性
        self.channel_gains = np.zeros(self.num_users)
        self.current_power_allocation = np.zeros(self.num_users)
        self.interference_received = np.zeros(self.num_users)
        self.interference_caused_to_all = np.zeros(self.num_users)
        self.channel_rates = np.zeros(self.num_users)
        
        agent_obs = self._get_downlink_obs(channel_obs)

        return agent_obs

    def step(self, action):
        # dual variable
        lambd = action[-self.constraint_dim:]

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.scale_power(power_action)

        # constraint violation
        action_penalty = (power_action.sum() - self.upperbound)  # constraint violation
        self.constraint_violation = action_penalty
        self.constraint_hist.append(np.atleast_1d(action_penalty))

        # 先计算当前状态下的性能指标
        SINR_per_link, pdr_per_link, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, lambd, action_penalty)

        # 记录数据到CSV文件（如果logger存在）- 在状态更新后、新状态生成前记录
        if hasattr(self, 'logger') and self.logger is not None:
            # 创建简化的观察数据用于记录
            channel_obs = self.H.flatten()
            normalized_power_allocation = power_action / self.max_pwr_perplant
            obs_data = np.hstack((channel_obs, normalized_power_allocation))
            
            # 记录数据
            self.logger.log_step_data(
                self.time_step,
                self,
                power_action,
                np.zeros(self.num_users),  # 默认不使用卫星
                obs_data=obs_data,
                env_id=getattr(self, 'env_id', 0)
            )

        # 生成新的信道状态（用于下一步）
        self.H, channel_states = self.sample_graph()  # new channel states --- downlink
        self.current_state = channel_states  # 简化状态只保留信道状态
        states_obs = self._get_downlink_obs(channel_states)

        # 构建info字典，包含TensorBoard需要记录的指标
        info = {
            # 吞吐量相关
            'throughput_hvft': float(np.sum(self.throughput_per_user)),  # 总吞吐量
            'throughput_others': float(np.sum(self.throughput_per_user)),  # 对于HSR环境，所有流量都视为HVFT
            'throughput_all_total': float(np.sum(self.throughput_per_user)),  # 总吞吐量
            'link_pdr_avg': float(np.mean(self.last_pdr_per_link)),
            
            # SINR相关
            'sinr_sum': float(np.sum(self.last_sinr_per_link)),
            'sinr_mean': float(np.mean(self.last_sinr_per_link)),
            'sinr_min': float(np.min(self.last_sinr_per_link)),
            'log_sinr_sum': float(np.sum(np.log(self.last_sinr_per_link + 1e-8))),
            
            # 延迟相关
            'delay_sum': float(np.sum(self.link_delays)),
            'delay_mean': float(np.mean(self.link_delays)),
            'rtt_delay_avg': float(np.mean(self.rtt_delays)),
            'link_delay_avg': float(np.mean(self.link_delays)),
            
            # 功率相关
            'total_power': float(np.sum(power_action)),
            'power_efficiency': float(np.sum(self.throughput_per_user) / (np.sum(power_action) + 1e-8)),
            
            # 干扰相关
            'interference_received_avg': float(np.mean(self.interference_received)),
            'interference_caused_avg': float(np.mean(self.interference_caused_to_all)),
            
            # 信道相关
            'channel_rate_avg': float(np.mean(self.channel_rates)),
            'channel_gain_avg': float(np.mean(self.channel_gains)),
            
            # 系统级指标
            'system_avg_rate': float(getattr(self, 'system_avg_rate', 0.0)),
            'total_interference_penalty': float(getattr(self, 'total_interference_penalty', 0.0)),
            'constraint_violation': float(self.constraint_violation),
            
            # HSR特有指标
            'hsr_throughput_total': float(np.sum(self.throughput_per_user)),
            'hsr_throughput_per_user_avg': float(np.mean(self.throughput_per_user)),
            'hsr_power_efficiency': float(np.sum(self.throughput_per_user) / (np.sum(power_action) + 1e-8)),
            'hsr_delay_performance': float(np.mean(self.link_delays)),
            'hsr_rtt_performance': float(np.mean(self.rtt_delays)),
            
            # 兼容custom_callback的字段（HSR环境简化版本）
            'satellite_users_count': 0,  # HSR环境不使用卫星
            'ground_users_count': self.num_users,  # 所有用户都是地面用户
            'hvft_accumulated_data_total': float(np.sum(self.throughput_per_user)),
            'hvft_accumulation_rounds_avg': 1.0,  # HSR环境简化处理
            'hvft_transmission_decisions_sum': self.num_users,
            'hvft_satellite_transmission_sum': 0,
            
            # 延迟统计（HSR环境简化版本）
            'others_delay_sum': float(np.sum(self.link_delays)),
            'others_delay_mean': float(np.mean(self.link_delays)),
            'others_delay_min': float(np.min(self.link_delays)),
            'others_delay_max': float(np.max(self.link_delays)),
            'others_delay_std': float(np.std(self.link_delays)),
            'others_count': self.num_users,
            
            'hvft_delay_sum': float(np.sum(self.link_delays)),
            'hvft_delay_mean': float(np.mean(self.link_delays)),
            'hvft_delay_min': float(np.min(self.link_delays)),
            'hvft_delay_max': float(np.max(self.link_delays)),
            'hvft_delay_std': float(np.std(self.link_delays)),
            'hvft_count': self.num_users,
        }
        
        return states_obs, one_step_reward, bool(done), info


class LQRAdHocDownlinkOutputConstraint(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, T=40,
                 gamma=0.99, pl=2., pp=5., p0=1, num_features=1, scaling=False):

        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, T=T,
                         gamma=gamma, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        # Downlink: continuous allocation decisions
        self.action_space = spaces.Box(low=-np.ones(num_users), high=np.ones(num_users))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        # 初始化观察状态属性
        self.channel_gains = np.zeros(self.num_users)
        self.current_power_allocation = np.zeros(self.num_users)
        self.interference_received = np.zeros(self.num_users)
        self.interference_caused_to_all = np.zeros(self.num_users)
        self.channel_rates = np.zeros(self.num_users)
        
        agent_obs = self._get_downlink_obs(channel_obs)

        return agent_obs

    def step(self, action):
        lambd = 0.
        action_penalty = 0.

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.normalize_scale_power(power_action)  # power decisions in [0, 1]

        # 先计算当前状态下的性能指标
        SINR_per_link, pdr_per_link, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, lambd, action_penalty)

        # 记录数据到CSV文件（如果logger存在）- 在状态更新后、新状态生成前记录
        if hasattr(self, 'logger') and self.logger is not None:
            # 创建简化的观察数据用于记录
            channel_obs = self.H.flatten()
            normalized_power_allocation = power_action / self.max_pwr_perplant
            obs_data = np.hstack((channel_obs, normalized_power_allocation))
            
            # 记录数据
            self.logger.log_step_data(
                self.time_step,
                self,
                power_action,
                np.zeros(self.num_users),  # 默认不使用卫星
                obs_data=obs_data,
                env_id=getattr(self, 'env_id', 0)
            )

        # 生成新的信道状态（用于下一步）
        self.H, channel_states = self.sample_graph()  # new channel states --- downlink
        self.current_state = channel_states  # 简化状态只保留信道状态
        states_obs = self._get_downlink_obs(channel_states)

        # 构建info字典，包含TensorBoard需要记录的指标
        info = {
            # 吞吐量相关
            'throughput_hvft': float(np.sum(self.throughput_per_user)),  # 总吞吐量
            'throughput_others': float(np.sum(self.throughput_per_user)),  # 对于HSR环境，所有流量都视为HVFT
            'throughput_all_total': float(np.sum(self.throughput_per_user)),  # 总吞吐量
            'link_pdr_avg': float(np.mean(self.last_pdr_per_link)),
            
            # SINR相关
            'sinr_sum': float(np.sum(self.last_sinr_per_link)),
            'sinr_mean': float(np.mean(self.last_sinr_per_link)),
            'sinr_min': float(np.min(self.last_sinr_per_link)),
            'log_sinr_sum': float(np.sum(np.log(self.last_sinr_per_link + 1e-8))),
            
            # 延迟相关
            'delay_sum': float(np.sum(self.link_delays)),
            'delay_mean': float(np.mean(self.link_delays)),
            'rtt_delay_avg': float(np.mean(self.rtt_delays)),
            'link_delay_avg': float(np.mean(self.link_delays)),
            
            # 功率相关
            'total_power': float(np.sum(power_action)),
            'power_efficiency': float(np.sum(self.throughput_per_user) / (np.sum(power_action) + 1e-8)),
            
            # 干扰相关
            'interference_received_avg': float(np.mean(self.interference_received)),
            'interference_caused_avg': float(np.mean(self.interference_caused_to_all)),
            
            # 信道相关
            'channel_rate_avg': float(np.mean(self.channel_rates)),
            'channel_gain_avg': float(np.mean(self.channel_gains)),
            
            # 系统级指标
            'system_avg_rate': float(getattr(self, 'system_avg_rate', 0.0)),
            'total_interference_penalty': float(getattr(self, 'total_interference_penalty', 0.0)),
            'constraint_violation': float(self.constraint_violation),
            
            # HSR特有指标
            'hsr_throughput_total': float(np.sum(self.throughput_per_user)),
            'hsr_throughput_per_user_avg': float(np.mean(self.throughput_per_user)),
            'hsr_power_efficiency': float(np.sum(self.throughput_per_user) / (np.sum(power_action) + 1e-8)),
            'hsr_delay_performance': float(np.mean(self.link_delays)),
            'hsr_rtt_performance': float(np.mean(self.rtt_delays)),
            
            # 兼容custom_callback的字段（HSR环境简化版本）
            'satellite_users_count': 0,  # HSR环境不使用卫星
            'ground_users_count': self.num_users,  # 所有用户都是地面用户
            'hvft_accumulated_data_total': float(np.sum(self.throughput_per_user)),
            'hvft_accumulation_rounds_avg': 1.0,  # HSR环境简化处理
            'hvft_transmission_decisions_sum': self.num_users,
            'hvft_satellite_transmission_sum': 0,
            
            # 延迟统计（HSR环境简化版本）
            'others_delay_sum': float(np.sum(self.link_delays)),
            'others_delay_mean': float(np.mean(self.link_delays)),
            'others_delay_min': float(np.min(self.link_delays)),
            'others_delay_max': float(np.max(self.link_delays)),
            'others_delay_std': float(np.std(self.link_delays)),
            'others_count': self.num_users,
            
            'hvft_delay_sum': float(np.sum(self.link_delays)),
            'hvft_delay_mean': float(np.mean(self.link_delays)),
            'hvft_delay_min': float(np.min(self.link_delays)),
            'hvft_delay_max': float(np.max(self.link_delays)),
            'hvft_delay_std': float(np.std(self.link_delays)),
            'hvft_count': self.num_users,
        
        }
        
        return states_obs, one_step_reward, bool(done), info


