###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

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

force_max_power = True

class LQR_Env(gym.Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, mu=1, T=40,
                 gamma=0.99, pl=2., pp=5., p0=1., num_features=3, scaling=True,
                 ideal_comm=False, force_max_power=force_max_power, weights=None,
                 snr_threshold_eta=10.0, max_neighbors=5, train_length=200, bs_hexagon_radius=600,
                 num_base_stations=7, bs_spacing=600):  # 增加基站至7个，间距改为600m
        super(LQR_Env, self).__init__()

        # dimensions (必须先设置，因为其他方法需要用到)
        self.num_features = num_features
        self.num_users = num_users
        
        # 高速铁路场景参数
        self.train_length = train_length  # 列车长度(米)
        self.num_base_stations = num_base_stations  # 基站数量
        self.bs_spacing = bs_spacing  # 基站间距(米)
        
        # 邻居选择参数
        self.snr_threshold_eta = snr_threshold_eta  # SNR阈值η
        self.max_neighbors = max_neighbors  # 最大邻居数目
        
        # 设备编号和位置信息
        self.device_ids = list(range(num_users))  # 为每个设备分配唯一编号
        self.device_positions = self._initialize_device_positions()  # 初始化设备位置
        self.base_station_positions = self._initialize_base_stations()  # 初始化多个基站位置
        self.device_bs_assignment = np.zeros(num_users, dtype=int)  # 每个设备对应的基站索引
        
        # 新的状态维度计算：根据图片定义更新
        # 1. 本地信息：每个用户的功率、PFS权重倒数、频谱效率、2次历史信道增益、2次历史干扰+噪声
        self.local_info_dim = num_users * (1 + 1 + 1 + 2 + 2)  # 每个用户7个本地特征
        
        # 2. 干扰邻居信息：接收到的干扰、权重倒数、网络贡献（3个特征）
        self.interference_neighbors_dim = max_neighbors * num_users * 3  # 每个用户最多max_neighbors个干扰邻居，每个邻居3个特征
        
        # 3. 被干扰邻居信息：反馈信息、权重倒数、信道增益、网络贡献（4个特征）
        self.interfered_neighbors_dim = max_neighbors * num_users * 4  # 每个用户最多max_neighbors个被干扰邻居，每个邻居4个特征
        
        self.local_state_dim = self.local_info_dim + self.interference_neighbors_dim + self.interfered_neighbors_dim
        
        # 信道状态维度：修正为num_users × num_base_stations
        self.channel_state_dim = num_users * num_base_stations
        
        self.state_dim = self.local_state_dim  # 新的状态维度
        self.action_dim = num_users
        self.constraint_dim = constraint_dim
        
        self.enhanced_state_dim = self.local_state_dim  
        self.control_state_dim = num_users * 3
        # using different seeds across different realizations of the WCS
        self.np_random = []
        self.seed()
        
        # system parameters
        self.T = T
        self.max_pwr_perplant = pp
        self.p0 = p0
        self.mu = mu  # parameter for distribution of channel states (fast fading)
        noise_power_dbm = -114  # dBm
        self.noise_power_linear = 10**(noise_power_dbm/10) * 1e-3 # 转换为瓦特 (W)
        self.sigma = np.sqrt(self.noise_power_linear)  # 噪声标准差 σ = √(噪声功率)
        self.n_transmitting = np.rint(num_users/3).astype(np.int32)  # number of plants transmitting at a given time
        self.gamma = gamma
        
        # 历史信息存储
        self.history_length = 2  # 保存最近2次测量值
        self.power_history = np.full((self.history_length, num_users), -0.01)  # 功率历史
        self.channel_gains_history = np.full((self.history_length, num_users), -0.01)  # 信道增益历史
        self.interference_noise_history = np.full((self.history_length, num_users), -0.01)  # 干扰+噪声功率历史
        self.spectral_efficiency_history = np.full((self.history_length, num_users), -0.01)  # 频谱效率历史
        
        # PFS相关历史信息
        self.H_matrix_history = []  # 完整信道矩阵历史，用于邻居确定
        self.long_term_avg_spectral_efficiency = np.full(num_users, 1e-6)  # 长期平均频谱效率 C-bar_i
        self.pfs_weights = np.ones(num_users)  # PFS权重 w_i
        
        self.current_interference_neighbors = {}  # 当前的干扰邻居集合
        self.current_interfered_neighbors = {}  # 当前的被干扰邻居集合
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
        # 值范围设为[0, 1]
        self.observation_space = spaces.Box(low=np.zeros(self.enhanced_state_dim),
                                            high=np.ones(self.enhanced_state_dim))
        self.scaling = scaling

        # HSR相关参数（高铁场景）
        self.f = 930 * 1e6   # 2 * 1e9  # 载波频率 (Hz) - 2 GHz
        self.A_b = 30  # 基站天线高度 (m)
        self.A_m = 3   # 移动台天线高度 (m)
        self.t = 1     # 时间参数
        self.v = 100   # 速度 (m/s)
        self.d = 400   # 距离参数 (m)
        self.R = 500  # 半径参数 (m)
        self.choice = 1  # Hata模型选择
        self.bandwidth = 20e6  # 带宽 (Hz)
        
        # Jakes衰落模型参数
        self.c_light = 3e8  # 光速 (m/s)
        self.time_slot_duration = 1.0  # 时间槽长度 (s)
        self.channel_initialized = False  # 信道初始化标志
        self.h_small_scale = None  # 小尺度衰落分量
        self.h_small_scale_prev = None  # 前一时刻的小尺度衰落
        self.alpha_large_scale = None  # 大尺度衰落分量
        
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
    
        self.channel_rates = np.zeros(self.num_users)  # 信道速率
        self.channel_gains = np.zeros(self.num_users)  # 信道增益
        
        self.traffic_loads = np.full(self.num_users, 3000*8)  # 流量负载 (Mbps) - 每个用户固定为3000*8
        self.rtt_delays = np.zeros(self.num_users)  # RTT延迟
        self.link_delays = np.zeros(self.num_users)  # 链路延迟
        self.throughput_per_user = np.zeros(self.num_users)  # 每个用户的吞吐量
        
        self.force_max_power = force_max_power
        
    def disc_cost(self, cost_vec):
        T = np.size(cost_vec)
        cost_discounted = np.zeros(T)
        cost_discounted[-1] = cost_vec[-1]

        # 向量化计算折扣回报：使用累积乘积和卷积
        # 计算gamma的幂次
        gamma_powers = self.gamma ** np.arange(T)
        
        # 反向累积：cost_discounted[i] = sum(gamma^(j-i) * cost_vec[j] for j in range(i, T))
        for i in range(T-2, -1, -1):
            cost_discounted[i] = cost_vec[i] + self.gamma * cost_discounted[i + 1]

        return cost_discounted

    def disc_constraint(self, cost_vec):
        T = np.size(cost_vec)
        if cost_vec.ndim > 1:
            T, constraint_dim = cost_vec.shape
            cost_discounted = np.zeros((T, constraint_dim))
            # 反向累积计算
            cost_discounted[-1] = cost_vec[-1]
            for i in range(T-2, -1, -1):
                cost_discounted[i] = cost_vec[i] + self.gamma * cost_discounted[i + 1]
        else:
            cost_discounted = np.zeros(T)
            cost_discounted[-1] = cost_vec[-1]
            # 反向累积计算
            for i in range(T-2, -1, -1):
                cost_discounted[i] = cost_vec[i] + self.gamma * cost_discounted[i + 1]

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

    # packet delivery rate: no interference (已去除PDR约束)
    # @staticmethod
    # def packet_delivery_rate(snr_value):
    #     return 1.0
    @staticmethod
    def packet_delivery_rate(snr_value):
        pdr = 1 - np.exp(-snr_value)
        pdr = np.nan_to_num(pdr)
        return pdr
    
    def compute_doppler(self, t, v, d, R):
        c = 3 * 10 ** 8
        b = math.sqrt(R ** 2 - d ** 2) if R > d else 0
        x = b - v * t  # 列车相对于基站的水平位置
        r = math.sqrt(x ** 2 + d ** 2)  # 实际距离
        cos_theta = x / r if r > 0 else 0  # 夹角余弦值
        # 计算多普勒频移：fd = f * v * cos(θ) / c
        doppler_effect_num = self.f * (v * cos_theta) / c
        
        return doppler_effect_num, r

    def calculate_correlation_coefficient(self):
        """
        计算每个设备-基站对的相关系数，考虑设备与其服务基站之间的相对运动
        返回形状为 (num_users, num_base_stations) 的相关系数矩阵
        """
        rho_matrix = np.zeros((self.num_users, self.num_base_stations))
        
        for i in range(self.num_users):
            for j in range(self.num_base_stations):
                # 计算设备i相对于基站j的多普勒频移
                device_pos = self.device_positions[i]
                bs_pos = self.base_station_positions[j]
                
                # 计算设备到基站的向量和距离
                device_to_bs_vector = bs_pos - device_pos
                distance = np.sqrt(np.sum(device_to_bs_vector**2))
                
                # 计算列车运动方向（假设沿x轴方向）
                train_velocity_vector = np.array([self.v, 0.0])  # 列车速度向量（m/s）
                
                # 计算设备运动方向与到基站方向的夹角
                if distance > 0:
                    # 计算角度：cos(θ) = (v·r) / (|v||r|)
                    # 这里r是从设备指向基站的向量
                    cos_angle = np.dot(train_velocity_vector, device_to_bs_vector) / (self.v * distance)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                else:
                    cos_angle = 0.0
                
                # 计算有效速度和多普勒频移
                effective_velocity = self.v * cos_angle
                doppler_frequency = effective_velocity * self.f / self.c_light
                
                # 计算相关系数：ρ = J₀(2π * f_d * T)
                rho_matrix[i, j] = j0(2 * np.pi * doppler_frequency * self.time_slot_duration)
        
        return rho_matrix
        
    def _initialize_device_positions(self):
        """
        初始化设备位置：列车在基站下方200米平行行驶
        列车尾部从第一个基站的最左边出发
        设备在列车长度上均匀分布
        """
        # 计算基站0的位置（最左侧基站）
        # 基站以x=0为中心对称分布
        # 对于奇数个基站，最左侧基站位置 = -(基站数量-1)/2 * 间距
        bs0_x = -(self.num_base_stations - 1) / 2 * self.bs_spacing
        
        # 计算六边形基站的覆盖半径（假设六边形内切圆半径）
        # 六边形边长 = 基站间距，内切圆半径 = 边长 * sqrt(3) / 2
        hexagon_radius = self.bs_spacing * np.sqrt(3) / 2
        
        # 列车尾部从第一个基站的最左边出发
        # 最左边 = 基站0位置 - 六边形半径
        train_tail_x = bs0_x - hexagon_radius
        
        # 设备在列车长度上均匀分布，从列车尾部开始
        x_positions = np.linspace(
            train_tail_x,  # 列车尾部（第一个设备）从基站0最左边出发
            train_tail_x + self.train_length,  # 列车头部（最后一个设备）位置
            self.num_users
        )
        
        # 列车在基站下方200米平行行驶
        y_positions = np.full(self.num_users, -200)
        
        # 组合成位置数组
        positions = np.column_stack((x_positions, y_positions))
        
        return positions
    
    def _initialize_base_stations(self):

        base_stations = []
        # 计算基站分布范围
        total_range = (self.num_base_stations - 1) * self.bs_spacing
        start_x = -total_range / 2
        
        for i in range(self.num_base_stations):
            x = start_x + i * self.bs_spacing
            y = 0  # 基站都在y=0位置
            base_stations.append([x, y])
        
        return np.array(base_stations)
    
    
    def update_device_bs_assignment(self):
        device_pos = self.device_positions  # (num_users, 2)
        bs_pos = self.base_station_positions  # (num_base_stations, 2)
        
        # 计算距离矩阵: (num_users, num_base_stations, 2) -> (num_users, num_base_stations)
        distances = np.sqrt(np.sum((device_pos[:, np.newaxis, :] - bs_pos[np.newaxis, :, :])**2, axis=2))
        
        # 找到每个设备最近的基站索引
        self.device_bs_assignment = np.argmin(distances, axis=1)
    
    def get_device_base_station(self, device_id):
        """
        获取设备对应的基站位置
        Args:
            device_id: 设备ID
        Returns:
            np.array: 基站位置 [x, y]
        """
        if device_id >= self.num_users:
            raise ValueError(f"设备ID超出范围: {device_id}")
        
        bs_index = self.device_bs_assignment[device_id]
        return self.base_station_positions[bs_index]
        
    def calculate_device_distance(self, device_id1, device_id2):
        if device_id1 >= self.num_users or device_id2 >= self.num_users:
            raise ValueError(f"设备ID超出范围: {device_id1}, {device_id2}")
            
        pos1 = self.device_positions[device_id1]
        pos2 = self.device_positions[device_id2]
        
        return np.linalg.norm(pos1 - pos2)
        
    def calculate_device_to_bs_distance(self, device_id):
        if device_id >= self.num_users:
            raise ValueError(f"设备ID超出范围: {device_id}")
            
        device_pos = self.device_positions[device_id]
        bs_pos = self.get_device_base_station(device_id)
        
        return np.linalg.norm(device_pos - bs_pos)
    
    def calculate_all_device_to_bs_distances(self):
        """
        向量化计算所有设备到其对应基站的距离
        Returns:
            np.array: 距离向量 (num_users,)
        """
        device_positions_array = np.array(self.device_positions)  # (num_users, 2)
        bs_indices = self.device_bs_assignment  # (num_users,)
        serving_bs_positions = self.base_station_positions[bs_indices]  # (num_users, 2)
        
        # 向量化计算距离
        distances = np.sqrt(np.sum((device_positions_array - serving_bs_positions)**2, axis=1))
        
        return distances
        
    def get_all_distances_matrix(self):
        # 向量化计算：使用广播机制
        positions = np.array(self.device_positions)  # (num_users, 2)
        
        # 计算所有设备对之间的距离
        # positions[:, np.newaxis, :] - positions[np.newaxis, :, :] 创建 (num_users, num_users, 2) 的差值矩阵
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        
        # 计算欧几里得距离：sqrt(sum(diff^2, axis=2))
        distances = np.sqrt(np.sum(diff**2, axis=2))
                    
        return distances
        
    def update_device_positions(self, time_step_delta=None):
        if time_step_delta is None:
            time_step_delta = getattr(self, 'time_slot_duration', 1)  # 默认1s
            
        # 获取列车速度 (m/s)
        train_velocity = getattr(self, 'v', 100)  # 默认100 m/s (360 km/h)
        
        # 计算位移
        displacement = train_velocity * time_step_delta
        
        # 更新所有设备的x坐标（列车沿x轴运动）
        self.device_positions[:, 0] += displacement
        
        # 更新设备基站分配（选择最近的基站）
        self.update_device_bs_assignment()
        
        # 位置更新完成，不进行基于位置的重置
        # 重置将在每个T（episode）结束时在reset()方法中进行
    
    def _reset_device_positions(self):
        """重置设备位置到初始状态（在每个T/episode开始时调用）"""
        # 重置到初始位置
        self.device_positions = self._initialize_device_positions()
        
        # 重新分配基站（选择最近的基站）
        self.update_device_bs_assignment()
        
        # 重新计算大尺度衰落矩阵
        self.L_matrix = self.calculate_large_scale_fading()
        
        # 重置Jakes信道模型的小尺度衰落状态
        if hasattr(self, 'channel_initialized') and self.channel_initialized:
            # 重新初始化小尺度衰落，保持时间相关性
            # 修正维度：num_users × num_base_stations
            self.g_small_scale = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_base_stations)) + \
                                1j * np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_base_stations))
            self.g_prev = self.g_small_scale.copy()
        
        
    def get_device_info(self, device_id):
        if device_id >= self.num_users:
            raise ValueError(f"设备ID超出范围: {device_id}")
            
        return {
            'device_id': device_id,
            'position': self.device_positions[device_id].copy(),
            'distance_to_bs': self.calculate_device_to_bs_distance(device_id),
            'neighbors': self._get_device_neighbors(device_id)
        }
        
    def _get_device_neighbors(self, device_id, max_distance=50):
        neighbors = []
        for i in range(self.num_users):
            if i != device_id:
                distance = self.calculate_device_distance(device_id, i)
                if distance <= max_distance:
                    neighbors.append(i)
        return neighbors
        
    def print_network_topology(self):
        for i, pos in enumerate(self.device_positions):
            dist_to_bs = self.calculate_device_to_bs_distance(i)

    def calculate_large_scale_fading(self):
        """
        计算设备到基站的大尺度衰落矩阵
        返回形状为 (num_users, num_base_stations) 的矩阵
        L_matrix[i, j] 表示设备i到基站j的大尺度衰落增益
        """
        # 初始化大尺度衰落矩阵：设备到基站
        L_matrix = np.zeros((self.num_users, self.num_base_stations))
        
        # 使用Hata模型参数
        choice = 1
        if choice == 1:
            delta1 = -21.42
            delta2 = -9.62
        else:
            log10_hb = math.log10(30)
            delta1 = 5.74 * log10_hb - 30.42
            delta2 = -6.72
        
        # 计算所有设备到所有基站的距离矩阵
        device_pos = self.device_positions  # (num_users, 2)
        bs_pos = self.base_station_positions  # (num_base_stations, 2)
        
        # 使用广播计算距离矩阵: (num_users, 1, 2) - (1, num_base_stations, 2) -> (num_users, num_base_stations, 2)
        device_pos_expanded = device_pos[:, np.newaxis, :]  # (num_users, 1, 2)
        bs_pos_expanded = bs_pos[np.newaxis, :, :]  # (1, num_base_stations, 2)
        
        # 计算距离: (num_users, num_base_stations)
        device_to_bs_distances = np.sqrt(np.sum((device_pos_expanded - bs_pos_expanded)**2, axis=2))
        device_to_bs_distances_km = device_to_bs_distances / 1000.0
        
        # 计算路径损失 (dB)
        # 避免log(0)错误
        valid_distances_mask = device_to_bs_distances > 0
        path_loss_dB = np.zeros((self.num_users, self.num_base_stations))
        
        if np.any(valid_distances_mask):
            d_km_valid = device_to_bs_distances_km[valid_distances_mask]
            path_loss_dB[valid_distances_mask] = (
                delta1 + 74.52 + 26.16 * np.log10(930) - 13.82 * np.log10(30)
                - 3.2 * (np.log10(11.75 * 3))**2
                + (44.9 - 6.55 * np.log10(30) + delta2) * np.log10(d_km_valid)
            )
        
        # 添加阴影衰落 (dB)
        shadowing_dB = np.random.normal(0, 8, (self.num_users, self.num_base_stations))
        total_loss_dB = path_loss_dB + shadowing_dB
        
        # 转换为线性增益
        L_matrix = 10 ** (-total_loss_dB / 10)
        
        # 处理距离为0的情况（设备与基站重合）
        L_matrix[device_to_bs_distances == 0] = 1.0
        
        return L_matrix

    def initialize_jakes_channel_model(self):
        # 初始化大尺度衰落系数 L_{m,s}
        self.L_matrix = self.calculate_large_scale_fading()
        
        # 初始化小尺度衰落系数 g_{m,s}
        # 生成初始的圆对称复高斯随机变量：CN(0,1)
        # 实部和虚部都是均值为0，方差为1/2的正态分布
        # 修正维度：num_users × num_base_stations
        real_part = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_base_stations))
        imag_part = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_base_stations))
        self.g_small_scale = real_part + 1j * imag_part
        
        # 保存前一时刻的小尺度衰落 g_{m-1,s}
        self.g_prev = self.g_small_scale.copy()
        
        # 标记信道已初始化
        self.channel_initialized = True

    def update_jakes_channel_model(self, update_positions=True):
        if not self.channel_initialized:
            self.initialize_jakes_channel_model()
            
        # 更新设备位置（模拟列车运动）
        if update_positions:
            self.update_device_positions()
            self.L_matrix = self.calculate_large_scale_fading()
        
        # 计算相关系数矩阵 rho = J₀(2π * f_d * T) 
        rho_matrix = self.calculate_correlation_coefficient()
        
        # 生成信道创新过程 e_{m,s}，其分布为 CN(0, 1 - rho^2) 
        innovation_variance = 1 - rho_matrix**2
        innovation_std = np.sqrt(innovation_variance)
        real_innovation = np.random.normal(0, innovation_std / np.sqrt(2), size=(self.num_users, self.num_base_stations))
        imag_innovation = np.random.normal(0, innovation_std / np.sqrt(2), size=(self.num_users, self.num_base_stations))
        e_innovation = real_innovation + 1j * imag_innovation
        
        # 更新小尺度衰落系数 
        self.g_small_scale = rho_matrix * self.g_prev + e_innovation
        
        # 计算最终信道功率增益 
        # 注意：这里不再使用L_matrix，因为新的H_matrix已经包含了大尺度衰落
        channel_power_gains = np.abs(self.g_small_scale)**2
        
        # 保存当前小尺度衰落作为下一时刻的 g_{m-1,s}
        self.g_prev = self.g_small_scale.copy()
        
        return channel_power_gains

    def calculate_multi_bs_channel_matrix(self):
        """
        计算多基站场景下的信道增益矩阵 - 修正版本
        矩阵元素H[i,j]表示基站j到设备i的信道增益
        每个设备连接到距离自己最近的基站进行下行传输
        """
        # 更新设备基站分配（确保连接到最近基站）
        self.update_device_bs_assignment()
        
        # 使用更新后的大尺度衰落矩阵（设备到基站）
        if hasattr(self, 'L_matrix') and self.L_matrix is not None:
            # 使用已计算的大尺度衰落矩阵
            large_scale_gain = self.L_matrix
        else:
            # 如果没有，重新计算
            large_scale_gain = self.calculate_large_scale_fading()
        
        # 计算小尺度衰落（使用Jakes模型更新的g_small_scale）
        if hasattr(self, 'g_small_scale') and self.g_small_scale is not None:
            # 使用Jakes模型更新的小尺度衰落，保持时间相关性
            small_scale_gain = np.abs(self.g_small_scale)**2
        else:
            # 如果没有初始化，使用随机值（正确维度）
            real_part = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_base_stations))
            imag_part = np.random.normal(0, 1/np.sqrt(2), size=(self.num_users, self.num_base_stations))
            small_scale_gain = np.abs(real_part + 1j * imag_part)**2
        
        # 计算最终信道增益矩阵：H_matrix[i, j] = 基站j到设备i的信道增益
        # 大尺度衰落 × 小尺度衰落
        H_matrix = large_scale_gain * small_scale_gain
        
        return H_matrix


    def compute_WICI(self, device_idx, serving_bs_idx):
        # 获取设备位置和基站位置
        device_pos = self.device_positions[device_idx]
        bs_pos = self.base_station_positions[serving_bs_idx]
        
        # 计算设备到基站的向量和距离
        device_to_bs_vector = bs_pos - device_pos
        distance = np.sqrt(np.sum(device_to_bs_vector**2))
        
        # 计算列车运动方向（假设沿x轴方向）
        train_velocity_vector = np.array([self.v, 0.0])  # 列车速度向量（m/s）
        
        # 计算设备运动方向与到基站方向的夹角
        # 多普勒频移公式：fd = (v * cos(θ) * f) / c
        # 其中θ是设备运动方向与到基站方向的夹角
        if distance > 0:
            # 计算角度：cos(θ) = (v·r) / (|v||r|)
            # 这里r是从设备指向基站的向量
            cos_angle = np.dot(train_velocity_vector, device_to_bs_vector) / (self.v * distance)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在有效范围内
        else:
            cos_angle = 0.0
        
        # 计算该设备相对于基站的有效速度
        # 有效速度 = 列车速度 * cos(θ)
        effective_velocity = self.v * cos_angle
        
        # 计算多普勒频移
        # 多普勒频移 = (v_effective * f) / c
        doppler_shift = (effective_velocity * self.f) / self.c_light
        
        # 计算该链路的载波间干扰(ICI)
        # 使用ICI计算公式：W_ICI = 1 - ∫(1-|x|) * J₀(2π * fd * Ts * x) dx
        # 其中 Ts = 1e-3 秒（符号时间，调整为1ms以反映高速移动场景）
        Ts = 1e-3  # 1ms符号时间，更适合高速移动场景
        x_values = np.linspace(-1, 1, 1000)
        integral_result = np.trapz(
            (1 - abs(x_values)) * j0(2 * np.pi * doppler_shift * Ts * x_values), 
            x_values
        )

        W_ICI_link = 1 - integral_result
        # 应用上限约束（基于理论分析）
        num = (np.pi * doppler_shift * Ts) ** 2
        den = 12
        if num/den < W_ICI_link:
            W_ICI_link = num/den
        return W_ICI_link
    
    def update_neighbor_sets(self, previous_power_allocation):
        num_users = self.num_users
        noise_power = self.noise_power_linear
        threshold = self.snr_threshold_eta * noise_power
        
        # 使用上一时隙的信道矩阵（如果有历史数据）
        if len(self.H_matrix_history) > 0:
            H_previous = self.H_matrix_history[-1]  # 最近的历史信道矩阵
        else:
            # 如果没有历史数据，使用当前信道矩阵作为近似
            if hasattr(self, 'H'):
                H_previous = self.H
            else:
                return  # 无法更新邻居集合
        
        # 修正：基于多基站场景计算干扰功率矩阵
        # 计算每个设备通过其服务基站对其他设备造成的干扰功率
        received_power_matrix = np.zeros((num_users, num_users))
        
        for i in range(num_users):
            for j in range(num_users):
                if i != j:  # 排除自己
                    # 设备j通过其服务基站对设备i造成干扰
                    serving_bs_j = self.device_bs_assignment[j]
                    received_power_matrix[i, j] = previous_power_allocation[j] * H_previous[i, serving_bs_j]
        
        # 创建掩码矩阵，排除对角线元素（自己不能是自己的邻居）
        mask = ~np.eye(num_users, dtype=bool)
        
        # 找到超过阈值的干扰功率
        above_threshold = (received_power_matrix > threshold) & mask
        
        # 初始化邻居集合
        interference_neighbors = {}
        interfered_neighbors = {}
        
        # 对每个用户处理邻居关系
        for i in range(num_users):
            # 干扰邻居：对用户i造成干扰的发射机j (干扰功率 > threshold)
            interfering_indices = np.where(above_threshold[:, i])[0]
            if len(interfering_indices) > 0:
                # 获取干扰功率
                interfering_powers = received_power_matrix[interfering_indices, i]
                # 获取对应的信道增益
                interfering_gains = np.array([H_previous[i, self.device_bs_assignment[j]] for j in interfering_indices])
                
                # 按信道增益乘以功率的大小排序（降序）
                sorted_indices = np.argsort(interfering_powers)[::-1]
                interference_list = [
                    (interfering_indices[idx], interfering_powers[idx], interfering_gains[idx])
                    for idx in sorted_indices[:self.max_neighbors]
                ]
            else:
                interference_list = []
            interference_neighbors[i] = interference_list
            
            # 被干扰邻居：被用户i干扰的接收机k (干扰功率 > threshold)
            interfered_indices = np.where(above_threshold[i, :])[0]
            if len(interfered_indices) > 0:
                # 获取造成的干扰功率
                caused_powers = received_power_matrix[i, interfered_indices]
                # 获取对应的信道增益
                caused_gains = np.array([H_previous[k, self.device_bs_assignment[i]] for k in interfered_indices])
                
                # 按信道增益乘以功率的大小排序（降序）
                sorted_indices = np.argsort(caused_powers)[::-1]
                interfered_list = [
                    (interfered_indices[idx], caused_powers[idx], caused_gains[idx])
                    for idx in sorted_indices[:self.max_neighbors]
                ]
            else:
                interfered_list = []
            interfered_neighbors[i] = interfered_list
        
        # 更新当前邻居集合
        self.current_interference_neighbors = interference_neighbors
        self.current_interfered_neighbors = interfered_neighbors
    
    def update_history(self, power_allocation, channel_gains, interference_noise, spectral_efficiency, H_matrix=None):
        # 滚动更新历史记录
        self.power_history[1:] = self.power_history[:-1]
        self.power_history[0] = power_allocation.copy()
        
        self.channel_gains_history[1:] = self.channel_gains_history[:-1]
        self.channel_gains_history[0] = channel_gains.copy()
        
        self.interference_noise_history[1:] = self.interference_noise_history[:-1]
        self.interference_noise_history[0] = interference_noise.copy()
        
        self.spectral_efficiency_history[1:] = self.spectral_efficiency_history[:-1]
        self.spectral_efficiency_history[0] = spectral_efficiency.copy()
        
        # 更新信道矩阵历史
        if H_matrix is not None:
            self.H_matrix_history.append(H_matrix.copy())
            # 只保存最近几次的历史，避免内存过大
            if len(self.H_matrix_history) > self.history_length:
                self.H_matrix_history.pop(0)
        
        # 更新PFS权重和长期平均频谱效率
        self.update_pfs_weights(spectral_efficiency)

    def update_pfs_weights(self, current_spectral_efficiency):
        # 时间衰减因子（可以调整以控制历史影响）
        alpha = 0.3  # 历史权重，取值接近1表示更重视历史
        
        # 更新长期平均频谱效率：C-bar_i^(t) = alpha * C-bar_i^(t-1) + (1-alpha) * C_i^(t)
        # 只对有效值（非负值）进行更新
        valid_mask = current_spectral_efficiency >= 0
        if np.any(valid_mask):
            self.long_term_avg_spectral_efficiency[valid_mask] = (
                alpha * self.long_term_avg_spectral_efficiency[valid_mask] + 
                (1 - alpha) * current_spectral_efficiency[valid_mask]
            )
        
        # 计算PFS权重：w_i^(t+1) = (C-bar_i^(t))^(-1)
        # 使用更严格的最小值限制避免权重过大
        min_avg_rate = 1e-3  # 提高最小值，从1e-6到1e-3
        raw_weights = 1.0 / np.maximum(self.long_term_avg_spectral_efficiency, min_avg_rate)
        max_absolute_weight = 5.0
        self.pfs_weights = np.minimum(raw_weights, max_absolute_weight)
        # 归一化
        self.pfs_weights = self.pfs_weights / np.max(self.pfs_weights)
    

    def interference_SINR(self, H_matrix, actions):
        """
        多基站场景下的SINR计算
        H_matrix[i,j]: 基站j到设备i的信道增益 (num_users × num_base_stations)
        actions: 每个设备的功率分配
        """
        num_users = self.num_users
        num_base_stations = self.num_base_stations
        
        # 初始化结果数组
        SINR = np.zeros(num_users)
        channel_rates = np.zeros(num_users)
        interference_plus_noise = np.zeros(num_users)
        
        # 初始化干扰统计
        self.interference_received = np.zeros(num_users)
        self.interference_caused_to_all = np.zeros(num_users)
        
        # 为每个设备单独计算SINR
        for i in range(num_users):
            # 1. 计算设备i的信号功率：P_i * h_{i, serving_bs_i}
            serving_bs_idx = self.device_bs_assignment[i]
            signal_power = actions[i] * H_matrix[i, serving_bs_idx]
            
            # 2. 为每个链路计算特定的噪声功率和W_ICI
            # 噪声功率：基础噪声 + 接收机噪声系数 + 环境变化
            noise_factor = 1.0 + 0.1 * np.random.normal(0, 1)  # 10%的噪声变化
            link_noise_power = self.noise_power_linear * noise_factor
            
            # W_ICI：为每个链路单独计算，考虑设备位置和移动特性
            W_ICI_link = self.compute_WICI(i, serving_bs_idx)
            
            # 3. 计算设备i受到的干扰功率
            interference_power = 0.0
            for j in range(num_users):
                if j != i:  # 排除自己
                    # 设备j的功率通过其服务基站对设备i造成干扰
                    serving_bs_idx_j = self.device_bs_assignment[j]
                    # 干扰功率：P_j * h_{i, serving_bs_j} * (1 + W_ICI_link)
                    interference_contribution = actions[j] * H_matrix[i, serving_bs_idx_j] * (1 + W_ICI_link)
                    interference_power += interference_contribution
            
            # 4. 计算总干扰加噪声功率
            interference_plus_noise[i] = interference_power + link_noise_power
            
            # 5. 计算SINR
            SINR[i] = signal_power / interference_plus_noise[i]
            SINR[i] = max(SINR[i], 1e-9)  # 数值稳定性
            
            # 6. 计算信道速率
            channel_rates[i] = self.bandwidth * np.log2(1 + SINR[i])
            
            # 7. 记录干扰统计
            self.interference_received[i] = interference_power
            self.interference_caused_to_all[i] = 0.0
            for k in range(num_users):
                if k != i:  # 排除自己
                    # 设备i对设备k造成的干扰：通过设备i的服务基站
                    # 注意：这里应该使用设备k的W_ICI，因为干扰是在设备k处计算的
                    serving_bs_idx_i = self.device_bs_assignment[i]
                    serving_bs_idx_k = self.device_bs_assignment[k]
                    W_ICI_k = self.compute_WICI(k, serving_bs_idx_k)
                    self.interference_caused_to_all[i] += actions[i] * H_matrix[k, serving_bs_idx_i] * (1 + W_ICI_k)
        
        return SINR, channel_rates, interference_plus_noise

    def set_logger(self, logger):
        self.logger = logger

    def set_env_id(self, env_id):
        self.env_id = env_id

    def hsr_rtt_delay(self, num_users):
        d_min = 0.065
        sigma = 0.0075
        p_spike = 0.018
        p_retransmission = 0.24
        max_spike_delay = 0.2
        
        # 向量化计算基础延迟
        d_vol = np.random.normal(loc=0, scale=sigma, size=num_users)
        
        # 向量化计算尖峰延迟
        spike_occur = np.random.rand(num_users) < p_spike
        d_spike = np.zeros(num_users)
        
        # 第一层重传
        d_spike[spike_occur] += np.random.uniform(0, max_spike_delay, size=np.sum(spike_occur))
        
        # 第二层重传
        retrans_occur = spike_occur & (np.random.rand(num_users) < p_retransmission)
        if np.any(retrans_occur):
            d_spike[retrans_occur] += np.random.uniform(0, max_spike_delay, size=np.sum(retrans_occur))
        
        # 第三层重传
        retrans2_occur = retrans_occur & (np.random.rand(num_users) < p_retransmission)
        if np.any(retrans2_occur):
            d_spike[retrans2_occur] += np.random.uniform(0, max_spike_delay, size=np.sum(retrans2_occur))
        
        # 计算总延迟
        total_delay = d_min + d_vol + d_spike
        RTT_delay = np.maximum(0, total_delay)

        return RTT_delay

    def link_delay(self, throughput_per_link_rate, traffic_loads_per_link, rtt_delays_current=None):
        max_delay = 3.0  # 延迟值上限
        
        # 向量化计算传输延迟
        # 避免除零，使用最大值函数
        safe_rates = np.maximum(throughput_per_link_rate, 1e-9)
        transmission_delays = traffic_loads_per_link / safe_rates
        
        # 对于信道速率过小的情况，设置为最大延迟
        low_rate_mask = throughput_per_link_rate < 1e-9
        transmission_delays[low_rate_mask] = max_delay
        
        # 限制延迟值不超过上限
        deliver_times_per_chunk = np.minimum(transmission_delays, max_delay)
    
        return deliver_times_per_chunk

    def get_observation_space_info(self):
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
        # 修正：从H_matrix中提取每个设备从其服务基站的信道增益
        channel_gains = np.array([self.H[i, self.device_bs_assignment[i]] for i in range(self.num_users)])
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

        u = (np.diagonal(h, axis1=1, axis2=2) * v) / (np.matmul(h2, v2)[:, :, 0] + self.noise_power_linear)
        w = 1 / (1 - u * np.diagonal(h, axis1=1, axis2=2) * v)
        N = 1000
        for n in np.arange(T):
            u2 = np.expand_dims(u ** 2, axis=2)
            w2 = np.expand_dims(w, axis=2)
            v = (w * u * np.diagonal(h, axis1=1, axis2=2)) / (np.matmul(np.transpose(h2, (0, 2, 1)), (w2 * u2)))[:, :, 0]
            v = np.minimum(np.sqrt(Pmax), np.maximum(0, v))
            v2 = np.expand_dims(v ** 2, axis=2)
            u = (np.diagonal(h, axis1=1, axis2=2) * v) / (np.matmul(h2, v2)[:, :, 0] + self.noise_power_linear)
            w = 1 / (1 - u * np.diagonal(h, axis1=1, axis2=2) * v)
        p = v ** 2
        return p

    # samples initial state and channel conditions
    def sample(self, batch_size):
        # graph, flat observation
        self.H, samples = self.sample_graph()
        # control states
        samples2 = self.np_random.normal(0, 1, size=self.control_state_dim)
        self.current_control_obs = samples2
        return np.hstack((samples, samples2))

    def sample_graph(self):  # downlink
        # 使用Jakes模型更新信道增益，同时更新设备位置
        self.update_jakes_channel_model(update_positions=True)
        A = self.calculate_multi_bs_channel_matrix()
        A[A < 1e-20] = 0.0
        # 归一化处理
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()
        
        return A, A_flat

    def sample_graph_uplink(self):  # uplink
        # 使用Jakes模型更新信道增益，同时更新设备位置
        self.update_jakes_channel_model(update_positions=True)
        A = self.calculate_multi_bs_channel_matrix()
        A[A < 1e-20] = 0.0

        A = A.T
        # 归一化处理
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()
        return A, A_flat

    def scale_power(self, power_action, force_max_power=False):
        if force_max_power:
            # 强制使用最大功率
            power_action = np.ones_like(power_action) * self.max_pwr_perplant
        else:
            # 正常的功率分配逻辑
            power_action = np.clip(power_action, -1., 1.)
            power_action += 1.
            power_action /= 2  # [0, 1.]
            power_action *= self.max_pwr_perplant

        return power_action

    def normalize_scale_power(self, power_action, force_max_power=False):
        if force_max_power:
            # 强制使用最大功率，但保持总功率约束
            power_action = np.ones_like(power_action) * self.upperbound / self.num_users
        else:
            # 正常的归一化功率分配逻辑
            power_action = np.clip(power_action, -1., 1.)
            power_action += 1.
            power_action = power_action / (power_action.sum() + 1e-8)
            power_action *= self.upperbound

        return power_action

    def _reset(self):
        # 重置设备位置到初始状态
        self._reset_device_positions()
        
        # 初始化Jakes信道模型（如果尚未初始化）
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

    def compute_reward(self, downlink_power, channel_rates, interference_plus_noise):
        spectral_efficiencies = channel_rates / self.bandwidth
        externality_costs = self.compute_all_externality_price_sum(downlink_power)
        efficiency_reward = np.mean(spectral_efficiencies)
        # fairness_reward = np.mean(self.pfs_weights * spectral_efficiencies)
        fairness_reward = np.mean(spectral_efficiencies)
        cost_penalty = np.mean(externality_costs)

        # print("fairness_reward:",fairness_reward)
        # print("cost_penalty:",cost_penalty)
        reward = fairness_reward - cost_penalty
        return reward

    def compute_delay_reward(self, downlink_power, channel_rates, interference_plus_noise):
        """
        延迟最小化奖励函数
        目标：最小化系统总延迟，延迟 = traffic_load / channel_rate
        使用PFS权重进行加权，并考虑干扰造成的延迟影响
        """
        # 计算每个用户的传输延迟：traffic_load / channel_rate
        safe_rates = np.maximum(channel_rates, 1e-9)  # 避免除零
        transmission_delays = self.traffic_loads / safe_rates
        
        # 设置延迟上限，避免极端值
        max_delay = 3.0
        transmission_delays = np.minimum(transmission_delays, max_delay)
        weighted_delays = self.pfs_weights * transmission_delays
        total_weighted_delay = np.sum(weighted_delays)
        
        mean_weighted_delay = np.mean(weighted_delays)

        delay_reward = -total_weighted_delay / (self.num_users * max_delay)  # 归一化到合理范围
        
        externality_costs = self.compute_all_externality_price_sum(downlink_power)
        delay_penalty = -np.mean(externality_costs) / self.bandwidth  # 归一化外部效应成本
        
        reward = delay_reward + delay_penalty
        
        return reward
    
    
    def compute_all_externality_price_sum(self, downlink_power):
        externality_costs = np.zeros(self.num_users)
        
        for i in range(self.num_users):
            # 获取智能体i的被干扰邻居集合
            interfered_neighbors = self.current_interfered_neighbors.get(i, [])
            if not interfered_neighbors:
                continue 
            neighbor_indices = [neighbor[0] for neighbor in interfered_neighbors]

            if len(neighbor_indices) > 0:
                # 修正：计算有智能体i干扰时邻居的频谱效率
                # 使用当前功率分配，包括智能体i的干扰
                signal_powers_with_i = np.array([downlink_power[k] * self.H[k, self.device_bs_assignment[k]] for k in neighbor_indices])
                
                # 计算干扰矩阵（包括智能体i的干扰）
                interference_matrix_with_i = np.zeros((len(neighbor_indices), self.num_users))
                for idx, k in enumerate(neighbor_indices):
                    for j in range(self.num_users):
                        if j != k:  # 排除自干扰
                            serving_bs_j = self.device_bs_assignment[j]
                            interference_matrix_with_i[idx, j] = downlink_power[j] * self.H[k, serving_bs_j]
                
                total_interference_with_i = np.sum(interference_matrix_with_i, axis=1) + self.noise_power_linear
                sinr_with_i = signal_powers_with_i / total_interference_with_i
                sinr_with_i = np.maximum(sinr_with_i, 1e-9)
                spectral_efficiency_with_interference = np.log2(1 + sinr_with_i)
                
                # 无智能体i干扰的频谱效率
                spectral_efficiency_without_i = self.compute_all_spectral_efficiency_without_agent(
                    neighbor_indices, i, downlink_power
                )
                
                weights_k = self.pfs_weights[neighbor_indices]
                externality_prices = weights_k * (spectral_efficiency_without_i - spectral_efficiency_with_interference)
                externality_costs[i] = np.sum(externality_prices)
        
        return externality_costs
    
    
    def compute_all_spectral_efficiency_without_agent(self, neighbor_indices, interfering_agent_i, downlink_power):
        if len(neighbor_indices) == 0:
            return np.array([])
        
        # 修正：计算邻居设备的信号功率（从其服务基站）
        signal_powers = np.array([downlink_power[i] * self.H[i, self.device_bs_assignment[i]] for i in neighbor_indices])
        
        # 修正：计算干扰矩阵（基于多基站场景）
        interference_matrix = np.zeros((len(neighbor_indices), self.num_users))
        for idx, i in enumerate(neighbor_indices):
            for j in range(self.num_users):
                if j != i and j != interfering_agent_i:  # 排除自己和干扰智能体
                    serving_bs_j = self.device_bs_assignment[j]
                    interference_matrix[idx, j] = downlink_power[j] * self.H[i, serving_bs_j]
        
        # 计算总干扰
        total_interference = np.sum(interference_matrix, axis=1) + self.noise_power_linear
        
        sinr = signal_powers / total_interference
        sinr = np.maximum(sinr, 1e-9)
        spectral_efficiencies = np.log2(1 + sinr)
        
        return spectral_efficiencies
    

    def _update_control_states_downlink(self, downlink_power, H, lambd, action_penalty):
        self.time_step += 1
        SINR,channel_rates,interference_plus_noise = self.interference_SINR(H, downlink_power)
        self.link_delays = self.link_delay(channel_rates, self.traffic_loads)
        self.rtt_delays = np.zeros(self.num_users)
        self.throughput_per_user = channel_rates
        self.channel_rates = channel_rates

        previous_power = self.current_power_allocation.copy() if hasattr(self, 'current_power_allocation') else np.zeros(self.num_users)
        self.current_power_allocation = downlink_power.copy()
        # 修正：从H_matrix中提取每个设备从其服务基站的信道增益
        self.channel_gains = np.array([H[i, self.device_bs_assignment[i]] for i in range(self.num_users)])
        self.current_interference_noise = interference_plus_noise.copy()  # 存储当前干扰+噪声
        
        if np.any(previous_power > 0):
            self.update_neighbor_sets(previous_power)
        
        # 更新历史信息
        spectral_efficiency = channel_rates / self.bandwidth
        self.update_history(
            power_allocation=downlink_power,
            channel_gains=self.channel_gains,
            interference_noise=interference_plus_noise,
            spectral_efficiency=spectral_efficiency,
            H_matrix=H
        )
    
        self.last_sinr_per_link = SINR

        reward = self.compute_reward(downlink_power, channel_rates, interference_plus_noise)
        self.system_avg_rate = float(np.mean(channel_rates))
        self.current_reward = reward
        self.constraint_hist.append(np.atleast_1d(action_penalty))
        self.Lagrangian_hist.append(-reward + np.dot(lambd, action_penalty))
        self.downlink_constraint_dualvar = np.dot(lambd, action_penalty)

        done = False
        if self.time_step > self.T - 1:
            done = True

        return SINR, reward, done

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
        构建下行链路观察数据，基于新的局部状态结构
        新状态结构已经包含所有必要信息，不需要额外添加功率、延迟等特征
        """
        # 构建完整的局部状态信息（已包含所有必要特征）
        local_state = self._build_local_channel_state(channel_obs)
        
        # 如果有额外特征，添加进来（通常情况下不需要）
        if additional_features is not None:
            obs = np.hstack([local_state, additional_features.flatten()])
        else:
            obs = local_state
        
        # 验证观察向量维度
        expected_dim = self.local_state_dim
        actual_dim = len(obs)
        if actual_dim != expected_dim:
            print(f"Warning: Observation dimension mismatch. Expected: {expected_dim}, Got: {actual_dim}")
            print(f"Local state components: local_info({self.local_info_dim}), interference_neighbors({self.interference_neighbors_dim}), interfered_neighbors({self.interfered_neighbors_dim})")
        
        return obs
    
    def _build_local_channel_state(self, H):
        state_components = []
        
        # 1. 本地信息构建
        local_info = self._build_local_info(H)
        state_components.append(local_info)
        
        # 2. 干扰邻居信息构建
        interference_neighbors_info = self._build_interference_neighbors_info(H)
        state_components.append(interference_neighbors_info)
        
        # 3. 被干扰邻居信息构建
        interfered_neighbors_info = self._build_interfered_neighbors_info(H)
        state_components.append(interfered_neighbors_info)
        
        # 合并所有状态组件
        local_state = np.hstack(state_components)
        
        return local_state
    
    def _build_local_info(self, H):
        """
        构建本地信息：上一时隙功率 + 最近频谱效率 + 当前信道增益 + 历史信道增益 + 当前干扰+噪声 + 历史干扰+噪声（向量化版本）
        处理初始化时的负值噪声，保持其含义
        """
        # 向量化处理所有用户的本地信息
        
        # 1. 上一时隙的发射功率
        previous_powers = self.power_history[0, :]
        # 对于负值（初始化噪声），保持原值；对于实际功率值，进行归一化
        normalized_powers = np.where(
            previous_powers < 0, 
            previous_powers,  # 保持负值噪声
            previous_powers / (self.max_pwr_perplant + 1e-8)  # 归一化实际功率
        )
        
        # 2. PFS权重的倒数 1/w_i (表示对目标函数的潜在贡献)
        inverse_pfs_weights = 1.0 / (self.pfs_weights + 1e-8)
        # 归一化权重倒数
        max_inv_weight = np.max(inverse_pfs_weights) + 1e-8
        normalized_inverse_weights = inverse_pfs_weights / max_inv_weight
        
        # 3. 最近的频谱效率
        recent_spectral_efficiency = self.spectral_efficiency_history[0, :]
        bandwidth = self.bandwidth
        normalized_efficiency = np.where(
            recent_spectral_efficiency < 0,
            recent_spectral_efficiency,  # 保持负值噪声
            recent_spectral_efficiency / (bandwidth + 1e-8)  # 归一化实际效率
        )
        
        # 3. 当前信道增益 + 最近1次的历史信道增益
        # 组合当前信道增益和历史信道增益
        current_gains = getattr(self, 'channel_gains', np.zeros(self.num_users))
        historical_gains = self.channel_gains_history[0, :]  # 最近一次的历史增益
        
        # 创建组合的信道增益矩阵：(num_users, 2) - 当前增益 + 历史增益
        combined_gains = np.column_stack([current_gains, historical_gains])
        
        # 只对正值进行归一化，负值保持原样
        positive_gains = combined_gains[combined_gains >= 0]
        if len(positive_gains) > 0:
            max_gain = np.max(positive_gains) + 1e-8
            normalized_gains = np.where(
                combined_gains < 0,
                combined_gains,  # 保持负值噪声
                combined_gains / max_gain  # 归一化实际增益
            )
        else:
            normalized_gains = combined_gains  # 全是负值，保持原样
        
        # 4. 当前干扰+噪声 + 最近1次的历史干扰+噪声
        # 组合当前干扰信息和历史干扰信息
        current_interference = getattr(self, 'current_interference_noise', np.zeros(self.num_users))
        historical_interference = self.interference_noise_history[0, :]  # 最近一次的历史干扰
        
        # 创建组合的干扰矩阵：(num_users, 2) - 当前干扰 + 历史干扰
        combined_interference = np.column_stack([current_interference, historical_interference])
        
        # 只对正值进行归一化，负值保持原样
        positive_interference = combined_interference[combined_interference >= 0]
        if len(positive_interference) > 0:
            max_interference = np.max(positive_interference) + 1e-8
            normalized_interference = np.where(
                combined_interference < 0,
                combined_interference,  # 保持负值噪声
                combined_interference / max_interference  # 归一化实际干扰
            )
        else:
            normalized_interference = combined_interference  # 全是负值，保持原样
        
        # 向量化组合所有特征
        # 创建特征矩阵：(num_users, num_features_per_user)
        num_features_per_user = 1 + 1 + 1 + 2 + 2  # 功率 + 权重倒数 + 效率 + 2信道增益(当前+历史) + 2干扰(当前+历史)
        features_matrix = np.zeros((self.num_users, num_features_per_user))
        
        # 填充特征矩阵
        features_matrix[:, 0] = normalized_powers  # 功率
        features_matrix[:, 1] = normalized_inverse_weights  # PFS权重倒数
        features_matrix[:, 2] = normalized_efficiency  # 频谱效率
        features_matrix[:, 3:5] = normalized_gains  # 当前信道增益 + 历史信道增益
        features_matrix[:, 5:7] = normalized_interference  # 当前干扰+噪声 + 历史干扰+噪声
        
        # 展平为一维数组
        return features_matrix.flatten()
    
    def _build_interference_neighbors_info(self, H):
        """
        构建干扰邻居信息（根据图片定义）：
        对智能体i，观察从其邻居j接收到的干扰
        三个输入端口：
        1. 接收到的干扰: g_{j→i}^{(t-1)} * p_j^{(t-1)}
        2. 权重倒数: 1/w_j^{(t-1)}
        3. 网络贡献: g_{j→i}^{(t-1)}
        """
        # 预计算归一化参数
        max_power = self.max_pwr_perplant * np.max(np.abs(H)) ** 2 + 1e-8
        max_channel = np.max(np.abs(H)) + 1e-8
        
        # 定义没有邻居时的固定噪声值（3个特征）
        no_neighbor_noise = np.array([-0.1, -0.1, -0.1])
        
        # 初始化结果数组，先填充噪声值
        interference_info = np.tile(no_neighbor_noise, self.num_users * self.max_neighbors)
        
        # 预计算PFS权重倒数的归一化值（避免重复计算）
        all_weight_inv = 1.0 / (self.pfs_weights + 1e-8)
        max_weight_inv = np.max(all_weight_inv) + 1e-8
        normalized_weight_inv = all_weight_inv / max_weight_inv
        
        # 批量处理所有用户的干扰邻居信息
        for i in range(self.num_users):
            interference_neighbors = self.current_interference_neighbors.get(i, [])
            base_idx = i * self.max_neighbors * 3  # 每个干扰邻居有3个特征
            
            if not interference_neighbors:
                continue
                
            # 向量化处理当前用户的邻居信息
            num_neighbors = min(len(interference_neighbors), self.max_neighbors)
            
            # 提取邻居数据
            neighbor_data = interference_neighbors[:num_neighbors]
            neighbor_indices = [item[0] for item in neighbor_data]
            received_powers = np.array([item[1] for item in neighbor_data])
            channel_gains = np.array([item[2] for item in neighbor_data])
            
            # 向量化计算所有邻居的特征
            start_idx = base_idx
            end_idx = base_idx + num_neighbors * 3
            
            # 重新排列数据为(特征, 邻居)的形状
            neighbor_features = np.zeros(num_neighbors * 3)
            
            # 特征1: 接收到的干扰功率（归一化）
            neighbor_features[0::3] = received_powers / max_power
            
            # 特征2: 权重倒数（预计算的归一化值）
            neighbor_features[1::3] = normalized_weight_inv[neighbor_indices]
            
            # 特征3: 网络贡献（信道增益归一化）
            neighbor_features[2::3] = np.abs(channel_gains) / max_channel
            
            # 批量赋值
            interference_info[start_idx:start_idx + len(neighbor_features)] = neighbor_features
        
        return interference_info
    
    def _build_interfered_neighbors_info(self, H):
        # 预计算归一化参数
        max_interference = self.max_pwr_perplant * np.max(np.abs(H)) ** 2 + 1e-8
        max_channel = np.max(np.abs(H)) + 1e-8
        bandwidth = self.bandwidth
        
        # 定义没有被干扰邻居时的固定噪声值（确定性值，不是随机噪声）
        no_interfered_noise = np.array([-0.2, -0.2, -0.2, -0.2])  # 增加权重倒数维度
        
        # 初始化结果数组，先填充噪声值
        interfered_info = np.tile(no_interfered_noise, self.num_users * self.max_neighbors)
        
        # 预计算PFS权重倒数的归一化值（复用前面的计算）
        all_weight_inv = 1.0 / (self.pfs_weights + 1e-8)
        max_weight_inv = np.max(all_weight_inv) + 1e-8
        normalized_weight_inv = all_weight_inv / max_weight_inv
        
        # 批量处理所有用户的被干扰邻居信息
        for i in range(self.num_users):
            interfered_neighbors = self.current_interfered_neighbors.get(i, [])
            base_idx = i * self.max_neighbors * 4  # 每个被干扰邻居有4个特征
            
            if not interfered_neighbors:
                continue
                
            # 向量化处理当前用户的被干扰邻居信息
            num_neighbors = min(len(interfered_neighbors), self.max_neighbors)
            
            # 提取邻居数据
            neighbor_data = interfered_neighbors[:num_neighbors]
            neighbor_indices = [item[0] for item in neighbor_data]
            caused_interferences = np.array([item[1] for item in neighbor_data])
            channel_gains = np.array([item[2] for item in neighbor_data])
            
            # 获取邻居的频谱效率
            neighbor_efficiencies = np.zeros(num_neighbors)
            for idx, k in enumerate(neighbor_indices):
                if k < len(self.spectral_efficiency_history[0]):
                    neighbor_efficiencies[idx] = self.spectral_efficiency_history[0, k]
            
            # 向量化计算所有邻居的特征
            neighbor_features = np.zeros(num_neighbors * 4)
            
            # 特征1: 反馈信息（干扰功率归一化）
            neighbor_features[0::4] = caused_interferences / max_interference
            
            # 特征2: 权重倒数（预计算的归一化值）
            neighbor_features[1::4] = normalized_weight_inv[neighbor_indices]
            
            # 特征3: 信道增益（归一化）
            neighbor_features[2::4] = np.abs(channel_gains) / max_channel
            
            # 特征4: 网络贡献（频谱效率归一化）
            neighbor_features[3::4] = neighbor_efficiencies / (bandwidth + 1e-8)
            
            # 批量赋值
            start_idx = base_idx
            interfered_info[start_idx:start_idx + len(neighbor_features)] = neighbor_features
        
        return interfered_info

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
                 gamma=0.99, pl=2., pp=5., p0=1., num_features=1, scaling=False, force_max_power=force_max_power, weights=None,
                 snr_threshold_eta=10.0, max_neighbors=5):

        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, T=T,
                         gamma=gamma, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling, force_max_power=force_max_power, weights=weights,
                         snr_threshold_eta=snr_threshold_eta, max_neighbors=max_neighbors)

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
        self.channel_rates = np.zeros(self.num_users)
        
        agent_obs = self._get_downlink_obs(channel_obs)

        return agent_obs

    def step(self, action):
        # dual variable
        lambd = action[-self.constraint_dim:]

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.scale_power(power_action, self.force_max_power)

        # constraint violation
        action_penalty = (power_action.sum() - self.upperbound)  # constraint violation
        self.constraint_violation = action_penalty
        self.constraint_hist.append(np.atleast_1d(action_penalty))

        # 先计算当前状态下的性能指标
        SINR, one_step_reward, done = \
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
                 gamma=0.99, pl=2., pp=5., p0=1, num_features=1, scaling=False, force_max_power=force_max_power,
                 snr_threshold_eta=10.0, max_neighbors=5, train_length=200, bs_hexagon_radius=1000,
                 num_base_stations=3, bs_spacing=2000):

        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, T=T,
                         gamma=gamma, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling, force_max_power=force_max_power,
                         snr_threshold_eta=snr_threshold_eta, max_neighbors=max_neighbors,
                         train_length=train_length, bs_hexagon_radius=bs_hexagon_radius,
                         num_base_stations=num_base_stations, bs_spacing=bs_spacing)

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
        self.channel_rates = np.zeros(self.num_users)
        
        agent_obs = self._get_downlink_obs(channel_obs)

        return agent_obs

    def step(self, action):
        lambd = 0.
        action_penalty = 0.

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.normalize_scale_power(power_action, self.force_max_power)  # power decisions in [0, 1]

        # 先计算当前状态下的性能指标
        SINR, one_step_reward, done = \
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
            'interference_received_avg': 0.0,  # 简化处理
            'interference_caused_avg': 0.0,  # 简化处理
            
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


