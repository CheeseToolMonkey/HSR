import csv
import os
import numpy as np
from datetime import datetime

class LinkDataLogger:

    def __init__(self, log_dir="link_data_logs", num_users=30, filename=None, log_interval=50, n_actors=1):
        """
        初始化数据记录器
        
        Args:
            log_dir: 日志文件保存目录
            num_users: 用户数量
            filename: 指定的文件名，如果为None则使用默认名称
            log_interval: 记录间隔，每隔多少episode记录一次完整数据（默认50）
            n_actors: actor数量，用于区分不同子环境
        """
        self.log_dir = log_dir
        self.num_users = num_users
        self.log_interval = log_interval
        self.n_actors = n_actors
        self.last_logged_episode = -1  # 记录上次记录的episode
        self.global_timestep = 0       # 全局时间步计数
        self.current_logging_episode = -1  # 当前正在记录的episode
        self.episode_timesteps_logged = 0  # 当前episode已记录的时间步数
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置文件名
        if filename is None:
            # 使用默认文件名，不包含时间戳
            self.data_file = os.path.join(log_dir, "link_data.csv")
        else:
            # 使用指定的文件名
            if not filename.endswith('.csv'):
                filename += '.csv'
            self.data_file = os.path.join(log_dir, filename)
        
        # 初始化CSV文件
        self._init_csv_file()
        
    def _init_csv_file(self):
        """初始化单个CSV文件，包含所有数据"""
        # 检查文件是否已存在
        file_exists = os.path.exists(self.data_file)
        
        # 创建或追加到CSV文件
        mode = 'a' if file_exists else 'w'
        with open(self.data_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                header = [
                    'episode', 'timestep', 'user_id', 'traffic_type', 'sub_env_id',
                    'power_allocation', 'agent_satellite_selection', 
                    'direct_channel_gain', 'interference_channel_gain', 
                    'interference_received', 'interference_caused_to_weak',
                    'channel_rate', 'sinr', 'pdr',
                    'normalized_power_allocation', 'system_avg_rate', 'weak_interference_penalty', 'reward'
                ]
                writer.writerow(header)
    
    def log_step_data(self, timestep, env, power_allocations, satellite_selections, 
                      obs_data=None, env_id=None):
        """
        记录一个时间步的所有数据到单个CSV文件
        
        Args:
            timestep: 当前时间步
            env: 环境对象
            power_allocations: 功率分配数组
            satellite_selections: 卫星选择数组（如果适用）
            obs_data: 环境的obs数据，如果为None则从环境中获取
            env_id: 子环境ID，如果为None则从环境中获取
            
        Note:
            观察空间结构（一维向量）：
            1. 信道矩阵：num_users^2 维
            2. 功率分配：num_users 维
            3. 接收干扰：num_users 维
            4. 造成干扰：num_users 维
            5. 信道速率：num_users 维
        """
        # 更新全局时间步计数
        self.global_timestep += 1
        
        # 获取当前episode（如果环境有episode计数）
        current_episode = getattr(env, 'current_episode', 0)
        
        # 检查是否需要记录数据（每隔log_interval个episode记录一次）
        should_log_episode = (current_episode % self.log_interval == 0)
        
        # 如果是新的需要记录的episode，开始记录
        if should_log_episode and current_episode != self.current_logging_episode:
            self.current_logging_episode = current_episode
            self.episode_timesteps_logged = 0
        
        # 如果当前episode正在记录中，则记录数据
        if current_episode == self.current_logging_episode:
            self.episode_timesteps_logged += 1

            # 如果记录完整个episode，重置状态
            if self.episode_timesteps_logged >= 30:
                self.current_logging_episode = -1
                self.episode_timesteps_logged = 0
        else:
            return  # 跳过记录
        
        with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 如果没有提供obs_data，尝试从环境中获取
            if obs_data is None and hasattr(env, 'current_obs'):
                obs_data = env.current_obs
            
            for user_id in range(min(self.num_users, getattr(env, 'num_users', self.num_users))):
                # 获取用户流量类型
                traffic_type = getattr(env, 'user_traffic_types', [0] * self.num_users)[user_id]
                
                # 初始化变量
                direct_channel_gain = 0.0
                interference_channel_gain = 0.0
                interference_received = 0.0
                interference_caused_to_weak = 0.0
                channel_rate = 0.0
                sinr = 0.0
                pdr = 0.0
                normalized_power_allocation = 0.0
                
                # 从环境直接获取数据（优先方案）
                if hasattr(env, 'H') and env.H is not None:
                    H_matrix = env.H
                    # 检查矩阵维度，处理非方阵情况
                    if H_matrix.shape[0] == H_matrix.shape[1]:
                        # 方阵情况：H_matrix[i,j] 表示用户j对用户i的干扰
                        direct_channel_gain = H_matrix[user_id, user_id]
                        # 干扰链路增益：其他用户对该用户的平均干扰
                        interference_column = H_matrix[:, user_id]
                        interference_column[user_id] = 0  # 排除自身
                        interference_channel_gain = np.mean(interference_column)
                    else:
                        # 非方阵情况：H_matrix[i,j] 表示基站j到用户i的信道增益
                        # 获取用户对应的基站索引
                        if hasattr(env, 'device_bs_assignment') and user_id < len(env.device_bs_assignment):
                            bs_index = env.device_bs_assignment[user_id]
                            if bs_index < H_matrix.shape[1]:
                                direct_channel_gain = H_matrix[user_id, bs_index]
                            else:
                                direct_channel_gain = 0.0
                        else:
                            # 如果没有基站分配信息，使用第一个基站
                            direct_channel_gain = H_matrix[user_id, 0] if H_matrix.shape[1] > 0 else 0.0
                        
                        # 干扰链路增益：计算其他基站对该用户的平均干扰
                        if H_matrix.shape[1] > 1:
                            interference_row = H_matrix[user_id, :]
                            # 排除用户自己的基站
                            if hasattr(env, 'device_bs_assignment') and user_id < len(env.device_bs_assignment):
                                bs_index = env.device_bs_assignment[user_id]
                                interference_row[bs_index] = 0
                            interference_channel_gain = np.mean(interference_row)
                        else:
                            interference_channel_gain = 0.0
                
                # 从环境属性获取新的观察数据
                if hasattr(env, 'interference_received') and user_id < len(env.interference_received):
                    interference_received = env.interference_received[user_id]
                
                if hasattr(env, 'interference_caused_to_weak') and user_id < len(env.interference_caused_to_weak):
                    interference_caused_to_weak = env.interference_caused_to_weak[user_id]
                    
                if hasattr(env, 'channel_rates') and user_id < len(env.channel_rates):
                    channel_rate = env.channel_rates[user_id]
                    
                if hasattr(env, 'current_power_allocation') and user_id < len(env.current_power_allocation):
                    normalized_power_allocation = env.current_power_allocation[user_id] / (env.max_pwr_perplant + 1e-8)
                
                # 从观察空间提取数据（备选方案）
                if obs_data is not None and len(obs_data) > 0:
                    # 新的观察空间结构：
                    # [信道矩阵(num_users^2), 功率分配(num_users), 接收干扰(num_users), 
                    #  造成干扰(num_users), 信道速率(num_users)]
                    
                    channel_dim = env.num_users ** 2
                    power_start = channel_dim
                    interference_recv_start = power_start + env.num_users
                    interference_caused_start = interference_recv_start + env.num_users
                    channel_rate_start = interference_caused_start + env.num_users
                    
                    # 如果从环境属性中无法获取，则从观察空间提取
                    if user_id < env.num_users and len(obs_data) >= channel_rate_start + env.num_users:
                        # 从观察空间获取归一化功率分配
                        if power_start + user_id < len(obs_data):
                            normalized_power_allocation = obs_data[power_start + user_id]
                        
                        # 从观察空间获取接收干扰（归一化后的）
                        if interference_recv_start + user_id < len(obs_data):
                            interference_received = obs_data[interference_recv_start + user_id]
                        
                        # 从观察空间获取造成的干扰（归一化后的）
                        if interference_caused_start + user_id < len(obs_data):
                            interference_caused_to_weak = obs_data[interference_caused_start + user_id]
                        
                        # 从观察空间获取信道速率（归一化后的）
                        if channel_rate_start + user_id < len(obs_data):
                            channel_rate = obs_data[channel_rate_start + user_id]
                        
                        # 从观察空间提取信道矩阵以获取直接信道增益
                        if not hasattr(env, 'H') or env.H is None:
                            if channel_dim > 0 and len(obs_data) >= channel_dim:
                                # 根据环境类型确定矩阵维度
                                if hasattr(env, 'num_base_stations'):
                                    # 多基站环境：矩阵维度为 (num_users, num_base_stations)
                                    expected_dim = env.num_users * env.num_base_stations
                                    if len(obs_data) >= expected_dim:
                                        H_matrix = obs_data[:expected_dim].reshape(env.num_users, env.num_base_stations)
                                        # 反归一化信道矩阵（乘以最大值）
                                        max_channel = np.max(np.abs(H_matrix)) + 1e-8
                                        H_matrix = H_matrix * max_channel
                                        
                                        # 获取用户对应的基站索引
                                        if hasattr(env, 'device_bs_assignment') and user_id < len(env.device_bs_assignment):
                                            bs_index = env.device_bs_assignment[user_id]
                                            if bs_index < H_matrix.shape[1]:
                                                direct_channel_gain = H_matrix[user_id, bs_index]
                                            else:
                                                direct_channel_gain = 0.0
                                        else:
                                            direct_channel_gain = H_matrix[user_id, 0] if H_matrix.shape[1] > 0 else 0.0
                                        
                                        # 干扰链路增益
                                        if H_matrix.shape[1] > 1:
                                            interference_row = H_matrix[user_id, :]
                                            if hasattr(env, 'device_bs_assignment') and user_id < len(env.device_bs_assignment):
                                                bs_index = env.device_bs_assignment[user_id]
                                                interference_row[bs_index] = 0
                                            interference_channel_gain = np.mean(interference_row)
                                        else:
                                            interference_channel_gain = 0.0
                                else:
                                    # 传统环境：方阵 (num_users, num_users)
                                    H_matrix = obs_data[:channel_dim].reshape(env.num_users, env.num_users)
                                    # 反归一化信道矩阵（乘以最大值）
                                    max_channel = np.max(np.abs(H_matrix)) + 1e-8
                                    H_matrix = H_matrix * max_channel
                                    direct_channel_gain = H_matrix[user_id, user_id]
                                    interference_column = H_matrix[:, user_id]
                                    interference_column[user_id] = 0
                                    interference_channel_gain = np.mean(interference_column)
                
                # 获取系统级指标
                system_avg_rate = getattr(env, 'system_avg_rate', 0.0)
                weak_interference_penalty = getattr(env, 'weak_interference_penalty', 0.0)
                current_reward = getattr(env, 'current_reward', 0.0)
                
                # 获取SINR和PDR（如果环境有保存）
                if hasattr(env, 'last_sinr_per_link') and user_id < len(env.last_sinr_per_link):
                    sinr = env.last_sinr_per_link[user_id]
                if hasattr(env, 'last_pdr_per_link') and user_id < len(env.last_pdr_per_link):
                    pdr = env.last_pdr_per_link[user_id]
                
                # 写入一行数据
                # 获取子环境ID（优先使用传入的env_id，否则从环境中获取）
                if env_id is not None:
                    sub_env_id = env_id
                else:
                    sub_env_id = getattr(env, 'env_id', 0)
                
                # 确保satellite_selections数组有效
                satellite_selection = 0.0
                if satellite_selections is not None and user_id < len(satellite_selections):
                    satellite_selection = satellite_selections[user_id]
                
                # 安全获取功率分配值
                power_allocation = power_allocations[user_id] if user_id < len(power_allocations) else 0.0
                
                writer.writerow([
                    current_episode,             # 当前episode
                    timestep,                    # 时间步
                    user_id,                     # 用户ID
                    traffic_type,                # 流量类型 (0=Others, 1=HVFT)
                    sub_env_id,                  # 子环境ID
                    power_allocation,            # 功率分配
                    satellite_selection,         # 智能体卫星选择
                    direct_channel_gain,         # 直接链路增益
                    interference_channel_gain,   # 干扰链路增益
                    interference_received,       # 接收到的干扰
                    interference_caused_to_weak, # 对弱信道设备的干扰
                    channel_rate,               # 信道速率
                    sinr,                       # SINR
                    pdr,                        # 包传递率
                    normalized_power_allocation, # 归一化功率分配
                    system_avg_rate,            # 系统平均信道速率
                    weak_interference_penalty,  # 弱信道干扰惩罚
                    current_reward              # 当前奖励
                ])
    
    def close(self):
        """关闭所有文件"""
        pass  # CSV文件会在写入时自动关闭
    
    def get_log_files(self):
        """获取日志文件路径"""
        return {
            'link_data': self.data_file
        }
    
    def get_logging_info(self):
        """获取记录信息"""
        return {
            'log_interval': self.log_interval,
            'last_logged_episode': self.last_logged_episode,
            'global_timestep': self.global_timestep,
            'data_file': self.data_file
        }
    
    def set_log_interval(self, interval):
        """设置记录间隔"""
        self.log_interval = interval
        print(f"记录间隔已设置为: 每隔 {interval} 个episode记录一次")
    
    def get_data_structure_info(self):
        """获取数据结构信息"""
        info = {
            'csv_columns': [
                'episode', 'timestep', 'user_id', 'traffic_type', 'sub_env_id',
                'power_allocation', 'agent_satellite_selection', 
                'direct_channel_gain', 'interference_channel_gain', 
                'interference_received', 'interference_caused_to_weak',
                'channel_rate', 'sinr', 'pdr',
                'normalized_power_allocation', 'system_avg_rate', 'weak_interference_penalty', 'reward'
            ],
            'observation_space_structure': {
                'channel_matrix': 'num_users^2 维 - 完整信道状态矩阵',
                'power_allocation': 'num_users 维 - 当前功率分配',
                'interference_received': 'num_users 维 - 接收到的干扰',
                'interference_caused_to_weak': 'num_users 维 - 对弱信道设备的干扰',
                'channel_rates': 'num_users 维 - 信道速率'
            },
            'data_sources': {
                'from_environment_attributes': ['H', 'channel_rates', 'interference_received', 'interference_caused_to_weak', 'current_power_allocation'],
                'from_observation_space': ['normalized values when environment attributes unavailable'],
                'system_level_metrics': ['system_avg_rate', 'weak_interference_penalty', 'current_reward']
            }
        }
        return info 