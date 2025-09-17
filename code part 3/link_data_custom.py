import csv
import os
import numpy as np
from datetime import datetime

class LinkDataLogger:

    def __init__(self, log_dir="link_data_logs", num_users=30, filename=None, log_interval=50):
        """
        初始化数据记录器
        
        Args:
            log_dir: 日志文件保存目录
            num_users: 用户数量
            filename: 指定的文件名，如果为None则使用默认名称
            log_interval: 记录间隔，每隔多少episode记录一次完整数据（默认50）
        """
        self.log_dir = log_dir
        self.num_users = num_users
        self.log_interval = log_interval
        self.last_logged_episode = -1  # 记录上次记录的episode
        self.global_timestep = 0       # 全局时间步计数
        
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
                    'episode', 'timestep', 'user_id', 'traffic_type',
                    'power_allocation', 'agent_satellite_selection', 
                    'actual_link_satellite_status', 'direct_channel_gain', 'interference_channel_gain',
                    'transmission_delay', 'rtt_delay', 'final_delay', 'log2_sinr', 
                    'hvft_accumulated_data', 'hvft_accumulation_rounds',
                    'hvft_transmission_decision', 'hvft_satellite_transmission',
                    'normalized_power_allocation'
                ]
                writer.writerow(header)
    
    def log_step_data(self, timestep, env, power_allocations, satellite_selections, 
                      obs_data=None):
        """
        记录一个时间步的所有数据到单个CSV文件
        
        Args:
            timestep: 当前时间步
            env: 环境对象
            power_allocations: 功率分配数组
            satellite_selections: 卫星选择数组
            obs_data: 环境的obs数据，如果为None则从环境中获取
        """
        # 更新全局时间步计数
        self.global_timestep += 1
        
        # 获取当前episode（如果环境有episode计数）
        current_episode = getattr(env, 'current_episode', 0)
        
        # 检查是否需要记录数据（每隔log_interval个episode记录一次）
        if current_episode % self.log_interval != 0 or current_episode == self.last_logged_episode:
            return  # 跳过记录
        
        # 更新上次记录的episode
        self.last_logged_episode = current_episode
        
        with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 如果没有提供obs_data，尝试从环境中获取
            if obs_data is None and hasattr(env, 'current_obs'):
                obs_data = env.current_obs
            
            for user_id in range(self.num_users):
                # 获取用户流量类型
                traffic_type = env.user_traffic_types[user_id] if hasattr(env, 'user_traffic_types') else 0
                
                # 从obs中提取数据（如果可用）
                direct_channel_gain = 0.0
                interference_channel_gain = 0.0
                transmission_delay = 0.0
                rtt_delay = 0.0
                final_delay = 0.0
                link_satellite_status = 0.0
                log2_sinr = 0.0
                hvft_accumulated_data = 0.0
                hvft_accumulation_rounds = 0.0
                hvft_transmission_decision = 0.0
                hvft_satellite_transmission = 0.0
                normalized_power_allocation = 0.0
                
                if obs_data is not None and len(obs_data) > 0:
                    # 解析obs数据
                    # obs结构：[channel_obs, user_traffic_types, transmission_delays, rtt_delays, 
                    #          link_satellite_status, log2_sinr, hvft_accumulation_info, normalized_updated_power]
                    
                    # 计算各个部分的起始索引
                    channel_dim = env.num_users ** 2
                    traffic_types_start = channel_dim
                    delays_start = traffic_types_start + env.num_users
                    rtt_start = delays_start + env.num_users
                    satellite_start = rtt_start + env.num_users
                    sinr_start = satellite_start + env.num_users
                    hvft_start = sinr_start + env.num_users
                    power_start = hvft_start + env.num_users * 4  # 每个用户4个HVFT信息
                    
                    # 提取用户特定的数据
                    if user_id < env.num_users:
                        # 提取H矩阵并重塑为矩阵形式
                        if channel_dim > 0 and len(obs_data) >= channel_dim:
                            H_matrix = obs_data[:channel_dim].reshape(env.num_users, env.num_users)
                            
                            # 直接链路增益（对角线元素）
                            direct_channel_gain = H_matrix[user_id, user_id]
                            
                            # 干扰链路增益（该用户对其他用户的干扰）
                            interference_channel_gain = np.mean(H_matrix[user_id, :]) - direct_channel_gain
                        else:
                            direct_channel_gain = 0.0
                            interference_channel_gain = 0.0
                        
                        # 传输延迟
                        if delays_start + user_id < len(obs_data):
                            transmission_delay = obs_data[delays_start + user_id]
                        
                        # RTT延迟
                        if rtt_start + user_id < len(obs_data):
                            rtt_delay = obs_data[rtt_start + user_id]
                        
                        # 从环境直接获取最终延迟（环境计算的结果）
                        final_delay = env.current_link_delays[user_id]
                        
                        # 卫星状态
                        if satellite_start + user_id < len(obs_data):
                            link_satellite_status = obs_data[satellite_start + user_id]
                        
                        # SINR
                        if sinr_start + user_id < len(obs_data):
                            log2_sinr = obs_data[sinr_start + user_id]
                        
                        # HVFT信息（每个用户4个值：累积数据、累积回合数、传输决策、卫星传输标记）
                        hvft_user_start = hvft_start + user_id * 4
                        if hvft_user_start + 3 < len(obs_data):  # 确保有足够的4个值
                            hvft_accumulated_data = obs_data[hvft_user_start]
                            hvft_accumulation_rounds = obs_data[hvft_user_start + 1]
                            hvft_transmission_decision = obs_data[hvft_user_start + 2]
                            hvft_satellite_transmission = obs_data[hvft_user_start + 3]
                        
                        # 归一化功率分配
                        if power_start + user_id < len(obs_data):
                            normalized_power_allocation = obs_data[power_start + user_id]
                
                # 写入一行数据
                writer.writerow([
                    current_episode,             # 当前episode
                    timestep,                    # 时间步
                    user_id,                     # 用户ID
                    traffic_type,                # 流量类型 (0=Others, 1=HVFT)
                    power_allocations[user_id],  # 功率分配
                    satellite_selections[user_id], # 智能体卫星选择 (0=地面, 1=卫星)
                    link_satellite_status,       # 实际卫星链路状态（从obs中提取）
                    direct_channel_gain,         # 直接链路增益（从H矩阵对角线提取）
                    interference_channel_gain,   # 干扰链路增益（从H矩阵非对角线提取）
                    transmission_delay,          # 传输延迟（从obs中提取）
                    rtt_delay,                   # RTT延迟（从obs中提取）
                    final_delay,                 # 最终延迟（从环境直接获取）
                    log2_sinr,                  # SINR（从obs中提取）
                    hvft_accumulated_data,      # HVFT累积数据（从obs中提取）
                    hvft_accumulation_rounds,   # HVFT累积回合数（从obs中提取）
                    hvft_transmission_decision, # HVFT传输决策（从obs中提取）
                    hvft_satellite_transmission, # HVFT卫星传输标记（从obs中提取）
                    normalized_power_allocation  # 归一化功率分配（从obs中提取）
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