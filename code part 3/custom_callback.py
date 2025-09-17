import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ThroughputLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ThroughputLoggingCallback, self).__init__(verbose)
        # For accumulating step-wise data for episodic logging if needed
        self.current_rollout_hvft = []
        self.current_rollout_others = []
        self.current_rollout_total = []
        self.current_rollout_pdr = []
        self.current_rollout_SINR = []
        self.current_rollout_delays = []
        self.current_rollout_ground_power = []
        self.current_rollout_satellite_assisted_users = []
        self.current_rollout_power_efficiency = []  # 添加功率效率记录
        
        # 添加Others流量延迟统计记录
        self.current_rollout_others_delay_sum = []
        self.current_rollout_others_delay_mean = []
        self.current_rollout_others_delay_min = []
        self.current_rollout_others_delay_max = []
        self.current_rollout_others_delay_std = []
        self.current_rollout_others_count = []
        
        # 添加HVFT流量延迟统计记录
        self.current_rollout_hvft_delay_sum = []
        self.current_rollout_hvft_delay_mean = []
        self.current_rollout_hvft_delay_min = []
        self.current_rollout_hvft_delay_max = []
        self.current_rollout_hvft_delay_std = []
        self.current_rollout_hvft_count = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])

        step_hvft_sum = 0
        step_others_sum = 0
        step_total_sum = 0
        step_pdr_sum = 0
        step_sinr_sum = 0
        step_delays_sum = 0
        step_ground_power_sum = 0
        step_satellite_assisted_users_count = 0
        step_power_efficiency_sum = 0  # 添加功率效率总和
        
        # 添加Others流量延迟统计变量
        step_others_delay_sum_sum = 0
        step_others_delay_mean_sum = 0
        step_others_delay_min_sum = 0
        step_others_delay_max_sum = 0
        step_others_delay_std_sum = 0
        step_others_count_sum = 0
        
        # 添加HVFT流量延迟统计变量
        step_hvft_delay_sum_sum = 0
        step_hvft_delay_mean_sum = 0
        step_hvft_delay_min_sum = 0
        step_hvft_delay_max_sum = 0
        step_hvft_delay_std_sum = 0
        step_hvft_count_sum = 0

        num_envs_with_data = 0
        num_envs_with_pdr = 0
        num_envs_with_SINR = 0
        num_envs_with_delays = 0
        num_envs_with_ground_power = 0
        num_envs_with_satellite_assisted_users = 0
        num_envs_with_power_efficiency = 0  # 添加功率效率环境计数
        
        # 添加Others流量延迟统计环境计数
        num_envs_with_others_delay = 0
        num_envs_with_hvft_delay = 0

        for info in infos:
            if isinstance(info, dict):
                # info 字典中直接包含了当前 step 环境的 HVFT 和 others 吞吐量
                hvft_val = info.get("throughput_hvft")
                others_val = info.get("throughput_others")
                pdr_val = info.get("link_pdr_avg")
                SINR_val = info.get("sinr_sum")
                delays_val = info.get("delay_sum")
                ground_power_val = info.get("total_power")
                satellite_assisted_users_val = info.get("satellite_assisted_users_count")
                power_efficiency_val = info.get("power_efficiency") # 获取功率效率
                
                # 获取Others流量延迟统计
                others_delay_sum_val = info.get("others_delay_sum")
                others_delay_mean_val = info.get("others_delay_mean")
                others_delay_min_val = info.get("others_delay_min")
                others_delay_max_val = info.get("others_delay_max")
                others_delay_std_val = info.get("others_delay_std")
                others_count_val = info.get("others_count")
                
                # 获取HVFT流量延迟统计
                hvft_delay_sum_val = info.get("hvft_delay_sum")
                hvft_delay_mean_val = info.get("hvft_delay_mean")
                hvft_delay_min_val = info.get("hvft_delay_min")
                hvft_delay_max_val = info.get("hvft_delay_max")
                hvft_delay_std_val = info.get("hvft_delay_std")
                hvft_count_val = info.get("hvft_count")

                if hvft_val is not None:
                    step_hvft_sum += hvft_val
                    self.current_rollout_hvft.append(hvft_val)  # 用于rollout结束时计算平均
                if others_val is not None:
                    step_others_sum += others_val
                    self.current_rollout_others.append(others_val)
                if pdr_val is not None:
                    step_pdr_sum += pdr_val
                    self.current_rollout_pdr.append(pdr_val)
                    num_envs_with_pdr += 1
                if SINR_val is not None:
                    step_sinr_sum += SINR_val
                    self.current_rollout_SINR.append(SINR_val)
                    num_envs_with_SINR += 1
                if delays_val is not None:
                    step_delays_sum += delays_val
                    self.current_rollout_delays.append(delays_val)
                    num_envs_with_delays += 1
                if ground_power_val is not None:
                    step_ground_power_sum += ground_power_val
                    self.current_rollout_ground_power.append(ground_power_val)
                    num_envs_with_ground_power += 1
                if satellite_assisted_users_val is not None:
                    step_satellite_assisted_users_count += satellite_assisted_users_val
                    self.current_rollout_satellite_assisted_users.append(satellite_assisted_users_val)
                    num_envs_with_satellite_assisted_users += 1
                if power_efficiency_val is not None:
                    step_power_efficiency_sum += power_efficiency_val
                    self.current_rollout_power_efficiency.append(power_efficiency_val)
                    num_envs_with_power_efficiency += 1
                
                # 处理Others流量延迟统计
                if others_delay_sum_val is not None:
                    step_others_delay_sum_sum += others_delay_sum_val
                    self.current_rollout_others_delay_sum.append(others_delay_sum_val)
                    num_envs_with_others_delay += 1
                if others_delay_mean_val is not None:
                    step_others_delay_mean_sum += others_delay_mean_val
                    self.current_rollout_others_delay_mean.append(others_delay_mean_val)
                if others_delay_min_val is not None:
                    step_others_delay_min_sum += others_delay_min_val
                    self.current_rollout_others_delay_min.append(others_delay_min_val)
                if others_delay_max_val is not None:
                    step_others_delay_max_sum += others_delay_max_val
                    self.current_rollout_others_delay_max.append(others_delay_max_val)
                if others_delay_std_val is not None:
                    step_others_delay_std_sum += others_delay_std_val
                    self.current_rollout_others_delay_std.append(others_delay_std_val)
                if others_count_val is not None:
                    step_others_count_sum += others_count_val
                    self.current_rollout_others_count.append(others_count_val)
                
                # 处理HVFT流量延迟统计
                if hvft_delay_sum_val is not None:
                    step_hvft_delay_sum_sum += hvft_delay_sum_val
                    self.current_rollout_hvft_delay_sum.append(hvft_delay_sum_val)
                    num_envs_with_hvft_delay += 1
                if hvft_delay_mean_val is not None:
                    step_hvft_delay_mean_sum += hvft_delay_mean_val
                    self.current_rollout_hvft_delay_mean.append(hvft_delay_mean_val)
                if hvft_delay_min_val is not None:
                    step_hvft_delay_min_sum += hvft_delay_min_val
                    self.current_rollout_hvft_delay_min.append(hvft_delay_min_val)
                if hvft_delay_max_val is not None:
                    step_hvft_delay_max_sum += hvft_delay_max_val
                    self.current_rollout_hvft_delay_max.append(hvft_delay_max_val)
                if hvft_delay_std_val is not None:
                    step_hvft_delay_std_sum += hvft_delay_std_val
                    self.current_rollout_hvft_delay_std.append(hvft_delay_std_val)
                if hvft_count_val is not None:
                    step_hvft_count_sum += hvft_count_val
                    self.current_rollout_hvft_count.append(hvft_count_val)

                # 计算单个环境的总吞吐量并累加,计算pdr均值
                if hvft_val is not None and others_val is not None:
                    current_env_total = hvft_val + others_val
                    step_total_sum += current_env_total
                    self.current_rollout_total.append(current_env_total)
                    self.logger.record(f"throughput/step_env_total_{info.get('env_id', num_envs_with_data)}",
                                       current_env_total)  # 可选：记录每个环境的总吞吐量
                    num_envs_with_data += 1
                elif hvft_val is not None:  # 只有hvft
                    step_total_sum += hvft_val
                    self.current_rollout_total.append(hvft_val)
                    self.logger.record(f"throughput/step_env_total_{info.get('env_id', num_envs_with_data)}", hvft_val)
                    num_envs_with_data += 1
                elif others_val is not None:  # 只有others
                    step_total_sum += others_val
                    self.current_rollout_total.append(others_val)
                    self.logger.record(f"throughput/step_env_total_{info.get('env_id', num_envs_with_data)}",
                                       others_val)
                    num_envs_with_data += 1

        # 记录当前 step 所有环境的吞吐量总和
        if num_envs_with_data > 0:  # 确保至少有一个环境返回了数据
            self.logger.record("throughput/step_sum_hvft", step_hvft_sum)
            self.logger.record("throughput/step_sum_others", step_others_sum)
            self.logger.record("throughput/step_sum_total", step_total_sum)  # 记录总和
            
            # 记录HSR环境特有的指标
            self.logger.record("hsr/step_sum_throughput", step_total_sum)  # HSR总吞吐量
            self.logger.record("hsr/step_avg_throughput_per_user", step_total_sum / (num_envs_with_data * 30))  # 平均每用户吞吐量
        # 记录当前 step 所有环境的pdr均值
        if num_envs_with_pdr > 0:
            self.logger.record("pdr/step_mean_pdr", step_pdr_sum / num_envs_with_pdr)  # 记录所有环境PDR的均值
            self.logger.record("pdr/step_sum_pdr", step_pdr_sum)  # 记录所有环境PDR的总和 (如果需要)
        # 记录当前 step 所有环境的SINR均值

        if num_envs_with_SINR > 0:
            self.logger.record("sinr/step_mean_sinr", step_sinr_sum / num_envs_with_SINR)  # Log mean SINR
            self.logger.record("sinr/step_sum_sinr", step_sinr_sum)

        if num_envs_with_delays > 0:
            self.logger.record("delay/step_mean_delay", step_delays_sum / num_envs_with_delays)  # Log mean delay
            self.logger.record("delay/step_sum_delay", step_delays_sum)
            
            # 记录HSR环境特有的延迟指标
            self.logger.record("hsr/step_mean_delay", step_delays_sum / num_envs_with_delays)
            self.logger.record("hsr/step_sum_delay", step_delays_sum)

        if num_envs_with_ground_power > 0:
            self.logger.record("power/step_sum_power", step_ground_power_sum )
        if num_envs_with_satellite_assisted_users > 0:
            self.logger.record("satellite/step_mean_satellite", step_satellite_assisted_users_count
                               / num_envs_with_satellite_assisted_users)  # Log mean satellite
            self.logger.record("satellite/step_sum_satellite", step_satellite_assisted_users_count)
        if num_envs_with_power_efficiency > 0:
            self.logger.record("power_efficiency/step_mean_power_efficiency", step_power_efficiency_sum / num_envs_with_power_efficiency)
            self.logger.record("power_efficiency/step_sum_power_efficiency", step_power_efficiency_sum)
            
            # 记录HSR环境特有的功率效率指标
            self.logger.record("hsr/step_mean_power_efficiency", step_power_efficiency_sum / num_envs_with_power_efficiency)
            self.logger.record("hsr/step_sum_power_efficiency", step_power_efficiency_sum)
        
        # 记录Others流量延迟统计
        if num_envs_with_others_delay > 0:
            self.logger.record("others_delay/step_mean_others_delay_sum", step_others_delay_sum_sum / num_envs_with_others_delay)
            self.logger.record("others_delay/step_mean_others_delay_mean", step_others_delay_mean_sum / num_envs_with_others_delay)
            self.logger.record("others_delay/step_mean_others_delay_min", step_others_delay_min_sum / num_envs_with_others_delay)
            self.logger.record("others_delay/step_mean_others_delay_max", step_others_delay_max_sum / num_envs_with_others_delay)
            self.logger.record("others_delay/step_mean_others_delay_std", step_others_delay_std_sum / num_envs_with_others_delay)
            self.logger.record("others_delay/step_mean_others_count", step_others_count_sum / num_envs_with_others_delay)
            self.logger.record("others_delay/step_sum_others_delay_sum", step_others_delay_sum_sum)
            self.logger.record("others_delay/step_sum_others_count", step_others_count_sum)
        
        # 记录HVFT流量延迟统计
        if num_envs_with_hvft_delay > 0:
            self.logger.record("hvft_delay/step_mean_hvft_delay_sum", step_hvft_delay_sum_sum / num_envs_with_hvft_delay)
            self.logger.record("hvft_delay/step_mean_hvft_delay_mean", step_hvft_delay_mean_sum / num_envs_with_hvft_delay)
            self.logger.record("hvft_delay/step_mean_hvft_delay_min", step_hvft_delay_min_sum / num_envs_with_hvft_delay)
            self.logger.record("hvft_delay/step_mean_hvft_delay_max", step_hvft_delay_max_sum / num_envs_with_hvft_delay)
            self.logger.record("hvft_delay/step_mean_hvft_delay_std", step_hvft_delay_std_sum / num_envs_with_hvft_delay)
            self.logger.record("hvft_delay/step_mean_hvft_count", step_hvft_count_sum / num_envs_with_hvft_delay)
            self.logger.record("hvft_delay/step_sum_hvft_delay_sum", step_hvft_delay_sum_sum)
            self.logger.record("hvft_delay/step_sum_hvft_count", step_hvft_count_sum)

        for i, done in enumerate(self.locals['dones']):
            if done:
                # info_done = self.locals['infos'][i] # 获取对应完成环境的info
                # if "throughput_hvft" in info_done: # 假设这是episode的最终值
                #     self.logger.record(f"throughput/ep_final_hvft_env{i}", info_done["throughput_hvft"])
                # if "throughput_others" in info_done:
                #     self.logger.record(f"throughput/ep_final_others_env{i}", info_done["throughput_others"])
                # if "throughput_hvft" in info_done and "throughput_others" in info_done :
                #      self.logger.record(f"throughput/ep_final_total_env{i}",info_done["throughput_hvft"] + info_done["throughput_others"])
                # 考虑使用Monitor wrapper
                pass

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Log the sum of throughputs over the entire rollout.
        """
        if self.current_rollout_hvft:
            self.logger.record("throughput/rollout_sum_hvft", np.sum(self.current_rollout_hvft))
            self.logger.record("throughput/rollout_mean_hvft_per_step", np.mean(self.current_rollout_hvft))
            self.current_rollout_hvft = []  # Reset for next rollout
        if self.current_rollout_others:
            self.logger.record("throughput/rollout_sum_others", np.sum(self.current_rollout_others))
            self.logger.record("throughput/rollout_mean_others_per_step", np.mean(self.current_rollout_others))
            self.current_rollout_others = []
        if self.current_rollout_total:
            self.logger.record("throughput/rollout_sum_total", np.sum(self.current_rollout_total))  # 记录整个rollout的总吞吐量和
            self.logger.record("throughput/rollout_mean_total_per_step", np.mean(self.current_rollout_total))
            
            # 记录HSR环境特有的吞吐量指标
            self.logger.record("hsr/rollout_sum_throughput", np.sum(self.current_rollout_total))
            self.logger.record("hsr/rollout_mean_throughput_per_step", np.mean(self.current_rollout_total))
            
            self.current_rollout_total = []
        if self.current_rollout_pdr:
            self.logger.record("pdr/rollout_mean_pdr", np.mean(self.current_rollout_pdr))
            self.logger.record("pdr/rollout_sum_pdr", np.sum(self.current_rollout_pdr))
            self.current_rollout_pdr = []
        if self.current_rollout_SINR:
            self.logger.record("rollout/mean_sinr", np.mean(self.current_rollout_SINR))  # Log mean SINR
            self.logger.record("rollout/sum_sinr", np.sum(self.current_rollout_SINR))
            self.current_rollout_sinr = []
        if self.current_rollout_delays:
            self.logger.record("delay/rollout_mean_delay", np.mean(self.current_rollout_delays))
            self.logger.record("delay/rollout_sum_delay", np.sum(self.current_rollout_delays))
            
            # 记录HSR环境特有的延迟指标
            self.logger.record("hsr/rollout_mean_delay", np.mean(self.current_rollout_delays))
            self.logger.record("hsr/rollout_sum_delay", np.sum(self.current_rollout_delays))
            
            self.current_rollout_delays = []
        if self.current_rollout_ground_power:
            self.logger.record("power/rollout_sum_ground_power", np.sum(self.current_rollout_ground_power))
            self.current_rollout_ground_power = []
        if self.current_rollout_satellite_assisted_users:
            self.logger.record("satellite_assisted_users/rollout_mean_satellite_assisted_users",
                               np.mean(self.current_rollout_satellite_assisted_users))
            self.logger.record("satellite_assisted_users/rollout_sum_satellite_assisted_users",
                               np.sum(self.current_rollout_satellite_assisted_users))
            self.current_rollout_satellite_assisted_users = []
        # Log rollout averages
        if len(self.current_rollout_hvft) > 0:
            self.logger.record("throughput/rollout_mean_hvft", np.mean(self.current_rollout_hvft))
        if len(self.current_rollout_others) > 0:
            self.logger.record("throughput/rollout_mean_others", np.mean(self.current_rollout_others))
        if len(self.current_rollout_total) > 0:
            self.logger.record("throughput/rollout_mean_total", np.mean(self.current_rollout_total))
        if len(self.current_rollout_pdr) > 0:
            self.logger.record("pdr/rollout_mean_pdr", np.mean(self.current_rollout_pdr))
        if len(self.current_rollout_SINR) > 0:
            self.logger.record("sinr/rollout_mean_sinr", np.mean(self.current_rollout_SINR))
        if len(self.current_rollout_delays) > 0:
            self.logger.record("delays/rollout_mean_delays", np.mean(self.current_rollout_delays))
        if len(self.current_rollout_ground_power) > 0:
            self.logger.record("power/rollout_mean_ground_power", np.mean(self.current_rollout_ground_power))
        if len(self.current_rollout_satellite_assisted_users) > 0:
            self.logger.record("satellite/rollout_mean_satellite_users", np.mean(self.current_rollout_satellite_assisted_users))
        if len(self.current_rollout_power_efficiency) > 0:  # 添加功率效率记录
            self.logger.record("power_efficiency/rollout_mean_power_efficiency", np.mean(self.current_rollout_power_efficiency))
            
            # 记录HSR环境特有的功率效率指标
            self.logger.record("hsr/rollout_mean_power_efficiency", np.mean(self.current_rollout_power_efficiency))
        
        # 记录Others流量延迟统计
        if len(self.current_rollout_others_delay_sum) > 0:
            self.logger.record("others_delay/rollout_mean_others_delay_sum", np.mean(self.current_rollout_others_delay_sum))
            self.logger.record("others_delay/rollout_sum_others_delay_sum", np.sum(self.current_rollout_others_delay_sum))
        if len(self.current_rollout_others_delay_mean) > 0:
            self.logger.record("others_delay/rollout_mean_others_delay_mean", np.mean(self.current_rollout_others_delay_mean))
        if len(self.current_rollout_others_delay_min) > 0:
            self.logger.record("others_delay/rollout_mean_others_delay_min", np.mean(self.current_rollout_others_delay_min))
        if len(self.current_rollout_others_delay_max) > 0:
            self.logger.record("others_delay/rollout_mean_others_delay_max", np.mean(self.current_rollout_others_delay_max))
        if len(self.current_rollout_others_delay_std) > 0:
            self.logger.record("others_delay/rollout_mean_others_delay_std", np.mean(self.current_rollout_others_delay_std))
        if len(self.current_rollout_others_count) > 0:
            self.logger.record("others_delay/rollout_mean_others_count", np.mean(self.current_rollout_others_count))
            self.logger.record("others_delay/rollout_sum_others_count", np.sum(self.current_rollout_others_count))
        
        # 记录HVFT流量延迟统计
        if len(self.current_rollout_hvft_delay_sum) > 0:
            self.logger.record("hvft_delay/rollout_mean_hvft_delay_sum", np.mean(self.current_rollout_hvft_delay_sum))
            self.logger.record("hvft_delay/rollout_sum_hvft_delay_sum", np.sum(self.current_rollout_hvft_delay_sum))
        if len(self.current_rollout_hvft_delay_mean) > 0:
            self.logger.record("hvft_delay/rollout_mean_hvft_delay_mean", np.mean(self.current_rollout_hvft_delay_mean))
        if len(self.current_rollout_hvft_delay_min) > 0:
            self.logger.record("hvft_delay/rollout_mean_hvft_delay_min", np.mean(self.current_rollout_hvft_delay_min))
        if len(self.current_rollout_hvft_delay_max) > 0:
            self.logger.record("hvft_delay/rollout_mean_hvft_delay_max", np.mean(self.current_rollout_hvft_delay_max))
        if len(self.current_rollout_hvft_delay_std) > 0:
            self.logger.record("hvft_delay/rollout_mean_hvft_delay_std", np.mean(self.current_rollout_hvft_delay_std))
        if len(self.current_rollout_hvft_count) > 0:
            self.logger.record("hvft_delay/rollout_mean_hvft_count", np.mean(self.current_rollout_hvft_count))
            self.logger.record("hvft_delay/rollout_sum_hvft_count", np.sum(self.current_rollout_hvft_count))
        
        # Reset lists for next rollout
        self.current_rollout_hvft = []
        self.current_rollout_others = []
        self.current_rollout_total = []
        self.current_rollout_pdr = []
        self.current_rollout_SINR = []
        self.current_rollout_delays = []
        self.current_rollout_ground_power = []
        self.current_rollout_satellite_assisted_users = []
        self.current_rollout_power_efficiency = []  # 清空功率效率列表
        
        # 清空Others流量延迟统计列表
        self.current_rollout_others_delay_sum = []
        self.current_rollout_others_delay_mean = []
        self.current_rollout_others_delay_min = []
        self.current_rollout_others_delay_max = []
        self.current_rollout_others_delay_std = []
        self.current_rollout_others_count = []
        
        # 清空HVFT流量延迟统计列表
        self.current_rollout_hvft_delay_sum = []
        self.current_rollout_hvft_delay_mean = []
        self.current_rollout_hvft_delay_min = []
        self.current_rollout_hvft_delay_max = []
        self.current_rollout_hvft_delay_std = []
        self.current_rollout_hvft_count = []