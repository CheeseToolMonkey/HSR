#!/usr/bin/env python3
"""
分析当前拓扑图是否满足T=30的计算需求
检查基站数量、多普勒效应计算和夹角问题
"""

import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LQREnvs_HSR import LQR_Env

def analyze_topology_for_T30():
    """分析拓扑图是否满足T=30的计算需求"""
    
    # 创建环境
    num_users = 30  # T=30，使用30个用户
    upperbound = 1.0
    constraint_dim = 1
    L = np.eye(num_users)
    assign = np.arange(num_users)
    
    env = LQR_Env(
        num_users=num_users,
        upperbound=upperbound,
        constraint_dim=constraint_dim,
        L=L,
        assign=assign
    )
    
    print("=== 拓扑图分析 (T=30) ===")
    print(f"用户数量: {env.num_users}")
    print(f"基站数量: {env.num_base_stations}")
    print(f"列车长度: {env.train_length}m")
    print(f"基站间距: {env.bs_spacing}m")
    print(f"基站六边形半径: {getattr(env, 'bs_hexagon_radius', 'N/A')}")
    
    # 初始化环境
    env._reset()
    
    print(f"\n=== 基站位置分析 ===")
    for i in range(env.num_base_stations):
        pos = env.base_station_positions[i]
        print(f"基站{i}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # 计算基站覆盖范围
    total_bs_coverage = (env.num_base_stations - 1) * env.bs_spacing
    print(f"\n基站覆盖范围: {total_bs_coverage}m")
    print(f"列车长度: {env.train_length}m")
    print(f"覆盖范围/列车长度比例: {total_bs_coverage/env.train_length:.2f}")
    
    # 分析设备分布
    print(f"\n=== 设备分布分析 ===")
    device_pos = env.device_positions
    print(f"设备x坐标范围: {np.min(device_pos[:, 0]):.1f}m 到 {np.max(device_pos[:, 0]):.1f}m")
    print(f"设备y坐标: {device_pos[0, 1]:.1f}m (所有设备)")
    
    # 计算基站分配
    env.update_device_bs_assignment()
    unique_assignments, counts = np.unique(env.device_bs_assignment, return_counts=True)
    print(f"\n基站分配统计:")
    for bs_idx, count in zip(unique_assignments, counts):
        print(f"  基站{bs_idx}: {count}个设备")
    
    # 分析多普勒效应
    print(f"\n=== 多普勒效应分析 ===")
    analyze_doppler_effect(env)
    
    # 分析T=30的可行性
    print(f"\n=== T=30可行性分析 ===")
    analyze_T30_feasibility(env)
    
    return True

def analyze_doppler_effect(env):
    """分析多普勒效应计算"""
    
    # 获取参数
    v = getattr(env, 'v', 100)  # 列车速度 m/s
    f = getattr(env, 'f', 930e6)  # 载波频率 Hz
    c = 3e8  # 光速 m/s
    d = 200  # 设备到基站距离 m
    
    print(f"列车速度: {v}m/s ({v*3.6:.1f}km/h)")
    print(f"载波频率: {f/1e6:.0f}MHz")
    print(f"设备到基站距离: {d}m")
    
    # 当前多普勒效应计算
    current_doppler = f * v / c
    print(f"当前多普勒频移: {current_doppler:.2f}Hz")
    print(f"多普勒频移/载波频率: {current_doppler/f*1e6:.2f}ppm")
    
    # 检查是否考虑夹角
    print(f"\n多普勒效应夹角分析:")
    print(f"当前计算: fd = f * v / c (未考虑夹角)")
    print(f"正确计算: fd = f * v * cos(θ) / c (考虑夹角)")
    
    # 计算不同夹角下的多普勒效应
    angles = [0, 30, 45, 60, 90]  # 度
    print(f"\n不同夹角下的多普勒频移:")
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        cos_theta = np.cos(angle_rad)
        doppler_with_angle = f * v * cos_theta / c
        print(f"  {angle_deg}°: {doppler_with_angle:.2f}Hz (cos(θ)={cos_theta:.3f})")
    
    # 分析夹角对ICI的影响
    print(f"\n夹角对ICI的影响:")
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        cos_theta = np.cos(angle_rad)
        doppler_with_angle = f * v * cos_theta / c
        
        # 计算ICI系数
        x_values = np.linspace(-1, 1, 1000)
        from scipy.special import j0
        integral_result = np.trapz((1 - abs(x_values)) * j0(2 * np.pi * doppler_with_angle * 0.15e-3 * x_values), x_values)
        W_ICI = 1 - integral_result
        
        print(f"  {angle_deg}°: W_ICI={W_ICI:.4f}")

def analyze_T30_feasibility(env):
    """分析T=30的可行性"""
    
    print(f"当前配置分析:")
    print(f"1. 基站数量: {env.num_base_stations}个")
    print(f"2. 基站间距: {env.bs_spacing}m")
    print(f"3. 列车长度: {env.train_length}m")
    print(f"4. 用户数量: {env.num_users}个")
    
    # 检查基站覆盖是否足够
    total_coverage = (env.num_base_stations - 1) * env.bs_spacing
    if total_coverage >= env.train_length:
        print(f"✅ 基站覆盖范围({total_coverage}m) >= 列车长度({env.train_length}m)")
    else:
        print(f"❌ 基站覆盖范围({total_coverage}m) < 列车长度({env.train_length}m)")
        print(f"   建议增加基站数量或减少基站间距")
    
    # 检查用户分布
    device_pos = env.device_positions
    x_range = np.max(device_pos[:, 0]) - np.min(device_pos[:, 0])
    if x_range <= env.train_length:
        print(f"✅ 用户分布范围({x_range:.1f}m) <= 列车长度({env.train_length}m)")
    else:
        print(f"❌ 用户分布范围({x_range:.1f}m) > 列车长度({env.train_length}m)")
    
    # 建议改进
    print(f"\n建议改进:")
    if env.num_base_stations < 5:
        print(f"1. 增加基站数量到5-7个，确保更好的覆盖")
    if env.bs_spacing > 800:
        print(f"2. 减少基站间距到800m以下，提高覆盖密度")
    if env.train_length < 3000:
        print(f"3. 增加列车长度到3000m以上，支持更多用户")
    
    # 多普勒效应改进建议
    print(f"\n多普勒效应改进建议:")
    print(f"1. 考虑夹角影响: fd = f * v * cos(θ) / c")
    print(f"2. 根据设备到基站的实际角度计算多普勒频移")
    print(f"3. 不同设备可能有不同的多普勒效应")

if __name__ == "__main__":
    analyze_topology_for_T30()

