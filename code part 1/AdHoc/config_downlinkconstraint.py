import numpy as np
import time
import os
import sys

from scipy import sparse
from datetime import date

from WirelessNets import *

# -------------------------------- WCS PARAMETERS -------------------------------- #

mu = 2.  # parameter for exponential / Rayleigh distribution of wireless channel distribution
n = 30  # number of base stations
k = 1  # number of users per base station

num_users = n*k

# ----------------------------- HSR WCS PARAMETERS ---------------------------- #
alpha = 0.2 # 初始化空闲链路
beta = 0.3 # 初始化传输HVFT的链路
link_all = num_users
link_idle = int(num_users * beta)
bandwidth = 1000 # 信道通频带宽 MHz
t = 1
v = 138 # 速度
R = 3000 # 覆盖范围
d = 400 # 基站与轨道垂直距离
choice = 1 # 1是郊区，2是高架桥
A_b = 30 # 基站高度
A_m = 3 # MR高度
f = 930000000 # 载波频率
max_delay = 3

# -------------------------------- SATELLITE PARAMETERS -------------------------------- #

satellite_enabled = True

leo1_altitude = 800e3
leo2_altitude = 1200e3
satellite_freq_hz = 30e9
satellite_bandwidth = 400e6
c_light = 3e8
k_boltzmann = 1.38e-23
satellite_noise_figure_db = 1.2
satellite_tx_power = 2.0
satellite_antenna_gain_tx = 10**(43.3 / 10)
satellite_g_over_t = 18.5
satellite_subchannel_num = 16

# -------------------------------- WCS PARAMETERS -------------------------------- #

lower_bound = 0.0
p0 = 0.2  # transmit power (uplink) 23dBm
upper_bound = p0*num_users  # maximum power used at a given time instant (uplink)
uplink_upper_bound = upper_bound
downlink_upper_bound = upper_bound  # maximum power used at a given time instant (downlink)
pp = 3*p0  # maximum power per plant (downlink)
p = 3  # dimension of each plant
q = 3  # dimension of control input (for each plant)
r = .001  # control effort cost
pl = 1.5  # path fading
W_obs = 1. * np.ones(num_users)  # observation / estimation noise
constraint_dim = 1  # dimension of the constraint
uplink_constraint_dim = num_users
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# control plants (AdHoc环境不需要这些LQR参数，但保留以避免导入错误)
a0 = 1.05  # diagonal element in the system matrix
Ao = np.zeros((num_users, p, p))
for ii in range(num_users):
    # 创建5x5的单位矩阵，避免维度不匹配
    Ao[ii, :, :] = np.eye(p) * a0
# estimator
estimator = 'linear'
# --------------------------------------------------------------------------------- #

# -------------------------------- SIMULATION PARAMETERS -------------------------------- #
n_feats = 3  # number of input features per node
n_feats_control = p + 1
n_layers = 10
gamma = 0.95  # discount factor
T = 30  # simulation horizon

n_actors = 1

n_episodes = 2000
n_epochs_pretrain = 50
train_batch_size = 4*T
pre_train_batch_size = 4*T
n_steps = 10
n_steps_ppo = 128
n_steps_reinforce = T
n_total_timesteps = n_episodes * n_actors * T
superv_lr = 5e-4
rl_lr = 5e-5
max_grad_norm = .5
lambda_lr = rl_lr/5
lambda_update_interval = T
lambda_0 = 1.
uplink_lambda_0 = 1.
scale_obs = True
normalize_obs = False
normalize_rewards = False
uplink_pretrain = True
downlink_pretrain = False
alg_name = 'LQR_AdHoc_PPO_WMMSEPreTrain_IdealComm_' + str(num_users) + 'u_' + str(T) + 'T_' + \
           str(a0) + 'a0_' + str(upper_bound) + 'ub_' + str(p) + 'p_' + str(q) + 'q_' + str(rl_lr) + 'lr_' + \
           str(r) + 'r_' + str(n) + 'n_' + str(k) + 'k_' + str(mu) + 'mu_' + str(n_episodes) + 'eps'
today_date = str(date.today())
save_dir = "C:\\Users\\89188\\Desktop\\code\\GRL_HSR_JPAPS2\\appendix\\Documents\\WCS\\SimResults\\" + today_date
extra_tests = False
# --------------------------------------------------------------------------------- #
