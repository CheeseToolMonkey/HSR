import sys
import control
import scipy.linalg

from datetime import date

from WirelessNets import *

# -------------------------------- WCS PARAMETERS -------------------------------- #
mu = 2.  # parameter for exponential / Rayleigh distribution of wireless channel distribution
n = 10  # number of base stations
k = 3  # number of users per base station
num_users = n*k
lower_bound = 0.0
p0 = 5.  #
upper_bound = p0*num_users  # maximum power used at a given time instant (uplink)
downlink_upper_bound = upper_bound  # maximum power used at a given time instant (downlink)
uplink_upper_bound = upper_bound
pp = upper_bound/n  # maximum power per plant
p = 3  # dimension of each plant
q = 3  # dimension of control input (for each plant)
r = .001  # control effort cost (to compute feedback gain)
pl = 1.5  # path fading
W_obs = 1. * np.ones(num_users)  # observation / estimation noise
constraint_dim = n  # dimension of the constraint (downlink)
uplink_constraint_dim = 1
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
a0 = 1.05  # diagonal element in the system matrix
Ao = np.zeros((num_users, p, p))
for ii in range(num_users):
    Ao[ii, :, :] = np.array([[a0, .2, .2], [0., a0, .2], [0, 0., a0]])
# feedback gain (for LQR expert)
if p == q:
    Bo = np.eye(q)
else:
    Bo = np.ones((p, 1))
fb_gain = np.zeros((num_users, q, p))
for ii in range(num_users):
    (_, _, fb_gain[ii, :, :]) = control.dare(Ao[ii, :, :], Bo, np.eye(p), r * np.eye(q))  # ARE
fb_gain_blk = fb_gain[0, :, :]
for ii in range(1, num_users):
    fb_gain_blk = scipy.linalg.block_diag(fb_gain_blk, fb_gain[ii, :, :])
fb_gain = fb_gain_blk
max_control_state = 50.
max_control_effort = 2*max_control_state
# --------------------------------------------------------------------------------- #

# -------------------------------- SIMULATION PARAMETERS -------------------------------- #
n_feats = 1  # number of input features per node
n_feats_control = p + 1
n_layers = 10
gamma = 0.95  # discount factor
T = 30  # simulation horizon
n_actors = 16
n_episodes = 10000  # ideally ~ 5000 eps // lr ~-3
n_epochs_pretrain = 50
train_batch_size = 4*T
pre_train_batch_size = 4*T
n_steps = 10
n_steps_ppo = 128
n_steps_reinforce = T
n_total_timesteps = n_episodes * n_actors * T
superv_lr = 5e-4
rl_lr = 5e-5
lambda_lr = rl_lr/5
lambda_update_interval = T
lambda_0 = 1.
uplink_lambda_0 = 1.
scale_obs = True
normalize_obs = False
normalize_rewards = False
uplink_pretrain = False
downlink_pretrain = False
alg_name = 'LQR_MultiCell_PPO_SchedulingUpDownlink_IdealComm_MBEst_Normalized' + str(num_users) + 'u_' + str(T) + 'T_' + \
           str(a0) + 'a0_' + str(upper_bound) + 'ub_' + str(p) + 'p_' + str(q) + 'q_' + str(rl_lr) + 'lr_' + \
           str(r) + 'r_' + str(n) + 'n_' + str(k) + 'k_' + str(mu) + 'mu_' + str(gamma) + 'gamma_' + str(n_episodes) + 'eps'
today_date = str(date.today())
save_dir = "../../../../../../../Documents/WCS/SimResults/" + today_date
extra_tests = False
# Windows
# save_dir = "C:/Users/vlsvi/Dropbox/PhD/Research/WCS/SimResults/" + today_date
# --------------------------------------------------------------------------------- #
