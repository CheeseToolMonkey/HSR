###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard Libraries
import time
import sys
import numpy as np
import os

# Local files
sys.path.append('/home/vinicius/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/')
sys.path.append('/home/vinicius/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/GNNs')
sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/')
sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/GNNs')

import MultiCellScheduling.config_scheduling as config
import LQREnvs
import WirelessNets
import GRLWCSUtils as utils

# Agents
from stable_baselines3 import PPO
import GNNPPO


def load_training_mean_var(file_mean, file_var, agent):
    try:
        mean = np.load(file_mean)
    except IOError:
        mean = np.zeros(3)
        print(agent + ' training mean not found. Using standard Gaussian Distribution for normalization. \n')
    try:
        var = np.load(file_var)
    except IOError:
        var = np.ones(3)
        print(agent + ' training variance not found. Using standard Gaussian Distribution for normalization. \n')

    return mean, var


###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__':
    start_time_overall = time.time()
    test_type = 'zero'
    policy_name = 'stable-baselines-ppo'
    runtime_dict = {}
    _, _, runtime_dict[policy_name] = utils.initialize_dictionaries()

    # Environment types
    test_env_type = LQREnvs.LQRMultiCellDownlinkScheduling

    # Loading GNN, DNN agents
    dnn_downlink = PPO.load("DownlinkSchedulingDNNAgentPPONoPreTrain_T30_kvarying_n10")
    gnn_downlink = GNNPPO.GNNPPO.load("DownlinkSchedulingGNNAgentPPONoPreTrain_T30_kvarying_n10")

    # Loading GNN, DNN mean and standard deviation for input normalization
    dnn_mean, dnn_var = load_training_mean_var('DownlinkSchedulingDNNAgentPPONoPreTrain_T30_kvarying_n10_mean.npy', 'DownlinkSchedulingDNNAgentPPONoPreTrain_var.npy', 'MLP')
    gnn_mean, gnn_var = load_training_mean_var('DownlinkSchedulingGNNAgentPPONoPreTrain_T30_kvarying_n10_mean.npy', 'DownlinkSchedulingGNNAgentPPONoPreTrain_var.npy', 'GNN')

    # Network data
    net_name = 'L_varying' + '_n' + str(config.n) + '_k' + str(config.k) + '.npy'
    assign_name = 'assign' + '_n' + str(config.n) + '_k' + str(config.k) + '.npy'
    try:
        L = np.load(net_name)
        assign = np.load(assign_name)
    except IOError:
        print('Warning: Network data not found. Redrawing multicellular network')
        k_min = 3
        k_max = 6
        L, assign = WirelessNets.build_varying_cellular_network(config.n, k_min, k_max, config.pl)
        np.save(net_name, L)
        np.save(assign_name, assign)

    # Tests
    num_users = L.shape[0]
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir, 0o755)
    alg_name = 'LQR_MultiCell_PPO_Scheduling_GNNPPO_nt10' + \
               str(num_users) + 'u_' + str(config.T) + 'T_' + str(config.a0) + 'a0_' + str(config.upper_bound) \
               + 'ub_' + str(config.p) + 'p_' + str(config.q) + 'q_' + str(config.rl_lr) + 'lr_' + str(config.r) + 'r_' + \
               str(config.n) + 'n_' + str(config.k) + 'k_' + str(config.mu) + 'mu_' + str(config.n_episodes) + 'eps'

    n_tests = 20
    n_total_tests = config.n_actors * n_tests
    T = 80
    options_name = test_type + alg_name + 'T' + str(T) + '.mat'

    test_env = test_env_type(num_users,
                             config.downlink_upper_bound,
                             config.constraint_dim,
                             L, assign,
                             config.n, config.k,
                             mu=config.mu, p=config.p, q=config.q, T=config.T, pl=config.pl, a0=config.a0, Ao=None,
                             r=config.r, pp=config.pp, p0=config.p0, num_features=config.n_feats,
                             scaling=config.scale_obs)

    utils.runtime_test(test_env, dnn_downlink, gnn_downlink, runtime_dict, n_total_tests,
                       T, config.upper_bound, config.n_actors, alg_name, config.save_dir, dnn_mean=dnn_mean,
                       dnn_var=dnn_var, gnn_mean=gnn_mean, gnn_var=gnn_var, pol_name=policy_name,
                       new_options_name=options_name, normalize_obs=config.normalize_obs, test_type='scheduling')

    if config.extra_tests:
        mu_list = [0.5, 1., 2.]
        aux_list = [.25, .5, 2.]
        ub_list = [config.num_users * aux for aux in aux_list]
        for new_mu in mu_list:
            for new_ub in ub_list:
                runtime_dict = {}
                _, _, runtime_dict[policy_name] = utils.initialize_dictionaries()

                options_name = test_type + alg_name + 'T' + str(T) + '_newmu' + str(new_mu) + '_newub' + str(
                    new_ub) + '.mat'

                test_env = test_env_type(config.num_users, new_ub, config.constraint_dim, L, assign, config.n, config.k,
                                         mu=new_mu, p=config.p, q=config.q, T=config.T, pl=config.pl, a0=config.a0,
                                         Ao=config.Ao, r=config.r, p0=config.p0, num_features=config.n_feats,
                                         pp=config.pp, scaling=config.scale_obs)
                utils.runtime_test(test_env, dnn_downlink, gnn_downlink, runtime_dict, n_total_tests, T,
                                   config.upper_bound, config.n_actors, alg_name, config.save_dir,
                                   dnn_mean=dnn_mean, dnn_var=dnn_var, gnn_mean=gnn_mean, gnn_var=gnn_var,
                                   pol_name=policy_name, new_options_name=options_name,
                                   normalize_obs=config.normalize_obs, test_type='scheduling')


