###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard Libraries
import time
import sys
import numpy as np
import os
import scipy

# Local files
sys.path.append('C:\\Users\\AWAITXM\\PycharmProjects\\GRL_WCSs_Clean\\')
sys.path.append('C:\\Users\\AWAITXM\\PycharmProjects\\GRL_WCSs_Clean\\GNNs')
# sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/')
# sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/GNNs')

import AdHoc.config_downlinkconstraint as config
import LQREnvs_AdHoc as LQREnvs
import WirelessNets
import GRLWCSUtils as utils

# Agents
from stable_baselines3 import PPO
import GNNPPO
# GNN Policy
import GNNReinforce
import GNNPolicies
import GNNPPO
import SubProcEnvMod

import torch


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


def save_rt_data(data, filename):
    data_dict = {}
    # plotting variables over time
    for data_name in ['gnn_mean', 'equal_mean', 'ca_mean', 'rr_mean', 'wmmse_mean', 'ra_mean',
                      'gnn_std', 'equal_std', 'ca_std', 'rr_std', 'wmmse_std', 'ra_std',]:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)


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
    test_env_type = LQREnvs.LQRAdHocDownlinkOutputConstraint

    # Loading GNN, DNN agents
    gnn_downlink_aux = GNNPPO.GNNPPO.load('AdHocDownlinkConstraintGNNAgentPPO_oldL_n30_k1')

    # Tests
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir, 0o755)
    alg_name = 'LQR_MultiCell_PPO_AdHocConstraintTrainingOutputConstraintTest_GNNPPO_TransferenceTests_30u'

    n_tests = 11
    n_realizations_per_test = 20
    scale_tests = np.arange(2, n_tests+1)
    og_n_users = 100
    p0 = 5.
    og_distance = 6.
    T = 80
    assign = 0
    n_actors = 2
    options_name = test_type + alg_name + 'T' + str(T) + '.mat'
    gamma = config.gamma
    save_dir = config.save_dir
    print("Transference test")
    cur_test = 1
    pl=1.5

    # runtime dictionary
    runtime_dict[policy_name] = {'gnn_mean': [], 'equal_mean': [], 'ca_mean': [], 'rr_mean': [], 'wmmse_mean': [], 'ra_mean': [],
                    'gnn_std': [], 'equal_std': [], 'ca_std': [], 'rr_std': [], 'wmmse_std': [], 'ra_std': []}

    for scale_factor in scale_tests:
        print('Test ' + str(cur_test) + ' of ' + str(n_tests) + '. \n')
        num_users = og_n_users*scale_factor
        upper_bound = num_users*p0
        L = WirelessNets.build_adhoc_network_samedistance(num_users, pl, og_distance)
        test_env = test_env_type(num_users,
                                 upper_bound,
                                 config.constraint_dim, L,
                                 assign, config.n,
                                 config.k, mu=config.mu, p=config.p,
                                 q=config.q, T=config.T,
                                 pl=config.pl, a0=config.a0,
                                 r=config.r,
                                 pp=config.pp, p0=config.p0,
                                 num_features=config.n_feats,
                                 scaling=config.scale_obs)
        vec_env = SubProcEnvMod.SubprocVecEnv([utils.make_env(test_env, i) for i in range(n_actors)], num_users=num_users)
        gnn_downlink = GNNPPO.GNNPPO.load('AdHocDownlinkConstraintGNNAgentPPO_n100_k1', env=vec_env)
        # dumb way, need to change this
        gnn_downlink.policy.n_agents = num_users
        gnn_downlink.policy.graph_dim = num_users**2
        device = gnn_downlink.policy.log_std.device
        gnn_downlink.policy.log_std = torch.zeros(num_users).to(device)
        gnn_downlink.policy.action_dim = num_users * 2  # 2 features per user: power and satellite selection
        gnn_costs = np.zeros(n_realizations_per_test)
        eq_costs = np.zeros(n_realizations_per_test)
        rr_costs = np.zeros(n_realizations_per_test)
        ra_costs = np.zeros(n_realizations_per_test)
        ca_costs = np.zeros(n_realizations_per_test)
        wmmse_costs = np.zeros(n_realizations_per_test)
        for jj in range(n_realizations_per_test):
            # reseeding environment
            test_env.seed()
            (gnn_cost_mtx, eqpwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, capwr_cost_mtx, rapwr_cost_mtx) = \
                test_env.test_transf(upper_bound, T, gnn_downlink, test_type='output_constraint')
            gnn_disc = utils.discounted_returns_test(gnn_cost_mtx[None, :], 1, T, gamma)
            eq_disc = utils.discounted_returns_test(eqpwr_cost_mtx[None, :], 1, T, gamma)
            rr_disc = utils.discounted_returns_test(rrpwr_cost_mtx[None, :], 1, T, gamma)
            ca_disc = utils.discounted_returns_test(capwr_cost_mtx[None, :], 1, T, gamma)
            wmmse_disc = utils.discounted_returns_test(wmmsepwr_cost_mtx[None, :], 1, T, gamma)
            ra_disc = utils.discounted_returns_test(rapwr_cost_mtx[None, :], 1, T, gamma)

            cost_gnn = gnn_disc[:, 0] / num_users
            cost_ca = ca_disc[:, 0] / num_users
            cost_wmmse = wmmse_disc[:, 0] / num_users
            cost_rr = rr_disc[:, 0] / num_users
            cost_eq = eq_disc[:, 0] / num_users
            cost_ra = ra_disc[:, 0] / num_users

            gnn_costs[jj] = cost_gnn
            eq_costs[jj] = cost_eq
            ca_costs[jj] = cost_ca
            rr_costs[jj] = cost_rr
            ra_costs[jj] = cost_ra
            wmmse_costs[jj] = cost_wmmse

        cur_test += 1
        gnn_mean = gnn_costs.mean()
        gnn_std = gnn_costs.std()
        runtime_dict[policy_name]['gnn_mean'].append(gnn_mean)
        runtime_dict[policy_name]['gnn_std'].append(gnn_std)

        eq_mean = eq_costs.mean()
        eq_std = eq_costs.std()
        runtime_dict[policy_name]['equal_mean'].append(eq_mean)
        runtime_dict[policy_name]['equal_std'].append(eq_std)

        ca_mean = ca_costs.mean()
        ca_std = ca_costs.std()
        runtime_dict[policy_name]['ca_mean'].append(ca_mean)
        runtime_dict[policy_name]['ca_std'].append(ca_std)

        rr_mean = rr_costs.mean()
        rr_std = rr_costs.std()
        runtime_dict[policy_name]['rr_mean'].append(rr_mean)
        runtime_dict[policy_name]['rr_std'].append(rr_std)

        ra_mean = ra_costs.mean()
        ra_std = ra_costs.std()
        runtime_dict[policy_name]['ra_mean'].append(ra_mean)
        runtime_dict[policy_name]['ra_std'].append(ra_std)

        wmmse_mean = wmmse_costs.mean()
        wmmse_std = wmmse_costs.std()
        runtime_dict[policy_name]['wmmse_mean'].append(wmmse_mean)
        runtime_dict[policy_name]['wmmse_std'].append(wmmse_std)

    print("Saving test data")
    options_name = alg_name + 'T' + str(T) + '.mat'
    file_path = save_dir + '/wireless_control_test' + options_name
    save_rt_data(runtime_dict, file_path)  # save "runtime" data (after training)

