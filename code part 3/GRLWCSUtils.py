###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard Libraries
import numpy as np
import scipy.io
import os
import sys
import time

# Pytorch
import torch
torch.set_default_dtype(torch.float32)

import SubProcEnvMod


###############################################################################
############################# U T I L I T I E S ###############################
###############################################################################

def moving_average(data, window=10):
    cumsum = np.cumsum(data)
    moving_sum = cumsum[window:] - cumsum[:-window]
    moving_avg = moving_sum / window

    moving_avg = np.concatenate((np.zeros(window), moving_avg))
    moving_avg[:window] = cumsum[:window] / window
    return moving_avg


# (i) training/learning phase: dnn
def save_dnn_data(data, filename):
    data_dict = {}
    for data_name in ['pg_loss', 'vf_loss',
                      'dnn_cost_downlink', 'lambda_dnn_downlink', 'dnn_cost_uplink', 'lambda_dnn_uplink',
                      'dnn_constraint_downlink', 'dnn_constraint_uplink', 'dnn_Lagrangian_downlink', 'dnn_Lagrangian_uplink',
                      'dnn_dual_cost', 'lambda_dnn_dual_downlink', 'dnn_dual_Lagrangian', 'dnn_dual_constraint_downlink',
                      'lambda_dnn_dual_uplink', 'dnn_dual_constraint_uplink',
                      'x_dnn', 'p_dnn']:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)

# (ii) training/learning phase: gnn
def save_gnn_data(data, filename):
    data_dict = {}
    for data_name in ['pg_loss', 'vf_loss',
                      'gnn_cost_downlink', 'lambda_gnn_downlink', 'gnn_cost_uplink', 'lambda_gnn_uplink',
                      'gnn_constraint_downlink', 'gnn_constraint_uplink', 'gnn_Lagrangian_downlink', 'gnn_Lagrangian_uplink',
                      'gnn_dual_cost', 'lambda_gnn_dual_downlink', 'gnn_dual_Lagrangian', 'gnn_dual_constraint_downlink',
                      'lambda_gnn_dual_uplink', 'gnn_dual_constraint_uplink',
                      'x_gnn', 'p_gnn']:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)


# (iii) test/runtime phase
def save_rt_data(data, filename):
    data_dict = {}
    # plotting variables over time
    for data_name in ['dnn', 'equal_power', 'ca_power', 'rr_power', 'wmmse_power', 'ra_power', 'gnn',
                      'dnn_avg', 'equal_power_avg', 'ca_power_avg', 'rr_power_avg', 'wmmse_power_avg', 'gnn_avg', 'ra_power_avg',
                      'x_dnn', 'p_dnn',
                      'x_gnn', 'p_gnn',
                      'x_eq', 'p_eq',
                      'x_ca', 'p_ca',
                      'x_wmmse', 'p_wmmse',
                      'x_rr', 'p_rr',
                      'x_ra', 'p_ra']:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)


def initialize_dictionaries():
    # train dict
    policy_dict_dnn = {'pg_loss': [], 'vf_loss': [],
                       'dnn_cost_downlink': [], 'lambda_dnn_downlink': [], 'dnn_cost_uplink': [], 'lambda_dnn_uplink': [],
                       'dnn_constraint_downlink': [], 'dnn_constraint_uplink': [], 'dnn_Lagrangian_downlink': [], 'dnn_Lagrangian_uplink': [],
                       'dnn_dual_cost': [], 'lambda_dnn_dual_downlink': [], 'dnn_dual_Lagrangian': [],
                       'dnn_dual_constraint_downlink': [], 'lambda_dnn_dual_uplink': [], 'dnn_dual_constraint_uplink': [],
                       'x_dnn': [], 'p_dnn': []}

    policy_dict_gnn = {'pg_loss': [], 'vf_loss': [],
                       'gnn_cost_downlink': [], 'lambda_gnn_downlink': [], 'gnn_cost_uplink': [], 'lambda_gnn_uplink': [],
                       'gnn_constraint_downlink': [], 'gnn_constraint_uplink': [], 'gnn_Lagrangian_downlink': [],
                       'gnn_Lagrangian_uplink': [],
                       'gnn_dual_cost': [], 'lambda_gnn_dual_downlink': [], 'gnn_dual_Lagrangian': [],
                       'gnn_dual_constraint_downlink': [], 'lambda_gnn_dual_uplink': [], 'gnn_dual_constraint_uplink': [],
                       'x_gnn': [], 'p_gnn': []}
    # runtime dict
    runtime_dict = {'dnn': [], 'gnn': [], 'equal_power': [], 'ca_power': [], 'rr_power': [], 'wmmse_power': [], 'ra_power': [],
                    'dnn_avg': [], 'gnn_avg': [], 'equal_power_avg': [], 'ca_power_avg': [], 'rr_power_avg': [], 'wmmse_power_avg': [],  'ra_power_avg': [],
                    'x_dnn': [], 'p_dnn': [],
                    'x_gnn': [], 'p_gnn': [],
                    'x_eq': [], 'p_eq': [],
                    'x_ca': [], 'p_ca': [],
                    'x_wmmse': [], 'p_wmmse': [],
                    'x_ra': [], 'p_ra': [],
                    'x_rr': [], 'p_rr': []}

    return policy_dict_dnn, policy_dict_gnn, runtime_dict


# matrix of discounted returns for the test phase
def discounted_returns_test(cost_matrix, batch_size, T, gamma):
    # learned policy --- deep nn (codesign)
    disc_cost = np.zeros((batch_size, T))
    disc_cost[:, -1] = cost_matrix[:, -1]

    # calculating discounted return backwards (step by step)
    for k in range(1, T):
        disc_cost[:, T - 1 - k] = cost_matrix[:, T - 1 - k] + gamma * disc_cost[:, T - 1 - k + 1]

    return disc_cost


# from OpenAI stable baselines
def make_env(env, rank, seed=0, link_logger=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = gym.make(env_id)
        env.seed(seed + rank)  # modification to use custom environment
        env.env_id = rank  # 设置环境ID，用于区分不同子环境
        return env
    return _init


# saving GNN policy parameters (.mat)
def save_gnn_params(gnn_pol_net, num_layers, file_name):

    gnn_policy_params = gnn_pol_net.params[0:2*num_layers+1]
    variable_names = []
    data_save = {}
    index = 0

    for i in np.arange(num_layers - 1):
        variable_names.append("weight" + str(i))
        variable_names.append("bias" + str(i))
    variable_names.append("weight" + str(num_layers - 1))
    variable_names.append("weight" + str(num_layers))
    variable_names.append("bias" + str(num_layers))

    tvars_vals = gnn_pol_net.sess.run(gnn_policy_params)
    for var, val in zip(gnn_policy_params, tvars_vals):
        data_save[variable_names[index]] = val
        index += 1
    scipy.io.savemat(file_name, data_save)

    return data_save


def create_envs(downlink_env_type, uplink_env_type, num_users, downlink_upper_bound, constraint_dim, L, assign, n, k,
                mu, p, q, T, pl, a0, Ao, r, pp, n_feats, estimator, scale_obs, gamma, uplink_upper_bound,
                uplink_constraint_dim, p0):
    env = downlink_env_type(num_users, downlink_upper_bound, constraint_dim, L, assign, n, k, mu=mu, p=p, q=q, T=T,
                            pl=pl, a0=a0, Ao=Ao, r=r, pp=pp, p0=p0, num_features=n_feats, estimator_type=estimator,
                            scaling=scale_obs, gamma=gamma)
    env_aux = downlink_env_type(num_users, downlink_upper_bound, constraint_dim, L, assign, n, k, mu=mu, p=p, q=q, T=T,
                                pl=pl, a0=a0, Ao=Ao, r=r, pp=pp, p0=p0,  num_features=n_feats, estimator_type=estimator,
                                scaling=scale_obs, gamma=gamma)

    uplink_env = uplink_env_type(num_users, uplink_upper_bound, uplink_constraint_dim, L, assign, n, k, mu=mu, p=p,
                                 q=q, T=T, a0=a0, Ao=Ao, r=r, pp=p0, p0=p0, num_features=n_feats, estimator_type=estimator,
                                 scaling=scale_obs, gamma=gamma)

    uplink_env_aux = uplink_env_type(num_users, uplink_upper_bound, uplink_constraint_dim, L, assign, n, k, mu=mu,
                                     p=p, q=q, T=T, a0=a0, Ao=Ao, r=r, pp=p0, p0=p0, num_features=n_feats,
                                     estimator_type=estimator, scaling=scale_obs, gamma=gamma)

    return env, env_aux, uplink_env, uplink_env_aux


def create_vecenv(single_env, T, n_actors, gamma, n_feats, num_users, constraint=True, normalization=False, const_dim=1,
                  lambda_init=0., lambda_lr=1e-4, normalize_rews=True, link_logger=None):
    if constraint:
        if normalization or normalize_rews:
            vec_env = \
                SubProcEnvMod.SubprocVecEnvConstraintNormalized(lambda_lr, T,
                                                                [make_env(single_env, i, link_logger=link_logger)  for i in range(n_actors)],
                                                                gamma, lambda_0=lambda_init,
                                                                constraint_dim=const_dim,
                                                                n_feats=n_feats,
                                                                n_users=num_users,
                                                                norm_obs=normalization,
                                                                norm_rews=normalize_rews)
        else:
            vec_env = SubProcEnvMod.SubprocVecEnvConstraint(lambda_lr, T,
                                                            [make_env(single_env, i, link_logger=link_logger) for i in range(n_actors)],
                                                            gamma, lambda_0=lambda_init,
                                                            constraint_dim=const_dim)
    else:
        if normalization or normalize_rews:
            vec_env = SubProcEnvMod.SubprocVecEnvNormalized([make_env(single_env, i, link_logger=link_logger)  for i in range(n_actors)], gamma,
                                                            n_feats=n_feats, n_users=num_users, norm_obs=normalization,
                                                            norm_rews=normalize_rews)
        else:
            vec_env = SubProcEnvMod.SubprocVecEnv([make_env(single_env, i, link_logger=link_logger)  for i in range(n_actors)])

    return vec_env


###############################################################################
############################## T R A I N I N G ################################
###############################################################################


def train_policy(agent, n_steps, vecenv, train_dict=[], pol_name='stable_baselines_ppo', param='dnn',
                 constraint=False, normalization=False, uplink_mean_init=0., uplink_var_init=1., return_mean_init=0.,
                 return_var_init=1., downlink_return_mean_init=0., downlink_return_var_init=1., agent_type='downlink',
                 callback=None, link_logger=None):
    # agent.learn(total_timesteps=n_steps)
        # 创建链路数据记录callback
    if link_logger is not None:
        from stable_baselines3.common.callbacks import BaseCallback
        
        class LinkDataCallback(BaseCallback):
            def __init__(self, link_logger, vecenv, verbose=0):
                super(LinkDataCallback, self).__init__(verbose)
                self.link_logger = link_logger
                self.vecenv = vecenv
                self.timestep = 0
            
            def _on_step(self) -> bool:
                # 数据记录已经在环境的step方法中处理，这里不需要额外操作
                self.timestep += 1
                return True
        
        # 创建链路数据记录callback
        link_callback = LinkDataCallback(link_logger, vecenv)
        
        # 合并callback
        if callback is not None:
            from stable_baselines3.common.callbacks import CallbackList
            final_callback = CallbackList([callback, link_callback])
        else:
            final_callback = link_callback
    else:
        final_callback = callback
        
    agent.learn(total_timesteps=n_steps, callback=final_callback)  # <-- 将 callback 传递给 learn 方法
    # saving training costs, constraint violation and dual variable
    ep_costs = vecenv.get_ep_costs()
    ep_lambda = []
    if constraint:
        ep_lambda = vecenv.lambda_hist
    ep_constraint = vecenv.get_ep_constraints()
    ep_Lagrangian = vecenv.get_ep_Lagrangian()
    str_aux = param + '_cost' + '_' + agent_type
    train_dict[pol_name][str_aux].append(ep_costs)
    if constraint:
        str_aux = param + '_Lagrangian' + '_' + agent_type
        train_dict[pol_name][str_aux].append(ep_Lagrangian)
        str_aux = param + '_constraint' + '_' + agent_type
        train_dict[pol_name][str_aux].append(ep_constraint)
        str_aux = 'lambda_' + param + '_' + agent_type
        train_dict[pol_name][str_aux].append(ep_lambda)

    if normalization:
        uplink_mean_init = [vecenv.channel_rms.mean, vecenv.plants_rms.mean, vecenv.interval_rms.mean]
        uplink_var_init = [vecenv.channel_rms.var, vecenv.plants_rms.var, vecenv.interval_rms.var]
        return_mean_init = vecenv.ret_rms.mean
        return_var_init = vecenv.ret_rms.var
        downlink_return_var_init = return_var_init
        downlink_return_mean_init = return_mean_init

    return agent, train_dict, uplink_mean_init, uplink_var_init, return_mean_init, return_var_init, \
           downlink_return_mean_init, downlink_return_var_init


###############################################################################
############################### R U N T I M E #################################
###############################################################################

def runtime_test(test_env, dnn_downlink, gnn_downlink, runtime_dict, n_total_tests, T,
                 upper_bound, n_actors, alg_name, save_dir,
                 dnn_mean=0., dnn_var=1., gnn_mean=0., gnn_var=1., pol_name='stable_baselines_ppo',
                 new_options_name=None, gamma=.95, normalize_obs=False, test_type='base_scheduling'):

    print("Test: comparing performance of GNN, DNN and some heuristics")

    for jj in range(n_total_tests):
        print('Test ' + str(jj) + ' of ' + str(n_total_tests) + '. \n')
        # reseeding environment
        test_env.seed()
        (dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, capwr_cost_mtx, rapwr_cost_mtx,
         dnn_power, gnn_power, equal_power, wmmse_power, rr_power, ca_power, ra_power,
         dnn_states, gnn_states, eq_states, wmmse_states, rr_states, ca_states, ra_states) = \
            test_env.test(upper_bound, T, dnn_downlink, gnn_downlink, test_type=test_type)

        # computes discounted cost
        dnn_disc = discounted_returns_test(dnn_cost_mtx[None, :], 1, T, gamma)
        gnn_disc = discounted_returns_test(gnn_cost_mtx[None, :], 1, T, gamma)
        eq_disc = discounted_returns_test(eqpwr_cost_mtx[None, :], 1, T, gamma)
        rr_disc = discounted_returns_test(rrpwr_cost_mtx[None, :], 1, T, gamma)
        ca_disc = discounted_returns_test(capwr_cost_mtx[None, :], 1, T, gamma)
        wmmse_disc = discounted_returns_test(wmmsepwr_cost_mtx[None, :], 1, T, gamma)
        ra_disc = discounted_returns_test(rapwr_cost_mtx[None, :], 1, T, gamma)

        # computes average cost
        dnn_avg = dnn_cost_mtx.sum() / (T*test_env.num_users)
        gnn_avg = gnn_cost_mtx.sum() / (T * test_env.num_users)
        eq_avg = eqpwr_cost_mtx.sum() / (T * test_env.num_users)
        rr_avg = rrpwr_cost_mtx.sum() / (T * test_env.num_users)
        ca_avg = capwr_cost_mtx.sum() / (T * test_env.num_users)
        wmmse_avg = wmmsepwr_cost_mtx.sum() / (T * test_env.num_users)
        ra_avg = rapwr_cost_mtx.sum() / (T * test_env.num_users)

        cost_dnn = dnn_disc[:, 0]
        cost_gnn = gnn_disc[:, 0]
        cost_ca = ca_disc[:, 0]
        cost_wmmse = wmmse_disc[:, 0]
        cost_rr = rr_disc[:, 0]
        cost_eq = eq_disc[:, 0]
        cost_ra = ra_disc[:, 0]

        runtime_dict[pol_name]['dnn'].append(cost_dnn)
        runtime_dict[pol_name]['gnn'].append(cost_gnn)
        runtime_dict[pol_name]['equal_power'].append(cost_eq)
        runtime_dict[pol_name]['ca_power'].append(cost_ca)
        runtime_dict[pol_name]['wmmse_power'].append(cost_wmmse)
        runtime_dict[pol_name]['rr_power'].append(cost_rr)
        runtime_dict[pol_name]['ra_power'].append(cost_ra)

        runtime_dict[pol_name]['dnn_avg'].append(dnn_avg)
        runtime_dict[pol_name]['gnn_avg'].append(gnn_avg)
        runtime_dict[pol_name]['equal_power_avg'].append(eq_avg)
        runtime_dict[pol_name]['ca_power_avg'].append(ca_avg)
        runtime_dict[pol_name]['wmmse_power_avg'].append(wmmse_avg)
        runtime_dict[pol_name]['rr_power_avg'].append(rr_avg)
        runtime_dict[pol_name]['ra_power_avg'].append(ra_avg)

        runtime_dict[pol_name]['x_dnn'].append(dnn_states)
        runtime_dict[pol_name]['p_dnn'].append(dnn_power)

        runtime_dict[pol_name]['x_gnn'].append(gnn_states)
        runtime_dict[pol_name]['p_gnn'].append(gnn_power)

        runtime_dict[pol_name]['x_eq'].append(eq_states)
        runtime_dict[pol_name]['p_eq'].append(equal_power)

        runtime_dict[pol_name]['x_ca'].append(ca_states)
        runtime_dict[pol_name]['p_ca'].append(ca_power)

        runtime_dict[pol_name]['x_rr'].append(rr_states)
        runtime_dict[pol_name]['p_rr'].append(rr_power)

        runtime_dict[pol_name]['x_wmmse'].append(wmmse_states)
        runtime_dict[pol_name]['p_wmmse'].append(wmmse_power)

        runtime_dict[pol_name]['x_ra'].append(ca_states)
        runtime_dict[pol_name]['p_ra'].append(ca_power)

    print("Saving test data")
    options_name = test_type + alg_name + 'T' + str(T) + '.mat'
    if new_options_name is not None:
        options_name = new_options_name
    file_path = save_dir + '/wireless_control_test' + options_name
    save_rt_data(runtime_dict, file_path)  # save "runtime" data (after training)
