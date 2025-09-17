###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard Libraries
import time
import sys
import numpy as np

# Local files
sys.path.append('/home/vinicius/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/')
sys.path.append('/home/vinicius/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/GNNs')
sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/')
sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/GNNs')

import MultiCellScheduling.config_scheduling as config
import Trainer
import LQREnvs as LQREnvs
import BehaviorCloning
import WirelessNets

###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__':
    start_time_overall = time.time()
    test_type = 'zero'
    policy_name = 'stable-baselines-ppo'

    # Environment types
    downlink_env_type = LQREnvs.LQRMultiCellDownlinkScheduling
    expert_downlink_envtype = LQREnvs.LQRMultiCellDownlinkScheduling

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

    num_users = L.shape[0]
    expert_downlink_env = expert_downlink_envtype(num_users, config.downlink_upper_bound,
                                                  config.constraint_dim, L, assign, config.n,
                                                  config.k, mu=config.mu, p=config.p, q=config.q, T=config.T,
                                                  pl=config.pl, a0=config.a0, Ao=None, r=config.r,
                                                  pp=config.pp, p0=config.p0, num_features=config.n_feats,
                                                  scaling=config.scale_obs, gamma=config.gamma)
    expert = BehaviorCloning.WMMSEBinaryExpert(expert_downlink_env, num_users, config.n, config.k,
                                                             config.gamma, nepochs=10, p=config.p, pp=config.pp,
                                                             n_feats=config.n_feats)

    # Creates expert dataset
    exp_path = 'wmmse_allocation'
    local_exp_path = '../../../../../../../Documents/WCS/BehaviorCloning/'

    exp_path = 'wmmse_allocation.npz'

    # Saving agents, training data
    alg_name = 'DownlinkSchedulingDNNAgentPPONoPreTrain_T30_kvarying_n' + str(config.n)

    Trainer.run_mlp(downlink_env_type, config.downlink_upper_bound,
                    config.constraint_dim, L, assign, config.n, config.k, config.mu, config.p, config.q,
                    config.T, config.pl, config.a0, None, config.r, config.pp, config.n_feats,
                    config.scale_obs, config.gamma, config.p0, config.n_actors, config.n_total_timesteps,
                    config.save_dir, expert, expert_downlink_envtype,
                    pretrain=False, norm_obs=config.normalize_obs,
                    norm_rewards=config.normalize_rewards,
                    constraint=False, exp_path=exp_path,
                    pretrain_batch_size=config.pre_train_batch_size, train_batch_size=config.train_batch_size,
                    num_users=num_users, policy_name=policy_name,
                    alg_name=alg_name, lambda_0=1, lambda_lr=config.lambda_lr,
                    agent_lr=config.rl_lr, supervised_lr=config.superv_lr, n_steps=config.n_steps_reinforce)


