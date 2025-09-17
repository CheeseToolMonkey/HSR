###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard Libraries
import time
import sys
import numpy as np

# Local files
sys.path.append('C:\\Users\\AWAITXM\\PycharmProjects\\GRL_WCSs_Clean\\')
sys.path.append('C:\\Users\\AWAITXM\\PycharmProjects\\GRL_WCSs_Clean\\GNNs')
# sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/')
# sys.path.append('/home/viniciuslima/Dropbox/PhD/Research/WL_control_RL/simulations/GRL_WCSs_PyTorch_Clean/GNNs')

import AdHoc.config_downlinkconstraint as config
import Trainer
import LQREnvs_AdHoc as LQREnvs
import BehaviorCloning
import WirelessNets

###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__':
    start_time_overall = time.time()
    test_type = 'zero'
    policy_name = 'stable-baselines-ppo'
    downlink_env_type = LQREnvs.LQRAdHocDownlink
    expert_downlink_envtype = LQREnvs.LQRAdHocDownlinkOutputConstraint

    # Network data
    assign = 0
    net_name = 'L' + '_n' + str(config.n) + '_k' + str(config.k) + '.npy'
    try:
        L = np.load(net_name)
        # = np.load('L.npy')
    except IOError:
        print('Warning: Network data not found. Redrawing adhoc network. \n')
        # Adhoc network
        L = WirelessNets.build_adhoc_network(config.num_users, config.pl)
        np.save(net_name, L)

    expert_downlink_env = expert_downlink_envtype(config.num_users, config.downlink_upper_bound,
                                                  config.constraint_dim, L, assign, config.n,
                                                  config.k, mu=config.mu, p=config.p, q=config.q, T=config.T,
                                                  pl=config.pl, a0=config.a0, Ao=config.Ao, r=config.r,
                                                  pp=config.pp, p0=config.p0, num_features=config.n_feats,
                                                  scaling=config.scale_obs, gamma=config.gamma)
    expert = BehaviorCloning.WMMSEScaledExpert(expert_downlink_env, config.num_users, config.n, config.k,
                                               config.gamma, nepochs=10, p=config.p, pp=config.pp,
                                               n_feats=config.n_feats)

    # Creates expert dataset
    exp_path = '../../GRL_WCSs_Clean/appendix/wmmse_allocation_adhoc'
    local_exp_path = '../../GRL_WCSs_Clean/appendix/Documents/WCS/BehaviorCloning/'
    exp_path = '../../GRL_WCSs_Clean/appendix/wmmse_allocation_adhoc.npz'

    # Saving agents, training data
    alg_name = 'AdHocDownlinkConstraintDNNAgentPPO_n' + str(config.n) + '_k' + str(config.k)

    Trainer.run_mlp(downlink_env_type, config.downlink_upper_bound,
                    config.constraint_dim, L, assign, config.n, config.k, config.mu, config.p, config.q,
                    config.T, config.pl, config.a0, config.Ao, config.r, config.pp, config.n_feats,
                    config.scale_obs, config.gamma, config.p0, config.n_actors, config.n_total_timesteps,
                    config.save_dir, expert, expert_downlink_envtype,
                    pretrain=config.downlink_pretrain, norm_obs=config.normalize_obs,
                    norm_rewards=config.normalize_rewards,
                    constraint=True, exp_path=exp_path,
                    pretrain_batch_size=config.pre_train_batch_size, train_batch_size=config.train_batch_size,
                    num_users=config.num_users, policy_name=policy_name,
                    alg_name=alg_name, lambda_0=1, lambda_lr=config.lambda_lr,
                    agent_lr=config.rl_lr, supervised_lr=config.superv_lr, n_steps=config.n_steps_reinforce)
