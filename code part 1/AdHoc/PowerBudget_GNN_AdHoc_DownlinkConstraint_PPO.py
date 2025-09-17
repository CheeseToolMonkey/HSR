import time
import sys
import numpy as np
import random
import torch

# # 设置随机种子以确保实验可重复性
# def set_random_seeds(seed=42):
#     """设置所有随机数生成器的种子"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     print(f"随机种子已设置为: {seed}")

# # 设置随机种子
# set_random_seeds(42)

sys.path.append('C:\\Users\\89188\\Desktop\\code\\GRL_HSR_JPAPS2\\')
sys.path.append('C:\\Users\\89188\\Desktop\\code\\GRL_HSR_JPAPS2\\GNNs')

import AdHoc.config_downlinkconstraint as config
import Trainer
# import LQREnvs_HSR as LQREnvs
import HSR as LQREnvs

import BehaviorCloning
import WirelessNets
import link_data_logger

if __name__ == '__main__':
    start_time_overall = time.time()
    test_type = 'zero'
    policy_name = 'stable-baselines-ppo'
    downlink_env_type = LQREnvs.LQRAdHocDownlink
    expert_downlink_envtype = LQREnvs.LQRAdHocDownlinkOutputConstraint

    assign = 0
    net_name = 'L' + '_n' + str(config.n) + '_k' + str(config.k) + '.npy'
    try:
        L = np.load(net_name)
    except IOError:
        print('Warning: Network data not found. Redrawing adhoc network. \n')
        L = WirelessNets.build_adhoc_network(config.num_users, config.pl)
        L = WirelessNets.build_adhoc_network_samedistance(config.num_users, config.pl, 6.)
        # L = WirelessNets.build_hsr_network_hata_samedistance(config.num_users)
        np.save(net_name, L)
        
    # 初始化链路数据记录器
    link_logger = link_data_logger.LinkDataLogger(
        log_dir="link_data_logs", 
        num_users=config.num_users, 
        filename=f"link_data_n{config.n}_k{config.k}.csv",
        log_interval=50,  # 每隔50轮次记录一次完整数据
        n_actors=config.n_actors  # 添加actor数量信息
    )
    
    # 打印记录器信息
    print(f"链路数据记录器已初始化:")
    print(f"  - 记录间隔: 每隔 {link_logger.log_interval} 个episode记录一次")
    print(f"  - 用户数量: {config.num_users}")
    print(f"  - Actor数量: {config.n_actors}")
    print(f"  - 日志文件: {link_logger.data_file}")
    
    # 移除 p, q, Ao, r 参数
    expert_downlink_env = expert_downlink_envtype(config.num_users, config.downlink_upper_bound,
                                                  config.constraint_dim, L, assign, config.n,
                                                  config.k, mu=config.mu,
                                                  T=config.T, pl=config.pl,
                                                  pp=config.pp, p0=config.p0, num_features=config.n_feats,
                                                  scaling=config.scale_obs, gamma=config.gamma)
    # BehaviorCloning 专家初始化也移除 p 参数
    expert = BehaviorCloning.WMMSEScaledExpert(expert_downlink_env, config.num_users, config.n, config.k,
                                            config.gamma, nepochs=10, pp=config.pp, n_feats=config.n_feats)

    exp_path = '../../GRL_HSR_JPAPS2/appendix/wmmse_allocation_adhoc'
    local_exp_path = '../../GRL_HSR_JPAPS2/appendix/Documents/WCS/BehaviorCloning/'
    exp_path = '../../GRL_HSR_JPAPS2/appendix/wmmse_allocation_adhoc.npz'

    alg_name = 'AdHocDownlinkConstraintGNNAgentPPO_n' + str(config.n) + '_k' + str(config.k)

    # Check if expert data exists, if not use DAgger instead of supervised learning
    import os
    if not os.path.exists(exp_path):
        print(f"Expert data not found at {exp_path}")
        print("Using DAgger for imitation learning instead of supervised learning...")
        dagger_init = True
    else:
        print(f"Expert data found at {exp_path}")
        dagger_init = False
    
    # Trainer.run_gnn_ppo 调用移除 p, q, Ao, r 参数
    Trainer.run_gnn_ppo(downlink_env_type, config.downlink_upper_bound,
                    config.constraint_dim, L, assign, config.n, config.k, config.mu,
                    config.T, config.pl,
                    config.pp, config.n_feats,
                    config.scale_obs, config.gamma, config.p0, config.n_actors, config.n_total_timesteps,
                    config.save_dir, expert, expert_downlink_envtype,
                    pretrain=config.downlink_pretrain, norm_obs=config.normalize_obs,
                    norm_rewards=config.normalize_rewards,
                    constraint=True, exp_path=exp_path,
                    pretrain_batch_size=config.pre_train_batch_size, train_batch_size=config.train_batch_size,
                    num_users=config.num_users, policy_name=policy_name,
                    alg_name=alg_name, lambda_0=1, lambda_lr=config.lambda_lr,
                    agent_lr=config.rl_lr, supervised_lr=config.superv_lr, n_steps=config.n_steps_reinforce,
                    dagger_init=dagger_init, link_logger=link_logger)