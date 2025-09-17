#!/usr/bin/env python3
"""
Script to generate expert data for imitation learning
生成专家数据用于模仿学习
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import AdHoc.config_downlinkconstraint as config
import LQREnvs_HSR as LQREnvs
import BehaviorCloning
import WirelessNets

def generate_expert_data():
    """Generate expert trajectories using WMMSE expert"""
    print("=" * 60)
    print("Generating Expert Data for Imitation Learning")
    print("=" * 60)
    
    # Load or create network
    assign = 0
    net_name = 'L' + '_n' + str(config.n) + '_k' + str(config.k) + '.npy'
    try:
        L = np.load(net_name)
        print(f"Loaded network from {net_name}")
    except IOError:
        print('Warning: Network data not found. Creating HSR network...')
        L = WirelessNets.build_hsr_network_hata_samedistance(config.num_users)
        np.save(net_name, L)
        print(f"Created and saved network to {net_name}")
    
    # Create expert environment
    expert_downlink_envtype = LQREnvs.LQRAdHocDownlinkOutputConstraint
    expert_downlink_env = expert_downlink_envtype(
        config.num_users, config.downlink_upper_bound,
        config.constraint_dim, L, assign, config.n,
        config.k, mu=config.mu,
        T=config.T, pl=config.pl,
        pp=config.pp, p0=config.p0, num_features=config.n_feats,
        scaling=config.scale_obs, gamma=config.gamma
    )
    
    # Create expert
    expert = BehaviorCloning.WMMSEScaledExpert(
        expert_downlink_env, config.num_users, config.n, config.k,
        config.gamma, nepochs=10, pp=config.pp, n_feats=config.n_feats
    )
    
    # Set up paths
    exp_path = '../../GRL_HSR_JPAPS2/appendix/wmmse_allocation_adhoc.npz'
    
    # Create directory if it doesn't exist
    exp_dir = os.path.dirname(exp_path)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Created directory: {exp_dir}")
    
    # Generate expert trajectories
    print(f"Generating expert trajectories...")
    print(f"Episodes: {config.n_epochs_pretrain}")
    print(f"Steps per episode: {config.T}")
    print(f"Output path: {exp_path}")
    
    expert_dict = BehaviorCloning.expert_traj(
        expert_downlink_env, expert, 
        n_steps=config.T, 
        n_epochs=config.n_epochs_pretrain, 
        expert_path=exp_path, 
        gamma=config.gamma
    )
    
    print("=" * 60)
    print("Expert Data Generation Complete!")
    print(f"Generated {len(expert_dict['observations'][0])} expert trajectories")
    print(f"Saved to: {exp_path}")
    print("=" * 60)
    
    return expert_dict

if __name__ == "__main__":
    try:
        expert_data = generate_expert_data()
        print("✓ Expert data generation successful!")
    except Exception as e:
        print(f"✗ Expert data generation failed: {e}")
        import traceback
        traceback.print_exc()
