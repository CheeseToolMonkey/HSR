###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard Libraries
import time
import sys
import numpy as np
import scipy.io
import math  # Added for doppler_shift_effect_helper
from scipy.special import j0  # Added for interference_packet_delivery_rate_helper

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stable Baselines: RL algorithms
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device, obs_as_tensor
from stable_baselines3.common.vec_env import VecNormalize  # Used for normalization
from stable_baselines3.common.running_mean_std import RunningMeanStd

# Local files
# Using LQREnvs_HSR.py as the environment file
import LQREnvs_HSR as LQREnvs  # Import the HSR environment
from AdHoc.config_downlinkconstraint import max_delay

# --- Start: WMMSE algorithm helper functions (moved from HSR_Envs_AdHoc.py) ---

# Helper function for doppler_shift_effect, as it's used by interference_packet_delivery_rate_helper
def doppler_shift_effect_helper(t, v, d, R, f_carrier, c_light=3e8):
    med = R ** 2 - d ** 2
    b = math.sqrt(med)
    b -= v * t
    r = math.sqrt(b ** 2 + d ** 2)  # Ensure r is positive for log10 later if applicable
    if r < 1e-9:  # Avoid division by zero if r is zero or very small
        cos_theta = 1.0
    else:
        cos_theta = b / r
    doppler_effect_num = f_carrier * (v * cos_theta) / c_light
    return doppler_effect_num, r


# Helper function for interference_packet_delivery_rate, as it's used by wmmse_power_allocation
# This function calculates SINR and throughput for a given H and actions (ground power)
def interference_packet_delivery_rate_helper(H, actions, bandwidth, sigma, t_time_step, v_speed, d_dist, R_radius,
                                             f_carrier_freq):
    actions_vec = actions[:, None]
    H_diag = np.diag(H)
    num = np.multiply(H_diag, actions)

    doppler_effect_coefficent, _ = doppler_shift_effect_helper(t_time_step, v_speed, d_dist, R_radius, f_carrier_freq)
    x_values = np.linspace(-1, 1, 1000)
    integral_result = np.trapz((1 - abs(x_values)) * j0(2 * np.pi * doppler_effect_coefficent * x_values), x_values)
    W_ICI = 1 - integral_result
    W_ICI = max(0.0, W_ICI)

    H_interference = (H - np.diag(np.diag(H))).transpose()
    den = (np.dot(H_interference * (1 + W_ICI), actions_vec) + sigma ** 2).flatten()

    den[den < 1e-9] = 1e-9
    SINR = num / den
    pdr = 1 - np.exp(- SINR)
    pdr = np.nan_to_num(pdr)
    channel_capacity_rate = bandwidth * np.log2(1 + SINR)
    effective_throughput_rate = channel_capacity_rate * pdr
    throughput = effective_throughput_rate * t_time_step

    return SINR, pdr, throughput


# Improved WMMSE Power Allocation algorithm
# This function takes the channel matrix H, and environment-specific parameters needed for its calculation
def wmmse_power_allocation(S_matrix, p0, sigma_noise, max_total_power=None):
    """
    Improved WMMSE power allocation algorithm with better initialization and constraints.
    
    Args:
        S_matrix: Channel gain matrix H, shape (num_users, num_users)
        p0: Base transmit power per user
        sigma_noise: Noise variance
        max_total_power: Maximum total power constraint (if None, uses p0 * num_users)
    
    Returns:
        p_alloc: Optimized power allocation, shape (num_users,)
    """
    num_users = S_matrix.shape[0]
    
    # Use total power budget instead of per-user power limit
    if max_total_power is None:
        max_total_power = p0 * num_users
    
    # h_squared is H^2, h is sqrt(H^2)
    h_squared = np.copy(S_matrix)
    
    # Improve numerical stability for extreme channel conditions
    # Clip very small values to avoid numerical issues
    h_squared = np.maximum(h_squared, 1e-8)
    h_sqrt = np.sqrt(h_squared)

    # h_diag represents the direct channel gains for each user, i.e., diag(sqrt(H^2))
    h_diag = np.diag(h_sqrt)

    # Improved initialization strategy for extreme channel conditions
    direct_gains = np.diag(S_matrix)
    
    # Handle extreme channel conditions
    if np.sum(direct_gains) > 0:
        # Use logarithmic scaling to reduce extreme differences
        log_gains = np.log(direct_gains + 1e-8)
        # Normalize and apply softmax-like scaling
        exp_gains = np.exp(log_gains - np.max(log_gains))
        initial_power = (exp_gains / np.sum(exp_gains)) * max_total_power
    else:
        # Equal allocation if all gains are zero
        initial_power = np.ones(num_users) * (max_total_power / num_users)
    
    # Initialize transmit power variables x_i = sqrt(p_i)
    x_current = np.sqrt(initial_power)

    # Improved convergence parameters for extreme channel conditions
    T_wmmse_iters = 300  # More iterations for complex scenarios
    tolerance = 1e-5  # Relaxed tolerance for numerical stability
    prev_power = np.copy(x_current)

    for iter_count in range(T_wmmse_iters):
        # Step 1: Update receive filters u_i
        # u_i = (h_ii * x_i) / (sum_{j} (h_ij^2 * x_j^2) + sigma_noise)
        
        # Calculate sum_{j} (h_ij^2 * x_j^2) for each receiver i
        interference_plus_noise_den = np.dot(h_squared, x_current ** 2) + sigma_noise

        # Avoid division by zero with better numerical stability
        interference_plus_noise_den[interference_plus_noise_den < 1e-8] = 1e-8

        u_current = (h_diag * x_current) / interference_plus_noise_den
        
        # Ensure u_current is numerically stable
        u_current = np.nan_to_num(u_current, nan=0.0, posinf=1.0, neginf=0.0)

        # Step 2: Update MMSE weights w_i
        # w_i = 1 / (1 - u_i * h_ii * x_i)
        denominator_w = 1 - u_current * h_diag * x_current
        # Avoid division by zero
        denominator_w[denominator_w < 1e-8] = 1e-8
        w_current = 1 / denominator_w
        # Handle potential NaN/Inf from division by zero or very small numbers
        w_current = np.nan_to_num(w_current, nan=1.0, posinf=1.0, neginf=1.0)

        # Step 3: Update transmit power variables x_i
        # x_i = (w_i * u_i * h_ii) / (sum_{j} (w_j * u_j^2 * h_ji^2))
        denominator_x = np.dot(h_squared.T, w_current * u_current ** 2) + 1e-8

        x_current = (w_current * u_current * h_diag) / denominator_x
        
        # Ensure numerical stability
        x_current = np.nan_to_num(x_current, nan=0.0, posinf=np.sqrt(max_total_power), neginf=0.0)

        # Apply total power constraint instead of per-user constraint
        current_total_power = np.sum(x_current ** 2)
        if current_total_power > max_total_power:
            scale_factor = np.sqrt(max_total_power / current_total_power)
            x_current = x_current * scale_factor
        
        # Ensure non-negative power
        x_current = np.maximum(0, x_current)

        # Check convergence
        if np.linalg.norm(x_current - prev_power) < tolerance:
            break
        prev_power = np.copy(x_current)

    p_alloc = x_current ** 2  # Final power allocation, shape (num_users,)
    
    # Final total power constraint enforcement
    total_power = np.sum(p_alloc)
    if total_power > max_total_power:
        scale_factor = max_total_power / total_power
        p_alloc = p_alloc * scale_factor
    
    return p_alloc


# --- End: WMMSE algorithm helper functions ---


# --- Start: Utility functions for imitation learning ---
def compute_returns(rewards, gamma):
    """
    Compute discounted returns for a sequence of rewards.
    
    Args:
        rewards: List or array of rewards
        gamma: Discount factor
        
    Returns:
        returns: Array of discounted returns
    """
    T = len(rewards)
    returns = np.zeros(T)
    returns[-1] = rewards[-1]

    for tt in np.arange(T-1):
        returns[T - 1 - tt - 1] = rewards[T - 1 - tt - 1] + gamma*returns[T - 1 - tt]
    return returns


def expert_traj(env, expert, n_steps: int, n_epochs: int, expert_path, gamma=0.95):
    """
    Generate expert trajectories for imitation learning.
    
    Args:
        env: Environment instance
        expert: Expert policy instance
        n_steps: Number of steps per episode
        n_epochs: Number of episodes to generate
        expert_path: Path to save expert data
        gamma: Discount factor
        
    Returns:
        expert_dict: Dictionary containing expert trajectories
    """
    print('Generating Expert Trajectories: \n')
    expert_obs = []
    expert_actions = []
    expert_returns = []
    expert_rews = []
    expert_dict = {'actions': [],
                   'observations': [],
                   'returns': []
                   }
    for jj in np.arange(n_epochs):
        if (jj + 1)%10 == 0:
            print('Epoch ' + str(jj + 1) + ' of ' + str(n_epochs) + '.\n')
        expert_rews = []
        obs = env.reset()
        for _ in np.arange(n_steps):
            action = expert.get_action(obs)
            expert_obs.append(obs)
            expert_actions.append(action)
            obs, rew, _, _ = env.step(action)
            expert_rews.append(rew)
        # computes discounted returns / cost-to-go
        ep_returns = compute_returns(expert_rews, gamma)
        expert_returns.append(ep_returns)
    expert_dict['actions'].append(expert_actions)
    expert_dict['observations'].append(expert_obs)
    expert_dict['returns'].append(expert_returns)

    np.savez(expert_path, **expert_dict)

    return expert_dict


# --- Start: DAgger (Dataset Aggregation) implementation ---
def dagger_init(agent, expert, expert_env, n_epochs, n_steps, n_miniepochs=20, batch_size=10, normalization=True,
                num_users=10, clip_obs=10., epsilon=1e-8, n_feats=4, clip_reward=10., vf_coef=.5, device='cuda:0',
                actor_critic=True, expert_prob=0.5, gamma=0.95, ent_coef=1e-3, max_grad_norm=None):
    """
    DAgger (Dataset Aggregation) implementation for throughput maximization.
    Updated to match LQREnvs_HSR observation space structure.
    """
    print('Supervised Initialization: DAGGER for Throughput Maximization \n')
    SupervLoss = nn.MSELoss()
    SupervLosses = []

    # Normalization for HSR environment observation components
    channel_rms = RunningMeanStd(shape=())
    power_rms = RunningMeanStd(shape=())
    delay_loss_rms = RunningMeanStd(shape=())
    rates_rms = RunningMeanStd(shape=())
    min_expert_prob = .4
    current_round = 0
    n_rounds = 20

    for jj in np.arange(n_epochs):
        # updates expert probability
        expert_prob = min(1, max(min_expert_prob, (n_rounds-current_round)/n_rounds))
        current_round += 1
        expert_obs = []
        expert_actions = []
        expert_returns = []
        expert_rews = []
        print('Epoch ' + str(jj + 1) + ' of ' + str(n_epochs) + '.\n')

        for _ in np.arange(n_miniepochs):
            expert_rews = []
            obs = expert_env.reset()

            for _ in np.arange(n_steps):
                norm_obs = obs
                if normalization:
                    # Parse observation components for HSR environment
                    channel_obs = obs[:num_users ** 2]  # Channel matrix
                    power_obs = obs[num_users ** 2:num_users ** 2 + num_users]  # Power allocation
                    delay_loss_obs = obs[num_users ** 2 + num_users:num_users ** 2 + 2*num_users]  # Delay loss
                    rates_obs = obs[num_users ** 2 + 2*num_users:num_users ** 2 + 3*num_users]  # Channel rates
                    
                    # Update running statistics
                    channel_rms.update(channel_obs)
                    power_rms.update(power_obs)
                    delay_loss_rms.update(delay_loss_obs)
                    rates_rms.update(rates_obs)

                    # Normalize observations
                    channel_obs = np.clip((channel_obs - channel_rms.mean) / np.sqrt(channel_rms.var + epsilon), -clip_obs, clip_obs)
                    power_obs = np.clip((power_obs - power_rms.mean) / np.sqrt(power_rms.var + epsilon), -clip_obs, clip_obs)
                    delay_loss_obs = np.clip((delay_loss_obs - delay_loss_rms.mean) / np.sqrt(delay_loss_rms.var + epsilon), -clip_obs, clip_obs)
                    rates_obs = np.clip((rates_obs - rates_rms.mean) / np.sqrt(rates_rms.var + epsilon), -clip_obs, clip_obs)
                    
                    # Reconstruct normalized observations
                    norm_obs = np.hstack((channel_obs, power_obs, delay_loss_obs, rates_obs))

                trial = np.random.binomial(1, expert_prob)
                if trial:
                    action = expert.get_action(obs)
                else:
                    with torch.no_grad():
                        action, _ = agent.predict(norm_obs)
                expert_obs.append(norm_obs)
                expert_actions.append(action)
                obs, rew, _, _ = expert_env.step(action)
                expert_rews.append(rew)
            ep_returns = compute_returns(expert_rews, gamma)
            expert_returns.append(ep_returns)

            if len(expert_obs) > batch_size:
                nsamples = len(expert_obs)
                noptepochs = int(nsamples / batch_size)
                train_obs = np.asarray(expert_obs)
                train_actions = np.asarray(expert_actions)
                train_rets = np.asarray(expert_returns).reshape(-1)

                # shuffling actions, observations. converting to tensor
                p = np.random.permutation(len(expert_obs))
                expert_actions_th = torch.from_numpy(train_actions[p]).to(device).float()
                train_obs = train_obs[p]
                expert_returns_th = torch.from_numpy(train_rets[p]).to(device).float()

                for ll in np.arange(noptepochs):
                    batchobs = train_obs[ll*batch_size:(ll+1)*batch_size, :]
                    batchactions_expert = expert_actions_th[ll*batch_size:(ll+1)*batch_size, :]
                    batchreturns_expert = expert_returns_th[ll*batch_size:(ll+1)*batch_size]
                    batchobs_th = torch.from_numpy(batchobs).to(device).float()
                    batchobs_agent = torch.from_numpy(batchobs).to(device).float()

                    if actor_critic:
                        _, log_prob, entropy = agent.policy.evaluate_actions(batchobs_th, batchactions_expert)
                    else:
                        # if following the imitation library
                        log_prob, entropy = agent.policy.evaluate_actions(batchobs_th, batchactions_expert)
                    log_prob = log_prob.mean()
                    entropy = entropy.mean()
                    ent_loss = - ent_coef * entropy
                    neglogp = -log_prob
                    EpochLoss = neglogp + ent_loss

                    # Optimization step
                    agent.policy.optimizer.zero_grad()
                    EpochLoss.float().backward()
                    if max_grad_norm is not None:
                        params = list(agent.policy.parameters())
                        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    agent.policy.optimizer.step()
                    EpochLossNp = EpochLoss.clone().detach().cpu().numpy()
                    SupervLosses.append(EpochLossNp)
                    del EpochLoss

        print('Imitation Learning Loss: ', EpochLossNp, '\n')
    dataset_mean = [channel_rms.mean, power_rms.mean, delay_loss_rms.mean, rates_rms.mean]
    dataset_var = [channel_rms.var, power_rms.var, delay_loss_rms.var, rates_rms.var]

    return agent, SupervLosses, dataset_mean, dataset_var


def supervised_init(agent, expert_file, batch_size, normalization=True, num_users=10, clip_obs=10., epsilon=1e-8,
                    n_feats=4, clip_reward=10., vf_coef=.5, device='cuda:0', actor_critic=True):
    """
    Supervised pre-training from a pre-collected expert dataset for throughput maximization.
    Updated to match LQREnvs_HSR observation space structure.
    """
    print('Supervised Initialization for Throughput Maximization \n')
    SupervLoss = nn.MSELoss()
    SupervLosses = []

    # Normalization for HSR environment observation components
    channel_mean = 0.
    channel_var = 1.
    power_mean = 0.
    power_var = 1.
    delay_loss_mean = 0.
    delay_loss_var = 1.
    rates_mean = 0.
    rates_var = 1.

    # Expert
    try:
        expert_dict = dict(np.load(expert_file, allow_pickle=True))
        obs = np.squeeze(expert_dict['observations'])
        actions = np.squeeze(expert_dict['actions'])
        returns = np.squeeze(expert_dict['returns'].reshape(-1))
    except FileNotFoundError:
        print(f"Error: Expert dataset not found at {expert_file}.")
        print("Note: Expert data generation is not implemented in supervised_init.")
        print("Please use dagger_init instead, or generate expert data separately.")
        return agent, [], np.zeros(agent.observation_space.shape), np.ones(agent.observation_space.shape)
    except KeyError as e:
        print(f"Error: Missing key {e} in expert dataset. Please check the dataset format.")
        return agent, [], np.zeros(agent.observation_space.shape), np.ones(agent.observation_space.shape)

    if normalization:
        # Parse observation components for HSR environment
        # obs structure: [channel_matrix, power_allocation, delay_loss_suffered, channel_rates]
        channel_obs = obs[:, :num_users**2]  # Channel matrix
        power_obs = obs[:, num_users**2:num_users**2 + num_users]  # Power allocation
        delay_loss_obs = obs[:, num_users**2 + num_users:num_users**2 + 2*num_users]  # Delay loss
        rates_obs = obs[:, num_users**2 + 2*num_users:num_users**2 + 3*num_users]  # Channel rates
        
        # Calculate normalization statistics
        channel_mean = channel_obs.mean()
        channel_var = channel_obs.var()
        power_mean = power_obs.mean()
        power_var = power_obs.var()
        delay_loss_mean = delay_loss_obs.mean()
        delay_loss_var = delay_loss_obs.var()
        rates_mean = rates_obs.mean()
        rates_var = rates_obs.var()
        returns_mean = returns.mean()
        returns_var = returns.var()
        
        # Normalize observations
        channel_obs = np.clip((channel_obs - channel_mean) / np.sqrt(channel_var + epsilon), -clip_obs, clip_obs)
        power_obs = np.clip((power_obs - power_mean) / np.sqrt(power_var + epsilon), -clip_obs, clip_obs)
        delay_loss_obs = np.clip((delay_loss_obs - delay_loss_mean) / np.sqrt(delay_loss_var + epsilon), -clip_obs, clip_obs)
        rates_obs = np.clip((rates_obs - rates_mean) / np.sqrt(rates_var + epsilon), -clip_obs, clip_obs)
        returns = np.clip(returns / np.sqrt(returns_var + epsilon), -clip_reward, clip_reward)
        
        # Reconstruct normalized observations
        obs[:, :num_users**2] = channel_obs
        obs[:, num_users**2:num_users**2 + num_users] = power_obs
        obs[:, num_users**2 + num_users:num_users**2 + 2*num_users] = delay_loss_obs
        obs[:, num_users**2 + 2*num_users:num_users**2 + 3*num_users] = rates_obs

        # Clean up temporary variables only if normalization was performed
        del channel_obs
        del power_obs
        del delay_loss_obs
        del rates_obs

    # shuffling actions, observations. converting to tensor
    p = np.random.permutation(len(actions))
    expert_actions = torch.from_numpy(actions[p]).to(device).float()
    expert_observations = obs[p]
    expert_returns = torch.from_numpy(returns[p]).to(device).float()

    nsamples = len(obs)
    noptepochs = int(nsamples / batch_size)

    del obs
    del actions
    del returns

    for jj in np.arange(noptepochs):
        batchobs = expert_observations[jj*batch_size:(jj+1)*batch_size, :]
        batchactions_expert = expert_actions[jj*batch_size:(jj+1)*batch_size, :]
        batchreturns_expert = expert_returns[jj*batch_size:(jj+1)*batch_size]
        batchobs_th = torch.from_numpy(batchobs).to(device)
        batchobs_agent = torch.from_numpy(batchobs).to(device).float()

        if actor_critic:
            batchactions_agent, _, _ = agent.policy.forward(batchobs_th)
            batchactions_agent = batchactions_agent.to(device)
            values, _, _ = agent.policy.evaluate_actions(batchobs_agent, batchactions_agent)
            values = values.flatten()
            EpochLoss = SupervLoss(batchactions_agent, batchactions_expert) + \
                        vf_coef * SupervLoss(batchreturns_expert, values)
        else:
            batchactions_agent, _ = agent.policy.forward(batchobs_th)
            batchactions_agent = batchactions_agent.to(device)
            EpochLoss = SupervLoss(batchactions_agent, batchactions_expert)

        # Optimization step
        agent.policy.optimizer.zero_grad()
        EpochLoss.float().backward()
        agent.policy.optimizer.step()
        EpochLossNp = EpochLoss.clone().detach().cpu().numpy()
        SupervLosses.append(EpochLossNp)

    dataset_mean = [channel_mean, power_mean, delay_loss_mean, rates_mean]
    dataset_var = [channel_var, power_var, delay_loss_var, rates_var]

    del expert_dict

    print(f"Supervised pre-training finished. Final loss: {np.mean(SupervLosses):.4f}")
    return agent, SupervLosses, dataset_mean, dataset_var


# --- Start: WMMSEScaledExpert Class ---
class WMMSEScaledExpert(object):
    """
    Expert policy using WMMSE for power allocation in HSR environment.
    Optimized for maximum throughput within power constraints.
    Updated for throughput maximization objective.
    """

    def __init__(self, env, num_users, n=3, k=5, gamma=0.99, nepochs=10, pp=5., n_feats=4):
        self.env = env
        self.num_users = num_users
        self.pp = pp  # max_pwr_perplant
        self.gamma = gamma
        self.n_feats = n_feats  # Number of features per user
        self.nepochs = nepochs
        self.p0 = env.p0  # Base power from environment
        self.sigma = env.sigma  # Noise variance from environment
        self.bandwidth = env.bandwidth  # Bandwidth for throughput calculation

        # Store environment-specific parameters needed for WMMSE
        self.env_bandwidth = env.bandwidth
        self.env_t = env.t
        self.env_v = env.v
        self.env_d = env.d
        self.env_R = env.R
        self.env_f = env.f  # Carrier frequency

    def get_action(self, obs, explore=True):
        """
        Get WMMSE-based power allocation action for maximum throughput.
        
        Args:
            obs: Observation vector containing channel matrix and other state info
            explore: Whether to add exploration noise (not used in expert)
            
        Returns:
            expert_action: Scaled power allocation action in [-1, 1] range
        """
        return self._step(obs)
    
    def _step(self, obs):
        """
        Internal step method for WMMSE-based power allocation.
        Updated to match LQREnvs_HSR observation space structure.
        
        Args:
            obs: Observation vector containing channel matrix and other state info
                Structure: [channel_matrix, power_allocation, delay_loss_suffered, channel_rates]
            
        Returns:
            expert_action: Scaled power allocation action in [-1, 1] range
        """
        try:
            # Parse observation for LQREnvs_HSR
            # obs structure: [channel_matrix, power_allocation, delay_loss_suffered, channel_rates]
            channel_obs_flat = obs[:self.num_users ** 2]
            H = channel_obs_flat.reshape(self.num_users, self.num_users)  # Reshape channel state matrix
            
            # Ensure H is valid (no NaN or infinite values)
            H = np.nan_to_num(H, nan=1e-6, posinf=1e-6, neginf=1e-6)
            
            # Apply power constraint: ensure total power doesn't exceed limit
            max_total_power = self.p0 * self.num_users
            
            # --- WMMSE for Power Allocation ---
            downlink_action_wmmse = self._optimized_wmmse_power_allocation(H, max_total_power)
            
            # Scale the WMMSE output to [-1, 1] range to match action_space
            # WMMSE output is power in Watts, scale to [0, max_pwr_perplant] then to [-1, 1]
            scaled_wmmse_power = downlink_action_wmmse / self.pp  # Normalize to [0,1]
            scaled_wmmse_power = np.clip(scaled_wmmse_power, 0, 1)  # Ensure [0,1] range
            scaled_wmmse_power = scaled_wmmse_power * 2 - 1  # Scale to [-1,1]
            
            # For LQREnvs_HSR, we only return power allocation (no satellite assistance)
            expert_action = scaled_wmmse_power
            
            return expert_action
            
        except Exception as e:
            print(f"Error in WMMSE expert: {e}")
            # Fallback: return uniform power allocation
            uniform_power = np.ones(self.num_users) * 0.5  # 50% of max power per user
            return uniform_power * 2 - 1  # Scale to [-1,1]

    def _optimized_wmmse_power_allocation(self, H, max_total_power):
        """
        Optimized WMMSE power allocation with improved algorithm.
        
        Args:
            H: Channel gain matrix [num_users, num_users]
            max_total_power: Maximum total power constraint
            
        Returns:
            power_allocation: Optimized power allocation [num_users]
        """
        try:
            # Use the improved WMMSE implementation with total power constraint
            power_allocation = wmmse_power_allocation(H, self.p0, self.sigma, max_total_power)
            
            # Validate the power allocation
            if self._validate_power_allocation(power_allocation, H):
                return power_allocation
            else:
                # Debug information for validation failure
                direct_gains = np.diag(H)
                print(f"WMMSE result invalid, using fallback allocation")
                print(f"  - Power allocation: min={np.min(power_allocation):.6f}, max={np.max(power_allocation):.6f}, std={np.std(power_allocation):.6f}")
                print(f"  - Channel gains: min={np.min(direct_gains):.6f}, max={np.max(direct_gains):.6f}, std={np.std(direct_gains):.6f}")
                print(f"  - Total power: {np.sum(power_allocation):.3f} / {max_total_power:.3f}")
                print(f"  - Has NaN: {np.any(np.isnan(power_allocation))}, Has Inf: {np.any(np.isinf(power_allocation))}")
                return self._channel_quality_based_allocation(H, max_total_power)
            
        except Exception as e:
            print(f"Error in WMMSE calculation: {e}")
            # Fallback: use channel-quality-based allocation
            return self._channel_quality_based_allocation(H, max_total_power)

    def _validate_power_allocation(self, power_allocation, H):
        """
        Simplified validation for power allocation - only check for basic sanity.
        
        Args:
            power_allocation: Power allocation to validate
            H: Channel gain matrix
            
        Returns:
            is_valid: True if allocation is valid, False otherwise
        """
        # Only perform basic sanity checks
        # Check for negative or excessive power values
        if np.any(power_allocation < 0) or np.any(power_allocation > self.p0 * 3):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(power_allocation)) or np.any(np.isinf(power_allocation)):
            return False
        
        # Check if total power is reasonable (not too low)
        total_power = np.sum(power_allocation)
        if total_power < 0.001 * self.p0 * self.num_users:  # Extremely low power usage
            return False
        
        # If we get here, the allocation is valid
        return True

    def _enforce_power_constraints(self, power_allocation, max_total_power):
        """
        Enforce power constraints to ensure total power doesn't exceed limit.
        
        Args:
            power_allocation: Initial power allocation
            max_total_power: Maximum total power constraint
            
        Returns:
            constrained_power: Power allocation satisfying constraints
        """
        # Individual power constraint: each user's power <= p0
        power_allocation = np.clip(power_allocation, 0, self.p0)
        
        # Total power constraint: sum of all powers <= max_total_power
        total_power = np.sum(power_allocation)
        if total_power > max_total_power:
            # Scale down proportionally
            scale_factor = max_total_power / total_power
            power_allocation = power_allocation * scale_factor
        
        return power_allocation

    def _channel_quality_based_allocation(self, H, max_total_power):
        """
        Improved fallback power allocation based on channel quality and interference.
        
        Args:
            H: Channel gain matrix
            max_total_power: Maximum total power constraint
            
        Returns:
            power_allocation: Channel-quality-based power allocation
        """
        # Get direct channel gains (diagonal elements)
        direct_gains = np.diag(H)
        
        # Calculate interference levels for each user
        interference_levels = np.sum(H - np.diag(np.diag(H)), axis=1)
        
        # Calculate effective channel quality (signal to interference ratio)
        effective_quality = direct_gains / (interference_levels + 1e-6)
        
        # Allocate power based on effective channel quality
        total_quality = np.sum(effective_quality)
        if total_quality > 0:
            # Use square root to reduce extreme allocations
            quality_weights = np.sqrt(effective_quality)
            total_weights = np.sum(quality_weights)
            if total_weights > 0:
                power_allocation = (quality_weights / total_weights) * max_total_power
            else:
                power_allocation = np.ones(self.num_users) * (max_total_power / self.num_users)
        else:
            # Equal allocation if all gains are zero
            power_allocation = np.ones(self.num_users) * (max_total_power / self.num_users)
        
        # Ensure individual power constraints
        power_allocation = np.clip(power_allocation, 0, self.p0)
        
        # Ensure total power constraint
        total_power = np.sum(power_allocation)
        if total_power > max_total_power:
            scale_factor = max_total_power / total_power
            power_allocation = power_allocation * scale_factor
        
        return power_allocation

    def calculate_throughput(self, H, power_allocation):
        """
        Calculate system throughput for given channel matrix and power allocation.
        
        Args:
            H: Channel gain matrix
            power_allocation: Power allocation vector
            
        Returns:
            total_throughput: Total system throughput in bps
        """
        try:
            # Calculate SINR for each user
            signal_powers = power_allocation * np.diag(H)
            interference_powers = np.dot(H - np.diag(np.diag(H)), power_allocation)
            sinr = signal_powers / (interference_powers + self.sigma ** 2)
            sinr = np.maximum(sinr, 1e-9)  # Avoid log(0)
            
            # Calculate channel rates
            channel_rates = self.bandwidth * np.log2(1 + sinr)
            
            # Total throughput
            total_throughput = np.sum(channel_rates)
            
            return total_throughput
            
        except Exception as e:
            print(f"Error calculating throughput: {e}")
            return 0.0
# --- End: WMMSEScaledExpert Class ---