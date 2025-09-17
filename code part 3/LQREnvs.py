###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

import numpy as np
import pdb
import scipy
import scipy.linalg
import scipy.io
import control
import gym
import math

from scipy.special import j0
from scipy.stats import bernoulli
from gym import spaces
from gym.utils import seeding
from WirelessNets import *
from StateEstimators import *


class LQR_Env(gym.Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign,
                 t, v, R, d, choice, A_b, A_m, f,
                 mu=1, p=2, q=1, Ao=None,
                 W=None, Wobs=None, Wobs_channels=None, T=40, a0=1.05,
                 gamma=0.99, r=0.001, pl=2., pp=5., p0=1., num_features=1, scaling=True):
        super(LQR_Env, self).__init__()
        # 计算多普勒所需参数：
        self.time = t
        self.v = v
        self.R = R
        self.d = d
        # 计算Hata模型损耗所需参数：
        self.A_b = A_b
        self.A_m = A_m
        self.f = f  # 载波频率
        self.choice = choice  # 郊区或者高架桥
        # dimensions
        self.num_features = num_features
        self.state_dim = num_users ** 2 + num_users * p
        self.action_dim = num_users
        self.constraint_dim = constraint_dim
        self.channel_state_dim = num_users ** 2
        self.control_state_dim = num_users * p  # number of agents \times features
        self.state_dim_dnn = self.channel_state_dim + num_users * num_features

        # using different seeds across different realizations of the WCS
        self.np_random = []
        self.seed()

        # control system parameters
        self.num_users = num_users
        self.p = p  # dimension of each plant state
        self.q = q  # dimension of control input
        self.r = r
        self.T = T
        self.max_pwr_perplant = pp
        self.p0 = p0
        if p == q:
            self.Bo = np.eye(q)
        else:
            self.Bo = np.ones((p, 1))

        # wireless network parameters
        self.mu = mu  # parameter for distribution of channel states (fast fading)
        self.sigma = 1.
        self.n_transmitting = 10  # np.rint(num_users/3).astype(np.int32)  # number of plants transmitting at a given time
        self.gamma = gamma
        self.upperbound = upperbound
        self.pl = pl  # path loss
        self.L = L  # build_adhoc_network(num_users, pl)
        self.assign = assign

        self.batch_size = 1
        self.cost_hist = []
        self.H = 0  # interference matrix

        self.max_control_state = 50.
        self.max_cost = self.max_control_state ** 2
        self.control_actions = []

        # open AI gym structure: separate between AdHoc, MultiCell, UpLink, Downlink envs!
        self.action_space = []
        self.observation_space = spaces.Box(low=-self.max_control_state * np.ones(self.state_dim_dnn),
                                            high=self.max_control_state * np.ones(self.state_dim_dnn))
        self.scaling = scaling

        if Ao is None:
            Ao = np.zeros((num_users, p, p))
            for ii in range(num_users):
                Ao[ii, :, :] = np.array([[a0, .2, .2], [0., a0, .2], [0, 0., a0]])
        else:
            if (Ao.shape != (num_users, p, p)):
                raise Exception("Ao is not the correct shape")
        self.Ao = Ao

        Ablk = Ao[0, :, :]
        for ii in range(1, num_users):
            Ablk = scipy.linalg.block_diag(Ablk, Ao[ii, :, :])
        self.A = Ablk

        # input matrix --- block version
        Bblk = self.Bo
        for ii in range(1, num_users):
            Bblk = scipy.linalg.block_diag(Bblk, self.Bo)
        self.B = Bblk

        # covariance matrix: random disturbance (plants)
        if W is None:
            self.W = np.eye(num_users * p)
        else:
            if W.shape != (1, num_users):
                raise Exception("W is not the correct shape")
            self.W = W

        # covariance matrix: observation noise
        if Wobs is None:
            self.Wobs = 1. * np.eye(num_users * p)
        else:
            if Wobs.shape != (num_users,):
                raise Exception("W is not the correct shape")
            self.Wobs = Wobs

        # Observation noise --- channel conditions
        if Wobs_channels is None:
            self.Wobs_channels = 1. * np.ones(num_users)
        else:
            if Wobs_channels.shape != (num_users, num_users):
                raise Exception("W channels is not the correct shape")
            self.Wobs_channels = Wobs_channels

        # feedback gain (LQR)
        fb_gain = np.zeros((num_users, q, p))
        for ii in range(num_users):
            (_, _, fb_gain[ii, :, :]) = control.dare(Ao[ii, :, :], self.Bo, np.eye(p), self.r * np.eye(q))  # ARE
        fb_gain_blk = fb_gain[0, :, :]
        for ii in range(1, num_users):
            fb_gain_blk = scipy.linalg.block_diag(fb_gain_blk, fb_gain[ii, :, :])
        self.fb_gain_ind = fb_gain[0, :, :]
        self.fb_gain = fb_gain_blk

        self.current_state = self.sample(batch_size=1)
        self.current_control_obs = []

        # to save training data
        self.current_episode = 0
        self.ep_cost_hist = []
        self.constraint_hist = []
        self.constraint_violation = 0
        self.ep_constraint = []
        self.Lagrangian_hist = []
        self.ep_Lagrangian = []
        self.downlink_constraint_dualvar = 0

        self.time_step = 0
        self.downlinkap_dnn = []
        self.downlinkap_gnn = []

    def disc_cost(self, cost_vec):
        T = np.size(cost_vec)
        cost_discounted = np.zeros(T)
        cost_discounted[-1] = cost_vec[-1]

        # calculating discounted return backwards (step by step)
        for k in range(1, T):
            cost_discounted[T - 1 - k] = cost_vec[T - 1 - k] + self.gamma * cost_discounted[T - 1 - k + 1]

        return cost_discounted

    def disc_constraint(self, cost_vec):
        T = np.size(cost_vec)
        if cost_vec.ndim > 1:
            T, constraint_dim = cost_vec.shape
            cost_discounted = np.zeros((T, constraint_dim))
        else:
            cost_discounted = np.zeros(T)
        cost_discounted[-1] = cost_vec[-1]

        # calculating discounted return backwards (step by step)
        for k in range(1, T):
            cost_discounted[T - 1 - k, :] = cost_vec[T - 1 - k, :] + self.gamma * cost_discounted[T - 1 - k + 1, :]

        return cost_discounted

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Normalizing H_t / GSO
    @staticmethod
    def normalize_gso(S):
        # norms = np.linalg.norm(S, ord=2, axis=(1, 2))
        norm = np.linalg.norm(S, ord=2, axis=None)
        Snorm = S / norm  # norms[:, None, None]
        return Snorm

    @staticmethod
    def normalize_inputs(inputs):
        input2 = inputs - inputs.mean(axis=1).reshape(-1, 1)
        return input2

    def normalize_obs(self, obs: np.ndarray, mean, var, epsilon=1e-8, clip_obs=10.) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # 观察空间的标准化处理
        channel_mean = mean[0]
        plants_mean = mean[1]
        interval_mean = mean[2]
        channel_var = var[0]
        plants_var = var[1]
        interval_var = var[2]
        # obs = np.clip((obs - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs)
        channel_obs = obs[:self.num_users ** 2]
        channel_obs = np.clip((channel_obs - channel_mean) / np.sqrt(channel_var + epsilon), -clip_obs, clip_obs)
        obs_aux = obs[self.num_users ** 2:].reshape(-1, self.num_features)
        plants_obs = obs_aux[:, 0]
        plants_obs = np.clip((plants_obs - plants_mean) / np.sqrt(plants_var + epsilon), -clip_obs, clip_obs)
        interval_obs = obs_aux[:, 1]
        interval_obs = np.clip((interval_obs - interval_mean) / np.sqrt(interval_var + epsilon), -clip_obs, clip_obs)
        obs_aux[:, 0] = plants_obs
        obs_aux[:, 1] = interval_obs
        obs_aux = obs_aux.reshape(-1)
        obs = np.hstack((channel_obs.flatten(), obs_aux.flatten()))
        return obs

    def control_plant_norm(self, control_obs):
        control_obs_aux = (np.square(control_obs.reshape(-1, self.p))).sum(axis=1)
        control_norms = np.sqrt(control_obs_aux)

        if self.scaling:
            control_norms /= self.max_control_state
            control_norms *= 10.

        return control_norms

    def control_action_norm(self, control_action_obs):
        control_obs_aux = (np.square(control_action_obs.reshape(-1, self.q))).sum(axis=1)
        control_norms = np.sqrt(control_obs_aux)

        if self.scaling:
            control_norms /= self.max_control_state

        return control_norms

    # packet delivery rate: no interference
    @staticmethod
    def packet_delivery_rate(snr_value):
        pdr = 1 - np.exp(-snr_value)
        pdr = np.nan_to_num(pdr)
        return pdr

    def doppler_shift_effect(self, t, v, d, R):
        c = 3 * 10 ^ 8
        med = R ** 2 - d ** 2
        b = math.sqrt(med)
        b -= v * t
        r = b ** 2 + d ** 2
        cos = b / r
        doppler_effect_num = self.f * (v * cos) / c
        return doppler_effect_num, r

    # interference
    def interference_packet_delivery_rate(self, H, actions):
        # 干扰状态下的信道容量
        actions_vec = actions[:, None]
        t = self.time
        # numerator: diagonal elements (hii)+
        # H_diag：提取矩阵 H 的对角线元素，这些元素表示各个发送器到对应接收器的直接信号增益
        H_diag = np.diag(H)
        num = np.multiply(H_diag, actions)
        doppler_effect_coefficent, _ = self.doppler_shift_effect()
        # denominator: off-diagonal elements 要加上W_ICI
        x_values = np.linspace(-1, 1, 1000)
        integral_result = np.trapz((1 - abs(x_values)) * j0(2 * np.pi * x_values), x_values)
        constant_term = (1 / 12) * (2 * np.pi) ** 2
        result = 1 - integral_result <= constant_term
        # 计算ICI
        W_ICI = result
        H_interference = (H - np.diag(np.diag(H))).transpose()
        # 加上多普勒的影响
        # den：计算分母，即干扰矩阵与动作向量的点积 加上噪声功率 self.sigma ** 2，表示总干扰强度。
        den = (np.dot(H_interference * W_ICI, actions_vec) + self.sigma ** 2).flatten()
        # SNIR
        SINR = num / den
        # pdr：根据 SNIR 计算数据包交付率，使用公式 1 - np.exp(-SNIR)
        pdr = 1 - np.exp(- SINR)
        # 如何利用SINR
        return pdr, SINR


    def greedy_control_aware_base_scheduling(self, control_states_obs_ca_pwr):
        ca_pwr = np.zeros(self.num_users)
        aux_obs = np.reshape(control_states_obs_ca_pwr, (-1, self.p))
        aux_cost = np.multiply(aux_obs, aux_obs)
        aux_norm = np.sqrt(aux_cost.sum(axis=1))
        control_states_norm = aux_norm.reshape(self.n, self.k)
        ind = np.argmax(control_states_norm, axis=1)
        idx_aux = np.arange(0, self.n * self.k, self.k) + ind
        ca_pwr[idx_aux] = 1.

        return ca_pwr

    def greedy_control_aware_scheduling(self, n_transmitting, control_states_obs_ca_pwr):
        ca_pwr = np.zeros(self.num_users)
        aux_obs = np.reshape(control_states_obs_ca_pwr, (-1, self.p))
        aux_cost = np.multiply(aux_obs, aux_obs)
        aux_norm = np.sqrt(aux_cost.sum(axis=1))
        ind = np.argpartition(aux_norm, -n_transmitting)[-n_transmitting:]
        ca_pwr[ind] = 1.

        return ca_pwr

    def random_power(self):
        num_users = self.num_users
        Pmax = self.max_pwr_perplant
        random_powers = np.random.uniform(0, Pmax, num_users)
        return random_powers

    def max_power(self):
        num_users = self.num_users
        Pmax = self.max_pwr_perplant
        max_powers = np.ones(num_users) * Pmax
        return max_powers

    def round_robin(self, n, last_idx):
        transmitting_plants = np.zeros((self.n, self.k))
        if (last_idx + 1) >= self.k:
            transmitting_plants[:, 0] = 1.
            last_idx = 0
        else:
            transmitting_plants[:, last_idx + 1] = 1.
            last_idx += 1
        rr_pwr = transmitting_plants.reshape(-1, self.num_users)

        return rr_pwr, last_idx

    def round_robin_scheduling(self, n, last_idx):
        transmitting_plants = np.zeros(self.num_users)
        if (last_idx + n) >= self.num_users:
            n_under = self.num_users - last_idx
            n_over = last_idx + n - self.num_users
            transmitting_plants[-n_under:] = 1.
            transmitting_plants[:n_over] = 1.
            last_idx = n_over
        else:
            transmitting_plants[last_idx:last_idx + n] = 1.
            last_idx += n
        rr_pwr = transmitting_plants

        return rr_pwr, last_idx

    def wmmse(self, S):
        Pmax = self.p0
        h2 = np.copy(S)
        h = np.sqrt(h2)
        m = S.shape[1]
        N = S.shape[0]
        v = np.ones((N, m)) * np.sqrt(Pmax) / 2
        T = 100
        v2 = np.expand_dims(v ** 2, axis=2)

        u = (np.diagonal(h, axis1=1, axis2=2) * v) / (np.matmul(h2, v2)[:, :, 0] + self.sigma)
        w = 1 / (1 - u * np.diagonal(h, axis1=1, axis2=2) * v)
        N = 1000
        for n in np.arange(T):
            u2 = np.expand_dims(u ** 2, axis=2)
            w2 = np.expand_dims(w, axis=2)
            v = (w * u * np.diagonal(h, axis1=1, axis2=2)) / (np.matmul(np.transpose(h2, (0, 2, 1)), (w2 * u2)))[:, :,
                                                             0]
            v = np.minimum(np.sqrt(Pmax), np.maximum(0, v))
            v2 = np.expand_dims(v ** 2, axis=2)
            u = (np.diagonal(h, axis1=1, axis2=2) * v) / (np.matmul(h2, v2)[:, :, 0] + self.sigma)
            w = 1 / (1 - u * np.diagonal(h, axis1=1, axis2=2) * v)
        p = v ** 2
        return p

    def Hata_PL(self, choice, f, A_b, A_m, t, v, d, R):
        num, r = self.doppler_shift_effect(t, v, d, R)
        delta1_1 = 5.74 * np.log10(A_b) - 30.42
        delta1_2 = -6.72, delta2_1 = -21.42, delta2_2 = -9.62
        # 郊区模型
        model1 = delta1_1 + 26.16 * np.log10(f) - 13.82 * np.log10(A_b) - 3.2 * (np.log10(11.75 * A_m) ** 2) + (
                    44.9 - 6.55 * np.log10(A_b) + delta1_2) * np.log10(r)
        # 高架桥模型
        model2 = delta2_1 + 26.16 * np.log10(f) - 13.82 * np.log10(A_b) - 3.2 * (np.log10(11.75 * A_m) ** 2) + (
                    44.9 - 6.55 * np.log10(A_b) + delta2_2) * np.log10(r)
        if choice == 1:
            PL_ii = model1
        PL_ii = model2
        return PL_ii

    # samples initial state and channel conditions
    def sample(self, batch_size):
        # graph, flat observation
        self.H, samples = self.sample_graph()
        # control states
        samples2 = self.np_random.normal(0, 1., size=self.control_state_dim)
        self.current_control_obs = samples2
        # self.ppp = samples3
        samples3 = samples# sample3 是另外的特征
        return np.hstack((samples, samples2, samples3))

    def sample_graph(self):  # downlink
        mu = self.mu
        PL = mu * self.Hata_PL()
        samples = self.np_random.rayleigh(mu, size=(self.L.shape[0], self.L.shape[1]))
        PP = samples[None, :, :] * self.L
        A = PP[0]
        A[A < 0.001] = 0.0

        # Multi-cell network
        A = build_cell_graph(A, self.assign)

        # Normalizing observations: should we add a flag?
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()
        return A, A_flat

    def sample_graph_uplink(self):  # downlink
        mu = self.mu
        PL = mu * self.Hata_PL()
        samples = self.np_random.rayleigh(mu, size=(self.L.shape[0], self.L.shape[1]))
        PP = samples[None, :, :] * self.L
        A = PP[0]
        A[A < 0.001] = 0.0

        # Multi-cell network
        A = build_cell_graph(A, self.assign)
        A = A.T

        # Normalizing observations: should we add a flag?
        A_normalized = self.normalize_gso(A)
        A_flat = A_normalized.flatten()

        return A, A_flat

    def scale_power(self, power_action):
        power_action = np.clip(power_action, -1., 1.)
        power_action += 1.
        power_action /= 2  # [0, 1.]
        power_action *= self.max_pwr_perplant

        return power_action

    def normalize_scale_power(self, power_action):

        power_action = np.clip(power_action, -1., 1.)
        power_action += 1.
        power_action = power_action / (power_action.sum() + 1e-8)
        power_action *= self.upperbound

        return power_action

    def _reset(self):
        obs = self.sample(batch_size=1)
        self.current_state = obs
        zerovec = np.zeros(self.num_users * self.p)
        self.current_control_obs = (obs[self.channel_state_dim:] + np.random.multivariate_normal(zerovec, self.Wobs))[:,
                                   None]
        current_control_state = obs[self.channel_state_dim:]
        self.time_step = 0

        # to save training data
        if self.cost_hist:
            cost_hist = np.array(self.cost_hist)
            cost_disc = self.disc_cost(cost_hist)
            ep_cost = cost_disc[0]
            self.ep_cost_hist.append(ep_cost)

        if self.Lagrangian_hist:
            cost_hist = np.array(self.Lagrangian_hist)
            cost_disc = self.disc_cost(cost_hist)
            ep_lagrangian = cost_disc[0]
            self.ep_Lagrangian.append(ep_lagrangian)

        if self.constraint_hist:
            constraint_hist = np.array(self.constraint_hist)
            constraint_disc = self.disc_constraint(constraint_hist)
            ep_constraint = constraint_disc[0]
            self.ep_constraint.append(ep_constraint)

        self.current_episode += 1
        self.cost_hist = []
        self.constraint_hist = []
        self.Lagrangian_hist = []

        return obs

    def _update_control_states_downlink(self, downlink_power, H, lambd, action_penalty):
        self.time_step += 1
        zerovec = np.zeros(self.num_users * self.p)
        control_states_obs = self.current_control_obs
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        # cost / reward -> plant states
        control_states = self.current_state[self.channel_state_dim:][None, :]
        cost_aux = np.multiply(control_states, control_states)

        # cost / reward -> control actions
        cost_aux2 = np.multiply(control_actions, control_actions)

        # downlink delivery rate
        qq = self.interference_packet_delivery_rate(H, downlink_power)
        qq = np.nan_to_num(qq)
        trials_aux = np.transpose(bernoulli.rvs(qq))
        trials = np.repeat(trials_aux, self.q, axis=0)[:, None]

        control_estimate = np.multiply(trials, control_actions)
        control_states = control_states.transpose()

        # new control states
        control_states = (np.dot(self.A, control_states) + np.dot(self.B, control_estimate) +
                          np.transpose(self.np_random.multivariate_normal(zerovec, self.W, size=self.batch_size)))


        control_states = np.transpose(control_states)
        control_states = np.clip(control_states, -self.max_control_state, self.max_control_state)
        control_states_obs = control_states + self.np_random.multivariate_normal(zerovec, self.Wobs,
                                                                                 size=self.batch_size)


        control_states_obs = np.clip(control_states_obs, -self.max_control_state, self.max_control_state)
        self.current_control_obs = control_states_obs.T

        # total reward; Q and R are diagonal -> reward should be computed with ''previous'' control state, not current!
        one_step_cost = cost_aux.sum(axis=1)  # objective function
        one_step_cost2 = cost_aux.sum(axis=1) + np.dot(lambd, action_penalty)  # Lagrangian

        self.cost_hist.append(one_step_cost)  # save cost, Lagrangian during training
        self.Lagrangian_hist.append(one_step_cost2)
        self.downlink_constraint_dualvar = np.dot(lambd, action_penalty)

        done = False
        if self.time_step > self.T - 1:
            done = True
            one_step_cost = cost_aux.sum(axis=1)

            # scaling reward
        one_step_reward = -1 * one_step_cost2[0] / (self.num_users * self.max_control_state)
        one_step_reward /= 100  # scaling reward to stabilize training

        return control_states, control_states_obs, one_step_reward, done

    def _update_control_states_uplink(self, control_actions, lambd, action_penalty):
        self.time_step += 1
        zerovec = np.zeros(self.num_users * self.p)

        # cost / reward -> plant states
        control_states = self.current_state[self.channel_state_dim:][None, :]
        cost_aux = np.multiply(control_states, control_states)

        # cost / reward -> control actions
        cost_aux2 = np.multiply(control_actions, control_actions)

        # in this case we assume ideal communications during downlink transmission
        control_estimate = control_actions
        control_states = control_states.transpose()

        # new control states
        control_states = (np.dot(self.A, control_states) + np.dot(self.B, control_estimate) +
                          np.transpose(self.np_random.multivariate_normal(zerovec, self.W, size=self.batch_size)))
        control_states = np.transpose(control_states)
        control_states = np.clip(control_states, -self.max_control_state, self.max_control_state)
        control_states_obs = control_states + self.np_random.multivariate_normal(zerovec, self.Wobs,
                                                                                 size=self.batch_size)
        control_states_obs = np.clip(control_states_obs, -self.max_control_state, self.max_control_state)

        # total reward; Q and R are diagonal -> reward should be computed with ''previous'' control state, not current!
        one_step_cost = cost_aux.sum(axis=1)  # objective function
        one_step_cost2 = cost_aux.sum(axis=1) + np.dot(lambd, action_penalty)

        self.cost_hist.append(one_step_cost)  # save cost, Lagrangian during training
        self.Lagrangian_hist.append(one_step_cost2)

        done = False
        if self.time_step > self.T - 1:
            done = True
            one_step_cost = cost_aux.sum(axis=1)

        one_step_reward = -1 * one_step_cost2[0] / (self.num_users * self.max_control_state)

        return control_states, control_states_obs, one_step_reward, done

    def _update_control_actions_uplink(self, control_states, uplink_power, H):

        # uplink delivery rate
        qq = self.interference_packet_delivery_rate(H, uplink_power.flatten())
        qq = np.nan_to_num(qq)
        trials = np.transpose(bernoulli.rvs(qq))

        # updates state estimate
        control_states_obs = np.multiply(trials, control_states)

        # control actions computed at the remote controller
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        return control_actions

    def _update_control_actions_downlink(self, control_states_obs):

        # control actions computed at the remote controller
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        return control_actions

    def _test_init(self, T, batch_size=1):
        n_comps = 7  # GNN, MLP, Equal, Control-Aware, WMMSE, Round Robin, Random Access

        # cost per time step
        cost_matrices = [np.zeros(T) for _ in np.arange(n_comps)]

        # states
        dnn_state = self.sample(batch_size=1)[np.newaxis]
        init_states = [dnn_state for _ in np.arange(n_comps)]

        # interference matrix
        H, _ = self.sample_graph()
        init_interference = [H for _ in np.arange(n_comps)]

        # saving trajectory
        trajectories = [np.zeros((T, batch_size, self.control_state_dim)) for _ in np.arange(n_comps)]

        # saving power allocation
        allocation_decisions = [np.zeros((T, batch_size, self.num_users)) for _ in np.arange(n_comps)]

        zerovec = np.zeros(self.num_users * self.p)
        channel_states = dnn_state[:, :self.channel_state_dim]
        channel_states_obs = channel_states
        control_states = dnn_state[:, self.channel_state_dim:]
        control_states_obs = control_states + np.random.multivariate_normal(zerovec, self.Wobs, size=self.batch_size)

        control_states_obs_dnn = self.control_plant_norm(control_states_obs)
        dnn_obs = np.hstack((channel_states_obs, control_states_obs))
        observations = [dnn_obs for _ in np.arange(n_comps)]

        # baselines
        eq_power = np.ones(self.num_users)
        eq_power *= self.p0
        last_idx = -1  # round robin

        return (cost_matrices, init_states, init_interference, trajectories, allocation_decisions, observations,
                zerovec, eq_power, last_idx)

    def _get_uplink_obs(self, channel_obs, control_estimates):

        control_estimates_obs = self.control_plant_norm(control_estimates)
        obs = np.hstack((channel_obs, control_estimates_obs))

        return control_estimates, control_estimates_obs, obs

    def _get_downlink_obs(self, channel_obs, control_estimates):

        control_estimates_obs = self.control_plant_norm(control_estimates)
        obs = np.hstack((channel_obs, control_estimates_obs))

        return obs.flatten()

    def _test_step_uplink(self, states, states_obs, H, action, estimator):

        control_states = states[:, self.channel_state_dim:]  # [None, :]

        # uplink delivery rate
        qq = self.interference_packet_delivery_rate(H, action.flatten())
        qq = np.nan_to_num(qq)
        trials = np.transpose(bernoulli.rvs(qq))

        # updates state estimate
        control_states_obs = np.multiply(trials, control_states)

        # control actions computed at the remote controller
        control_actions = np.dot(self.fb_gain, -control_states_obs)

        # cost / reward -> plant states
        cost_aux = np.multiply(control_states, control_states)

        # cost / reward -> control input
        cost_aux2 = np.multiply(control_actions, control_actions)

        # total reward
        one_step_cost = cost_aux.sum(axis=1)

        return one_step_cost, control_states, control_actions

    def _test_step_downlink(self, control_states, control_actions, H, downlink_action, zerovec):

        # downlink delivery rate
        qq = self.interference_packet_delivery_rate(H, downlink_action)
        qq = np.nan_to_num(qq)
        trials_aux = np.transpose(bernoulli.rvs(qq))
        trials = np.repeat(trials_aux, self.q, axis=0)[:, None]

        control_estimate = np.multiply(trials, control_actions)
        control_states = control_states.transpose()

        # new control states
        control_states = (np.dot(self.A, control_states) + np.dot(self.B, control_estimate) +
                          np.transpose(self.np_random.multivariate_normal(zerovec, self.W, size=self.batch_size)))
        control_states = np.transpose(control_states)
        control_states = np.clip(control_states, -self.max_control_state, self.max_control_state)
        control_states_obs = control_states + self.np_random.multivariate_normal(zerovec, self.Wobs,
                                                                                 size=self.batch_size)
        control_states_obs = np.clip(control_states_obs, -self.max_control_state, self.max_control_state)

        # new channel states
        H, channel_states = self.sample_graph_uplink()

        states = np.hstack((channel_states, control_states.flatten()))
        states_obs = np.hstack((channel_states, control_states_obs.flatten()))

        return states, states_obs, H

    # heuristics always satisfy instantaneous power constraints
    def test_equal_power(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        eq_power_downlink = np.ones(self.num_users)
        eq_power_downlink *= (upper_bound / self.num_users)

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, eq_power_downlink,
                                                             zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = eq_power_downlink
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_round_robin_scheduling(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx,
                                    batch_size=1, last_idx=0):
        zerovec = np.zeros(self.num_users * self.p)
        n_transmitting = self.n_transmitting

        for tt in range(T):
            rr_pwr, last_idx = self.round_robin_scheduling(n_transmitting, last_idx)
            rr_pwr /= rr_pwr.sum()
            rr_pwr *= upper_bound
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, rr_pwr.flatten(),
                                                             zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = rr_pwr
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_control_aware_scheduling(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx,
                                      batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)
        n_transmitting = self.n_transmitting

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            downlink_action = (self.greedy_control_aware_scheduling(n_transmitting, control_states_obs)).flatten()
            downlink_action /= downlink_action.sum()
            downlink_action *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, downlink_action,
                                                             zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = downlink_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_wmmse(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx, batch_size=1,
                   last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            downlink_action = (self.wmmse(H[None, :])).flatten()
            downlink_action /= downlink_action.sum()
            downlink_action *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, downlink_action,
                                                             zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = downlink_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_random_access_scheduling(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx,
                                      batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)
        transmitting_plants = np.hstack((np.ones(self.n_transmitting), np.zeros(self.num_users - self.n_transmitting)))

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            ra_pwr = np.random.permutation(transmitting_plants)
            ra_pwr /= ra_pwr.sum()
            ra_pwr *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)
            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, ra_pwr, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = ra_pwr
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_control_aware_base_scheduling(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx,
                                           batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            downlink_action = (self.greedy_control_aware_base_scheduling(control_states_obs)).flatten()
            downlink_action /= downlink_action.sum()
            downlink_action *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, downlink_action,
                                                             zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = downlink_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_round_robin_base_scheduling(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx,
                                         batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)

        for tt in range(T):
            rr_pwr, last_idx = self.round_robin(self.n_transmitting, last_idx)
            rr_pwr /= rr_pwr.sum()
            rr_pwr *= upper_bound
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, rr_pwr.flatten(),
                                                             zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = rr_pwr
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_random_access_base_scheduling(self, upper_bound, T, states, states_obs, states_mtx, power_mtx, H, cost_mtx,
                                           batch_size=1, last_idx=-1):
        zerovec = np.zeros(self.num_users * self.p)

        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            ra_pwr = (np.eye(self.k)[np.random.choice(self.k, self.n)]).reshape(self.k * self.n)
            ra_pwr /= ra_pwr.sum()
            ra_pwr *= upper_bound
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)
            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, ra_pwr, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = ra_pwr
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_mlp_scheduling(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                            cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            dnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_dnn.predict(dnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            power_action = np.clip(allocation_action, 0., 1.)
            power_action /= (power_action.sum() + 1e-8)
            power_action *= upper_bound

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_gnn_scheduling(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                            cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            gnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_gnn.predict(gnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            power_action = np.clip(allocation_action, 0., 1.)
            power_action /= (power_action.sum() + 1e-8)
            power_action *= upper_bound

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_mlp_base_scheduling(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                 cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            dnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_dnn.predict(dnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            power_action = np.zeros((self.n, self.k))
            for jj in np.arange(self.n):
                power_action[jj, allocation_action[jj].astype(int)] = 1.
            power_action = power_action.reshape(self.n * self.k)
            power_action = np.clip(power_action, 0., 1.)
            power_action *= (upper_bound / self.n)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_gnn_base_scheduling(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                 cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            gnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_gnn.predict(gnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            power_action = np.zeros((self.n, self.k))
            for jj in np.arange(self.n):
                power_action[jj, allocation_action[jj].astype(int)] = 1.
            power_action = power_action.reshape(self.n * self.k)
            power_action = np.clip(power_action, 0., 1.)
            power_action *= (upper_bound / self.n)

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_mlp_base_output_constraint(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx,
                                        H, cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            dnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_dnn.predict(dnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()

            allocation_action = self.scale_power(allocation_action)  # power decisions in [0, 1]
            power_aux = allocation_action.reshape(self.n, self.k)
            sum_per_base_station = power_aux.sum(axis=1)[:, None]
            power_aux /= (sum_per_base_station + 1e-8)
            power_aux *= (upper_bound / self.n)
            power_action = power_aux.reshape(-1).flatten()

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_gnn_base_output_constraint(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx,
                                        H, cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            gnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_gnn.predict(gnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()

            allocation_action = self.scale_power(allocation_action)  # power decisions in [0, 1]
            power_aux = allocation_action.reshape(self.n, self.k)
            sum_per_base_station = power_aux.sum(axis=1)[:, None]
            power_aux /= (sum_per_base_station + 1e-8)
            power_aux *= (upper_bound / self.n)
            power_action = power_aux.reshape(-1).flatten()

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_mlp_base_constraint(self, allocation_dnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                 cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        current_budget = np.zeros(self.n)
        overall_constraint = upper_bound * T / self.n
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            dnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_dnn.predict(dnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            allocation_action += 1.
            allocation_action /= 2  # allocation decisions in [0, 1]
            power_action = allocation_action * self.max_pwr_perplant
            power_action_aux = power_action.reshape(self.n, self.k)

            allocation_action_per_base_station = power_action.reshape(self.n, self.k).sum(axis=1)
            for jj in np.arange(self.n):
                if current_budget[jj] + allocation_action_per_base_station[jj] > overall_constraint:
                    remaining_budget = max(overall_constraint - current_budget[jj], 0.)
                    allocation_action_aux = power_action_aux[jj, :]
                    allocation_action_aux /= (allocation_action_aux.sum() + 1e-8)
                    allocation_action_aux *= remaining_budget
                    power_action_aux[jj, :] = allocation_action_aux
                    allocation_action_per_base_station[jj] = allocation_action_aux.sum()
                current_budget[jj] += allocation_action_per_base_station[jj]

            power_action = power_action_aux.reshape(-1).flatten()

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test_gnn_base_constraint(self, allocation_gnn, upper_bound, T, states, states_obs, states_mtx, power_mtx, H,
                                 cost_mtx, batch_size=1):
        zerovec = np.zeros(self.num_users * self.p)
        current_budget = np.zeros(self.n)
        overall_constraint = upper_bound * T / self.n
        for tt in range(T):
            states_mtx[tt, :, :] = states[:, self.channel_state_dim:]
            control_states = states[:, self.channel_state_dim:]
            control_states_obs = states_obs[:, self.channel_state_dim:]
            # control actions computed at the remote controller
            control_actions = np.dot(self.fb_gain, -control_states_obs.T)
            # cost / reward -> plant states
            cost_aux = np.multiply(control_states, control_states)
            # cost / reward -> control input
            cost_aux2 = np.multiply(control_actions, control_actions)
            # total reward
            one_step_cost = cost_aux.sum(axis=1)

            # power decisions
            control_states_obs = states_obs[:, self.channel_state_dim:].flatten()
            channel_states_obs = states_obs[:, :self.channel_state_dim].flatten()
            gnn_obs = self._get_downlink_obs(channel_states_obs, control_states_obs)
            allocation_action, _ = allocation_gnn.predict(gnn_obs, deterministic=True)
            allocation_action = allocation_action.flatten()
            allocation_action += 1.
            allocation_action /= 2  # allocation decisions in [0, 1]
            power_action = allocation_action * self.max_pwr_perplant
            power_action_aux = power_action.reshape(self.n, self.k)

            allocation_action_per_base_station = power_action.reshape(self.n, self.k).sum(axis=1)
            for jj in np.arange(self.n):
                if current_budget[jj] + allocation_action_per_base_station[jj] > overall_constraint:
                    remaining_budget = max(overall_constraint - current_budget[jj], 0.)
                    allocation_action_aux = power_action_aux[jj, :]
                    allocation_action_aux /= (allocation_action_aux.sum() + 1e-8)
                    allocation_action_aux *= remaining_budget
                    power_action_aux[jj, :] = allocation_action_aux
                    allocation_action_per_base_station[jj] = allocation_action_aux.sum()
                current_budget[jj] += allocation_action_per_base_station[jj]

            power_action = power_action_aux.reshape(-1).flatten()

            states, states_obs, H = self._test_step_downlink(control_states, control_actions, H, power_action, zerovec)

            cost_mtx[tt] = one_step_cost
            power_mtx[tt, :, :] = power_action
            states = states[None, :]
            states_obs = states_obs[None, :]

        return (cost_mtx, power_mtx, states_mtx)

    def test(self, upper_bound, T, allocation_dnn, allocation_gnn, batch_size=1, test_type='base_scheduling'):

        (cost_matrices, current_states, interference_matrices, states_matrices, allocation_decisions, observations,
         zerovec, eq_power, last_idx) = \
            self._test_init(T, batch_size=batch_size)

        [dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, capwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx,
         rapwr_cost_mtx] = cost_matrices
        [dnn_state, gnn_state, eq_state, ca_state, wmmse_state, rr_state, ra_state] = current_states
        [dnn_H, gnn_H, eq_H, ca_H, wmmse_H, rr_H, ra_H] = interference_matrices
        [dnn_states, gnn_states, eq_states, ca_states, wmmse_states, rr_states, ra_states] = states_matrices
        [dnn_power, gnn_power, equal_power, ca_power, wmmse_power, rr_power, ra_power] = allocation_decisions
        [dnn_obs, gnn_obs, eq_obs, ca_obs, wmmse_obs, rr_obs, ra_obs] = observations

        # Heuristics
        # Equal power
        eqpwr_cost_mtx, equal_power, eq_states = self.test_equal_power(upper_bound, T, eq_state, eq_obs, eq_states,
                                                                       equal_power, eq_H, eqpwr_cost_mtx)
        # WMMSE
        wmmsepwr_cost_mtx, wmmse_power, wmmse_states = self.test_wmmse(upper_bound, T, wmmse_state, wmmse_obs,
                                                                       wmmse_states,
                                                                       wmmse_power, wmmse_H, wmmsepwr_cost_mtx)
        if test_type == 'base_scheduling':
            # Control-Aware
            capwr_cost_mtx, ca_power, ca_states = self.test_control_aware_base_scheduling(upper_bound, T, ca_state,
                                                                                          ca_obs, ca_states,
                                                                                          ca_power, ca_H,
                                                                                          capwr_cost_mtx)
            # Round Robin
            rrpwr_cost_mtx, rr_power, rr_states = self.test_round_robin_base_scheduling(upper_bound, T, rr_state,
                                                                                        rr_obs, rr_states,
                                                                                        rr_power, rr_H, rrpwr_cost_mtx)
            # Random Access
            rapwr_cost_mtx, ra_power, ra_states = self.test_random_access_base_scheduling(upper_bound, T, ra_state,
                                                                                          ra_obs, ra_states,
                                                                                          ra_power, ra_H,
                                                                                          rapwr_cost_mtx)

            # Learned Policies
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_base_scheduling(allocation_dnn, upper_bound, T,
                                                                                dnn_state, dnn_obs, dnn_states,
                                                                                dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_base_scheduling(allocation_gnn, upper_bound, T,
                                                                                gnn_state, gnn_obs, gnn_states,
                                                                                gnn_power, gnn_H, gnn_cost_mtx)
        elif test_type == 'scheduling':
            # Control-Aware
            capwr_cost_mtx, ca_power, ca_states = self.test_control_aware_scheduling(upper_bound, T, ca_state, ca_obs,
                                                                                     ca_states, ca_power, ca_H,
                                                                                     capwr_cost_mtx)
            # Round Robin
            rrpwr_cost_mtx, rr_power, rr_states = self.test_round_robin_scheduling(upper_bound, T, rr_state, rr_obs,
                                                                                   rr_states, rr_power, rr_H,
                                                                                   rrpwr_cost_mtx)
            # Random Access
            rapwr_cost_mtx, ra_power, ra_states = self.test_random_access_scheduling(upper_bound, T, ra_state, ra_obs,
                                                                                     ra_states, ra_power, ra_H,
                                                                                     rapwr_cost_mtx)

            # Learned Policies
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_scheduling(allocation_dnn, upper_bound, T, dnn_state,
                                                                           dnn_obs, dnn_states,
                                                                           dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_scheduling(allocation_gnn, upper_bound, T, gnn_state,
                                                                           gnn_obs, gnn_states,
                                                                           gnn_power, gnn_H, gnn_cost_mtx)

        elif test_type == 'base_constraint':
            # Does it make more sense to compare the base constraint solutions against scheduling or triggering policies?
            # Compare
            # Control-Aware
            capwr_cost_mtx, ca_power, ca_states = self.test_control_aware_base_scheduling(upper_bound, T, ca_state,
                                                                                          ca_obs, ca_states,
                                                                                          ca_power, ca_H,
                                                                                          capwr_cost_mtx)
            # Round Robin
            rrpwr_cost_mtx, rr_power, rr_states = self.test_round_robin_base_scheduling(upper_bound, T, rr_state,
                                                                                        rr_obs, rr_states,
                                                                                        rr_power, rr_H, rrpwr_cost_mtx)
            # Random Access
            rapwr_cost_mtx, ra_power, ra_states = self.test_random_access_base_scheduling(upper_bound, T, ra_state,
                                                                                          ra_obs, ra_states,
                                                                                          ra_power, ra_H,
                                                                                          rapwr_cost_mtx)
            # Learned Policies
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_base_constraint(allocation_dnn, upper_bound, T,
                                                                                dnn_state,
                                                                                dnn_obs, dnn_states,
                                                                                dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_base_constraint(allocation_gnn, upper_bound, T,
                                                                                gnn_state,
                                                                                gnn_obs, gnn_states,
                                                                                gnn_power, gnn_H, gnn_cost_mtx)

        elif test_type == 'base_output_constraint':
            # Does it make more sense to compare the base constraint solutions against scheduling or triggering policies?
            # Compare
            # Control-Aware
            capwr_cost_mtx, ca_power, ca_states = self.test_control_aware_base_scheduling(upper_bound, T, ca_state,
                                                                                          ca_obs, ca_states,
                                                                                          ca_power, ca_H,
                                                                                          capwr_cost_mtx)
            # Round Robin
            rrpwr_cost_mtx, rr_power, rr_states = self.test_round_robin_base_scheduling(upper_bound, T, rr_state,
                                                                                        rr_obs, rr_states,
                                                                                        rr_power, rr_H, rrpwr_cost_mtx)
            # Random Access
            rapwr_cost_mtx, ra_power, ra_states = self.test_random_access_base_scheduling(upper_bound, T, ra_state,
                                                                                          ra_obs, ra_states,
                                                                                          ra_power, ra_H,
                                                                                          rapwr_cost_mtx)
            # Learned Policies
            # DNN / MLP
            dnn_cost_mtx, dnn_power, dnn_states = self.test_mlp_base_output_constraint(allocation_dnn, upper_bound, T,
                                                                                       dnn_state,
                                                                                       dnn_obs, dnn_states,
                                                                                       dnn_power, dnn_H, dnn_cost_mtx)
            # GNN
            gnn_cost_mtx, gnn_power, gnn_states = self.test_gnn_base_output_constraint(allocation_gnn, upper_bound, T,
                                                                                       gnn_state,
                                                                                       gnn_obs, gnn_states,
                                                                                       gnn_power, gnn_H, gnn_cost_mtx)

        return (
        dnn_cost_mtx, gnn_cost_mtx, eqpwr_cost_mtx, wmmsepwr_cost_mtx, rrpwr_cost_mtx, capwr_cost_mtx, rapwr_cost_mtx,
        dnn_power, gnn_power, equal_power, wmmse_power, rr_power, ca_power, ra_power,
        dnn_states, gnn_states, eq_states, wmmse_states, rr_states, ca_states, ra_states)


# ------------------------------------------- Downlink Environments ------------------------------------------ #

class LQRMultiCellDownlink(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, p=2, q=1, Ao=None,
                 W=None, Wobs=None, Wobs_channels=None, T=40, a0=1.01,
                 gamma=0.99, r=0.001, pl=2., pp=5., p0=1., num_features=1, scaling=False):
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, p=p, q=q, Ao=Ao, W=W, Wobs=Wobs,
                         Wobs_channels=Wobs_channels, T=T, a0=a0, gamma=gamma, r=r, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        # Downlink: continuous allocation decisions
        self.action_space = spaces.Box(low=-np.ones(num_users), high=np.ones(num_users))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        control_obs = self.current_control_obs
        agent_obs = self._get_downlink_obs(channel_obs, control_obs)

        return agent_obs

    def step(self, action):
        # dual variable
        lambd = action[-self.constraint_dim:]

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.scale_power(power_action)

        # constraint violation
        action_penalty = (power_action.sum() - self.upperbound)  # constraint violation
        self.constraint_violation = action_penalty
        self.constraint_hist.append(action_penalty)

        control_states, control_states_obs, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, lambd, action_penalty)

        self.H, channel_states = self.sample_graph()  # new channel states --- downlink

        states = np.hstack((channel_states, control_states.flatten()))
        self.current_state = states
        states_obs = self._get_downlink_obs(channel_states, control_states_obs)

        return states_obs, one_step_reward, bool(done), {}


class LQRMultiCellDownlinkBaseConstraint(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, p=2, q=1, Ao=None,
                 W=None, Wobs=None, Wobs_channels=None, T=40, a0=1.01,
                 gamma=0.99, r=0.001, pl=2., pp=5., p0=1., num_features=1, scaling=False):
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, p=p, q=q, Ao=Ao, W=W, Wobs=Wobs,
                         Wobs_channels=Wobs_channels, T=T, a0=a0, gamma=gamma, r=r, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        # Downlink: continuous allocation decisions
        self.action_space = spaces.Box(low=-np.ones(num_users), high=np.ones(num_users))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        control_obs = self.current_control_obs
        agent_obs = self._get_downlink_obs(channel_obs, control_obs)

        return agent_obs

    def step(self, action):
        # dual variable
        lambd = action[-self.constraint_dim:]

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.scale_power(power_action)

        # constraint violation
        step_constraint = power_action.reshape(self.n, self.k).sum(axis=1)
        action_penalty = step_constraint - (self.upperbound / self.n)
        self.constraint_violation = action_penalty
        self.constraint_hist.append(action_penalty)

        control_states, control_states_obs, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, lambd, action_penalty)

        self.H, channel_states = self.sample_graph()  # new channel states --- downlink

        states = np.hstack((channel_states, control_states.flatten()))
        self.current_state = states
        states_obs = self._get_downlink_obs(channel_states, control_states_obs)

        return states_obs, one_step_reward, bool(done), {}


class LQRMultiCellDownlinkBaseScheduling(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, p=2, q=1, Ao=None,
                 W=None, Wobs=None, Wobs_channels=None, T=40, a0=1.01,
                 gamma=0.99, r=0.001, pl=2., pp=5., p0=1., num_features=1, scaling=False):
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, p=p, q=q, Ao=Ao, W=W, Wobs=Wobs,
                         Wobs_channels=Wobs_channels, T=T, a0=a0, gamma=gamma, r=r, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        self.action_space = spaces.MultiDiscrete(k * np.ones(n))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        control_obs = self.current_control_obs
        agent_obs = self._get_downlink_obs(channel_obs, control_obs)

        return agent_obs

    def step(self, action):
        # downlink power allocation policy
        power_action = np.zeros((self.n, self.k))
        for jj in np.arange(self.n):
            power_action[jj, action[jj].astype(int)] = 1.
        power_action = power_action.reshape(self.n * self.k)
        power_action = np.clip(power_action, 0., 1.)
        power_action *= (self.upperbound / self.n)

        control_states, control_states_obs, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, 0., 0.)

        self.H, channel_states = self.sample_graph()  # new channel states --- downlink

        states = np.hstack((channel_states, control_states.flatten()))
        self.current_state = states
        states_obs = self._get_downlink_obs(channel_states, control_states_obs)

        return states_obs, one_step_reward, bool(done), {}


class LQRMultiCellDownlinkScheduling(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, p=2, q=1, Ao=None,
                 W=None, Wobs=None, Wobs_channels=None, T=40, a0=1.01,
                 gamma=0.99, r=0.001, pl=2., pp=5., p0=1., num_features=1, scaling=False):
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, p=p, q=q, Ao=Ao, W=W, Wobs=Wobs,
                         Wobs_channels=Wobs_channels, T=T, a0=a0, gamma=gamma, r=r, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        self.action_space = spaces.MultiBinary(num_users)
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        control_obs = self.current_control_obs
        agent_obs = self._get_downlink_obs(channel_obs, control_obs)

        return agent_obs

    def step(self, action):
        # downlink power allocation policy
        power_action = np.clip(action, 0., 1.)
        power_action /= (power_action.sum() + 1e-8)
        power_action *= (self.upperbound)

        control_states, control_states_obs, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, 0., 0.)

        self.H, channel_states = self.sample_graph()  # new channel states --- downlink

        states = np.hstack((channel_states, control_states.flatten()))
        self.current_state = states
        states_obs = self._get_downlink_obs(channel_states, control_states_obs)

        return states_obs, one_step_reward, bool(done), {}


class LQRMultiCellDownlinkBaseOutputConstraint(LQR_Env):
    def __init__(self, num_users, upperbound, constraint_dim, L, assign, n, k, mu=1, p=2, q=1, Ao=None,
                 W=None, Wobs=None, Wobs_channels=None, T=40, a0=1.01,
                 gamma=0.99, r=0.001, pl=2., pp=5., p0=1, num_features=1, scaling=False):
        super().__init__(num_users, upperbound, constraint_dim, L, assign, mu=mu, p=p, q=q, Ao=Ao, W=W, Wobs=Wobs,
                         Wobs_channels=Wobs_channels, T=T, a0=a0, gamma=gamma, r=r, pl=pl, pp=pp, p0=p0,
                         num_features=num_features, scaling=scaling)

        # Downlink: continuous allocation decisions
        self.action_space = spaces.Box(low=-np.ones(num_users), high=np.ones(num_users))
        self.n = n
        self.k = k

    def reset(self):
        obs = self._reset()
        channel_obs = obs[:self.channel_state_dim]
        control_obs = self.current_control_obs
        agent_obs = self._get_downlink_obs(channel_obs, control_obs)

        return agent_obs

    def step(self, action):
        lambd = 0.
        action_penalty = 0.

        # downlink power allocation policy
        power_action = np.nan_to_num(action[:self.num_users])
        power_action = self.scale_power(power_action)  # power decisions in [0, 1]

        # constraint violation
        power_aux = power_action.reshape(self.n, self.k)
        sum_per_base_station = power_aux.sum(axis=1)[:, None]
        power_aux /= (sum_per_base_station + 1e-8)
        power_aux *= (self.upperbound / self.n)
        power_action = power_aux.reshape(-1)

        control_states, control_states_obs, one_step_reward, done = \
            self._update_control_states_downlink(power_action, self.H, lambd, action_penalty)

        self.H, channel_states = self.sample_graph()  # new channel states --- downlink

        states = np.hstack((channel_states, control_states.flatten()))
        self.current_state = states
        states_obs = self._get_downlink_obs(channel_states, control_states_obs)

        return states_obs, one_step_reward, bool(done), {}

