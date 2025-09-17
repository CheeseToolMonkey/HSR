import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import collections
from collections import deque
import copy
from abc import ABC, abstractmethod

import gym
from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import time

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor

import GNNs.Modules.architectures as archit
import GNNs.Modules.architecturesTime as architTime

from GNNDistributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    if device == "auto":
        device = "cuda"
    device = th.device(device)

    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class GNNPolicy(nn.Module):
    """
    Policy object (policy network only)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr: float,
            batch_size: int,
            n_users: int,
            GSO,
            n_features_input: int,  # Modified: 3 features per node
            n_features_action: int,  # Modified: 2 action features per node
            net_arch: Optional[Dict] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            log_std_init: float = -1.3,
            full_std: bool = True,
            squash_output: bool = False,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            K1: int = 5,
            bias: bool = True,
            gnn_layers: Optional[List[int]] = 3 * [5],
            gnn_feats: Optional[List[int]] = 3 * [10],
            device: Union[th.device, str] = "auto",
    ):
        super(GNNPolicy, self).__init__()

        self.device = get_device(device)
        self.n_features_input = n_features_input
        self.n_features_actions = n_features_action
        self.n_agents = n_users
        self.graph_dim = n_users ** 2
        self.batch_size = batch_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.squash_output = squash_output
        self.action_dim = self.n_agents * self.n_features_actions
        self.activation_fn = activation_fn

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        beta1 = 0.9
        beta2 = 0.999
        self.beta1 = beta1
        self.beta2 = beta2

        if net_arch is None:
            net_arch = {}
            net_arch['name'] = 'LocalGNN'
            net_arch['archit'] = architTime.LocalGNN_B
            net_arch['device'] = self.device
            net_arch['dimNodeSignals'] = [n_features_input] + gnn_feats + [n_features_action]
            net_arch['nFilterTaps'] = [K1] + gnn_layers
            net_arch['bias'] = bias
            net_arch['nonlinearity'] = activation_fn
            net_arch['dimReadout'] = []
            net_arch['dimEdgeFeatures'] = 1

        self.net_arch = net_arch
        self.log_std_init = log_std_init
        self.policy_net = self.archit_builder(net_arch, GSO)

        self.action_dist = make_proba_distribution(action_space, n_users=n_users, n_features_action=n_features_action)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.log_std = self.action_dist.make_log_std(
                log_std_init=self.log_std_init
            )
            self.log_std = self.log_std.to(self.device)

        self.trainer_lr = lr
        self.optimizer = self.trainer_builder(self.policy_net, beta1, beta2, optimizer_class)

    def archit_builder(self, net_archit, S):
        pol_archit = net_archit['archit'](
            net_archit['dimNodeSignals'],
            net_archit['nFilterTaps'],
            net_archit['bias'],
            net_archit['nonlinearity'],
            net_archit['dimReadout'],
            net_archit['dimEdgeFeatures'])

        self.n_params = 0
        for param in list(pol_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        return pol_archit

    def trainer_builder(self, grnn_archit, beta1, beta2, optimizer_class):
        trainer = optimizer_class(self.parameters(), self.trainer_lr, betas=(beta1, beta2))
        return trainer

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        obs = obs.float()
        latent_pi = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, log_prob

    def _get_latent(self, obs: th.Tensor) -> th.Tensor:
        current_batch_size = obs.shape[0]
        
        # 检查观察空间是否是HSR的复杂结构
        if obs.shape[1] > self.graph_dim + self.n_features_input * self.n_agents:
            # HSR环境：复杂的邻居关系结构
            channel_obs, gnn_input = self._process_hsr_observation(obs, current_batch_size)
        else:
            # 标准环境：简单的[channel_matrix, node_features]结构
            channel_obs = obs[:, :self.graph_dim]
            # gnn_input is (batch_size, n_features_input, n_agents)
            gnn_input = obs[:, self.graph_dim:].reshape(current_batch_size, self.n_features_input, self.n_agents)
        
        S_t = channel_obs.reshape(current_batch_size, self.n_agents, self.n_agents)
        latent_pi = self.policy_net(gnn_input, S_t).reshape(-1, self.action_dim)
        return latent_pi

    def _process_hsr_observation(self, obs: th.Tensor, current_batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        处理HSR环境的复杂观察结构（GNNPolicy版本）
        HSR观察包含：本地信息 + 干扰邻居信息 + 被干扰邻居信息
        
        Args:
            obs: HSR观察张量 (batch_size, local_state_dim)
            current_batch_size: 批次大小
            
        Returns:
            channel_obs: 信道观察矩阵 (batch_size, graph_dim)
            gnn_input: GNN输入特征 (batch_size, n_features_input, n_agents)
        """
        # HSR状态结构分析（基于LQREnvs_HSR.py的设计）
        # 假设max_neighbors=5, num_users=30, num_features=6
        local_info_dim = self.n_agents * 6  # 每个用户6个本地特征
        max_neighbors = 5  # 从HSR环境获取
        interference_neighbors_dim = max_neighbors * self.n_agents * 3
        interfered_neighbors_dim = max_neighbors * self.n_agents * 3
        
        # 验证观察维度
        expected_dim = local_info_dim + interference_neighbors_dim + interfered_neighbors_dim
        if obs.shape[1] != expected_dim:
            # 如果维度不匹配，尝试动态计算max_neighbors
            remaining_dim = obs.shape[1] - local_info_dim
            estimated_max_neighbors = int(remaining_dim / (2 * self.n_agents * 3))
            max_neighbors = max(1, min(estimated_max_neighbors, 10))  # 限制在合理范围内
        
        # 提取各部分信息
        local_info = obs[:, :local_info_dim]  # 本地信息
        start_idx = local_info_dim
        end_idx = start_idx + max_neighbors * self.n_agents * 3
        interference_info = obs[:, start_idx:end_idx]  # 干扰邻居信息
        interfered_info = obs[:, end_idx:end_idx + max_neighbors * self.n_agents * 3]  # 被干扰邻居信息
        
        # 1. 构造图邻接矩阵（信道矩阵）
        # 从本地信息中提取信道增益信息来构造图结构
        local_features = local_info.reshape(current_batch_size, self.n_agents, 6)
        
        # 使用本地信道增益构造对角矩阵作为基础
        channel_gains = local_features[:, :, 2:4].mean(dim=2)  # 使用历史信道增益的平均值
        
        # 构造对称的信道矩阵
        channel_obs = th.zeros(current_batch_size, self.graph_dim, device=obs.device)
        for i in range(self.n_agents):
            # 对角线元素：自身信道增益
            channel_obs[:, i * self.n_agents + i] = channel_gains[:, i]
            
            # 非对角线元素：基于邻居关系
            for j in range(self.n_agents):
                if i != j:
                    # 使用简化的干扰关系，基于信道增益差异
                    interference_weight = th.exp(-th.abs(channel_gains[:, i] - channel_gains[:, j]))
                    channel_obs[:, i * self.n_agents + j] = interference_weight * 0.1
        
        # 2. 构造节点特征矩阵 
        # 将复杂的邻居信息压缩为每个节点的标准特征
        node_features = th.zeros(current_batch_size, self.n_features_input, self.n_agents, device=obs.device)
        
        # 重新排列本地信息以适配GNN输入格式
        local_reshaped = local_features.permute(0, 2, 1)  # (batch, features, agents)
        
        # 根据n_features_input选择相应的特征
        feature_count = min(self.n_features_input, local_reshaped.shape[1])
        node_features[:, :feature_count, :] = local_reshaped[:, :feature_count, :]
        
        # 如果需要更多特征，从邻居信息中提取聚合特征
        if self.n_features_input > feature_count:
            # 聚合干扰邻居信息
            interference_reshaped = interference_info.reshape(current_batch_size, self.n_agents, max_neighbors, 3)
            interference_aggregated = interference_reshaped.mean(dim=2)  # 平均聚合
            
            remaining_features = min(self.n_features_input - feature_count, 3)
            if remaining_features > 0:
                interference_features = interference_aggregated[:, :, :remaining_features].permute(0, 2, 1)
                node_features[:, feature_count:feature_count+remaining_features, :] = interference_features
        
        return channel_obs, node_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = latent_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        observation = observation.float()
        latent_pi = self._get_latent(observation)  # Changed from _get_latent_actor for GNNPolicy
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent_pi = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        return log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        latent_pi = self._get_latent(obs)  # Changed from _get_latent_actor for GNNPolicy
        return self._get_action_dist_from_latent(latent_pi)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_features_input=self.n_features_input,
            n_features_action=self.n_features_actions,
            n_users=self.n_agents,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            log_std_init=self.log_std_init,
            squash_output=self.squash_output,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    def save(self, path: str) -> None:
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> "GNNPolicy":
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)
        model = cls(**saved_variables["data"])
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            observation = copy.deepcopy(observation)
            for key, obs_val in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_val = maybe_transpose(obs_val, obs_space)
                else:
                    obs_val = np.array(obs_val)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_val, obs_space)
                observation[key] = obs_val.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions[0]

        return actions, 0

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def reset_logstd(self):
        action_dim = self.n_features_actions * self.n_agents
        self.log_std = None
        self.log_std = th.ones(self.action_dim).to(self.device) * self.log_std_init


class GNNActorCriticPolicy(nn.Module):
    """
    Policy object (policy network only)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr: float,
            batch_size: int,
            n_users: int,
            GSO,
            n_features_input: int,  # Modified: 3 features per node
            n_features_action: int,  # Modified: 2 action features per node
            net_arch: Optional[Dict] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            log_std_init: float = 0.0,
            full_std: bool = True,
            squash_output: bool = False,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            K1: int = 5,
            bias: bool = True,
            gnn_layers: Optional[List[int]] = 3 * [5],
            gnn_feats: Optional[List[int]] = 3 * [10],
            device: Union[th.device, str] = "auto",
    ):
        super(GNNActorCriticPolicy, self).__init__()

        self.device = get_device(device)
        self.n_features_input = n_features_input
        self.n_features_actions = n_features_action
        self.n_agents = n_users
        self.graph_dim = n_users ** 2
        self.batch_size = batch_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.squash_output = squash_output
        self.action_dim = self.n_agents * self.n_features_actions
        self.activation_fn = activation_fn

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        beta1 = 0.9
        beta2 = 0.999
        self.beta1 = beta1
        self.beta2 = beta2

        if net_arch is None:
            net_arch = {}
            net_arch['name'] = 'LocalGNN'
            net_arch['archit'] = architTime.LocalGNN_B
            net_arch['architvf'] = architTime.SelectionGNN_B
            net_arch['device'] = self.device
            net_arch['dimNodeSignals'] = [n_features_input] + gnn_feats + [n_features_action]
            net_arch['dimNodeSignalsQ'] = [n_features_input] + gnn_feats + [1]
            net_arch['nFilterTaps'] = [K1] + gnn_layers
            net_arch['bias'] = bias
            net_arch['nonlinearity'] = activation_fn
            net_arch['dimReadout'] = []
            net_arch['dimReadoutVF'] = [n_users, 1]
            net_arch['dimEdgeFeatures'] = 1

        self.net_arch = net_arch
        self.log_std_init = log_std_init
        self.policy_net, self.value_net = self.archit_builder(net_arch, GSO)

        self.action_dist = make_proba_distribution(action_space, n_users=n_users, n_features_action=n_features_action)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.log_std = self.action_dist.make_log_std(
                log_std_init=self.log_std_init
            )
            self.log_std = self.log_std.to(self.device)

        self.trainer_lr = lr
        self.optimizer = self.trainer_builder(self.policy_net, beta1, beta2, optimizer_class)

    def archit_builder(self, net_archit, S):
        pol_archit = net_archit['archit'](
            net_archit['dimNodeSignals'],
            net_archit['nFilterTaps'],
            net_archit['bias'],
            net_archit['nonlinearity'],
            net_archit['dimReadout'],
            net_archit['dimEdgeFeatures'])

        value_archit = net_archit['architvf'](
            net_archit['dimNodeSignalsQ'],
            net_archit['nFilterTaps'],
            net_archit['bias'],
            net_archit['nonlinearity'],
            net_archit['dimReadoutVF'],
            net_archit['dimEdgeFeatures'])

        self.n_params = 0
        for param in list(pol_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        for param in list(value_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        return pol_archit, value_archit

    def trainer_builder(self, grnn_archit, beta1, beta2, optimizer_class):
        trainer = optimizer_class(self.parameters(), self.trainer_lr, betas=(beta1, beta2))
        return trainer

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = obs.float()
        latent_pi, values = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        current_batch_size = obs.shape[0]
        
        # 检查观察空间是否是HSR的复杂结构
        if obs.shape[1] > self.graph_dim + self.n_features_input * self.n_agents:
            # HSR环境：复杂的邻居关系结构
            channel_obs, gnn_input = self._process_hsr_observation(obs, current_batch_size)
        else:
            # 标准环境：简单的[channel_matrix, node_features]结构
            channel_obs = obs[:, :self.graph_dim]
            # gnn_input is (batch_size, n_features_input, n_agents)
            gnn_input = obs[:, self.graph_dim:].reshape(current_batch_size, self.n_features_input, self.n_agents)

        S_t = channel_obs.reshape(current_batch_size, self.n_agents, self.n_agents)

        latent_pi = self.policy_net(gnn_input, S_t).reshape(-1, self.action_dim)
        vf = self.value_net(gnn_input, S_t).reshape(-1, 1)

        return latent_pi, vf

    def _process_hsr_observation(self, obs: th.Tensor, current_batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        处理HSR环境的复杂观察结构
        HSR观察包含：本地信息 + 干扰邻居信息 + 被干扰邻居信息
        
        Args:
            obs: HSR观察张量 (batch_size, local_state_dim)
            current_batch_size: 批次大小
            
        Returns:
            channel_obs: 信道观察矩阵 (batch_size, graph_dim)
            gnn_input: GNN输入特征 (batch_size, n_features_input, n_agents)
        """
        # HSR状态结构分析（基于LQREnvs_HSR.py的设计）
        # 假设max_neighbors=5, num_users=30, num_features=6
        local_info_dim = self.n_agents * 6  # 每个用户6个本地特征
        max_neighbors = 5  # 从HSR环境获取
        interference_neighbors_dim = max_neighbors * self.n_agents * 3
        interfered_neighbors_dim = max_neighbors * self.n_agents * 3
        
        # 验证观察维度
        expected_dim = local_info_dim + interference_neighbors_dim + interfered_neighbors_dim
        if obs.shape[1] != expected_dim:
            # 如果维度不匹配，尝试动态计算max_neighbors
            remaining_dim = obs.shape[1] - local_info_dim
            estimated_max_neighbors = int(remaining_dim / (2 * self.n_agents * 3))
            max_neighbors = max(1, min(estimated_max_neighbors, 10))  # 限制在合理范围内
        
        # 提取各部分信息
        local_info = obs[:, :local_info_dim]  # 本地信息
        start_idx = local_info_dim
        end_idx = start_idx + max_neighbors * self.n_agents * 3
        interference_info = obs[:, start_idx:end_idx]  # 干扰邻居信息
        interfered_info = obs[:, end_idx:end_idx + max_neighbors * self.n_agents * 3]  # 被干扰邻居信息
        
        # 1. 构造图邻接矩阵（信道矩阵）
        # 从本地信息中提取信道增益信息来构造图结构
        local_features = local_info.reshape(current_batch_size, self.n_agents, 6)
        
        # 使用本地信道增益构造对角矩阵作为基础
        channel_gains = local_features[:, :, 2:4].mean(dim=2)  # 使用历史信道增益的平均值
        
        # 构造对称的信道矩阵
        channel_obs = th.zeros(current_batch_size, self.graph_dim, device=obs.device)
        for i in range(self.n_agents):
            # 对角线元素：自身信道增益
            channel_obs[:, i * self.n_agents + i] = channel_gains[:, i]
            
            # 非对角线元素：基于邻居关系
            for j in range(self.n_agents):
                if i != j:
                    # 使用简化的干扰关系，基于信道增益差异
                    interference_weight = th.exp(-th.abs(channel_gains[:, i] - channel_gains[:, j]))
                    channel_obs[:, i * self.n_agents + j] = interference_weight * 0.1
        
        # 2. 构造节点特征矩阵 
        # 将复杂的邻居信息压缩为每个节点的标准特征
        node_features = th.zeros(current_batch_size, self.n_features_input, self.n_agents, device=obs.device)
        
        # 重新排列本地信息以适配GNN输入格式
        local_reshaped = local_features.permute(0, 2, 1)  # (batch, features, agents)
        
        # 根据n_features_input选择相应的特征
        feature_count = min(self.n_features_input, local_reshaped.shape[1])
        node_features[:, :feature_count, :] = local_reshaped[:, :feature_count, :]
        
        # 如果需要更多特征，从邻居信息中提取聚合特征
        if self.n_features_input > feature_count:
            # 聚合干扰邻居信息
            interference_reshaped = interference_info.reshape(current_batch_size, self.n_agents, max_neighbors, 3)
            interference_aggregated = interference_reshaped.mean(dim=2)  # 平均聚合
            
            remaining_features = min(self.n_features_input - feature_count, 3)
            if remaining_features > 0:
                interference_features = interference_aggregated[:, :, :remaining_features].permute(0, 2, 1)
                node_features[:, feature_count:feature_count+remaining_features, :] = interference_features
        
        return channel_obs, node_features

    def _get_latent_actor(self, obs: th.Tensor) -> th.Tensor:
        current_batch_size = obs.shape[0]
        
        # 检查观察空间是否是HSR的复杂结构
        if obs.shape[1] > self.graph_dim + self.n_features_input * self.n_agents:
            # HSR环境：复杂的邻居关系结构
            channel_obs, gnn_input = self._process_hsr_observation(obs, current_batch_size)
        else:
            # 标准环境：简单的[channel_matrix, node_features]结构
            channel_obs = obs[:, :self.graph_dim]
            # gnn_input is (batch_size, n_features_input, n_agents)
            gnn_input = obs[:, self.graph_dim:].reshape(-1, self.n_features_input, self.n_agents)
        
        S_t = channel_obs.reshape(-1, self.n_agents, self.n_agents)
        latent_pi = self.policy_net(gnn_input, S_t).reshape(-1, self.action_dim)

        return latent_pi

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = latent_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        observation = observation.float()
        latent_pi = self._get_latent_actor(observation)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, values = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        latent_pi = self._get_latent_actor(obs)
        return self._get_action_dist_from_latent(latent_pi)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_features_input=self.n_features_input,
            n_features_action=self.n_features_actions,
            n_users=self.n_agents,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            log_std_init=self.log_std_init,
            squash_output=self.squash_output,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    def save(self, path: str) -> None:
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> "GNNPolicy":
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)
        model = cls(**saved_variables["data"])
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            observation = copy.deepcopy(observation)
            for key, obs_val in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_val = maybe_transpose(obs_val, obs_space)
                else:
                    obs_val = np.array(obs_val)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_val, obs_space)
                observation[key] = obs_val.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions[0]

        return actions, 0

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def reset_logstd(self):
        action_dim = self.n_features_actions * self.n_agents
        self.log_std = None
        self.log_std = nn.Parameter(th.ones(action_dim).to(self.device) * self.log_std_init, requires_grad=True)