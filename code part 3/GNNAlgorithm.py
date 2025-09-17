import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable
import collections
from collections import deque
import copy
from abc import ABC, abstractmethod

import gym
from gym import spaces
import io
import pathlib
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import time

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor, safe_mean
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, \
    save_to_zip_file
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

from stable_baselines3.common.logger import Logger
from stable_baselines3.common import utils
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
)

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq
from stable_baselines3.common.vec_env import VecEnv

import time
from collections import deque
from stable_baselines3.common import utils

from GNNPolicies import GNNPolicy, GNNActorCriticPolicy
from PGBuffer import PGRolloutBuffer
from SubProcEnvMod import SubprocVecEnv


class GNNBaseAlgorithm(object):
    """
    Base RL Algorithm parametrized by a Random Edge Graph Neural Network (REGNN).

    param policy: policy type
    param env: gym environment (vectorized)
    param learning rate: learning rate for the optimizer
    param n_users: number of agents in the network
    param n_features_input: dimension of input feature at each node
    param n_features_action: dimension of output feature at each node
    p
    """

    def __init__(
            self,
            policy: Union[str, Type[GNNPolicy]],
            env: Union[GymEnv, str],
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            gamma: float = 0.99,
            ent_coef: float = 0.0,
            max_grad_norm: float = 0.5,
            batch_size: int = 128,
            n_users: int = 10,
            n_features_input: int = 2,
            # Modified: 3 features per node (traffic_type, current_link_delay, link_satellite_status)
            n_features_action: int = 1,  # Modified: 2 action features per node (ground_power, satellite_request_signal)
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            n_epochs: int = 4,
    ):
        self.device = get_device(device)

        # Algorithm parameters
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.n_steps = n_steps

        # Environment
        self.n_users = n_users
        self.n_features_input = n_features_input
        self.n_features_action = n_features_action
        self.env = None
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self.eval_env = None

        # Policy
        self.policy_class = policy
        self.policy = None

        if verbose > 0:
            print(f"Using {self.device} device")
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # Misc
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.start_time = None
        self._last_obs = None
        self._last_episode_starts = None
        self._last_original_obs = None
        self._episode_num = 0

        # Buffer
        self.rollout_buffer = None

        self._current_progress_remaining = 1
        self.ep_info_buffer = None
        self.ep_success_buffer = None
        self._n_updates = 0
        self._logger = None
        self._custom_logger = False

        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        if _init_setup_model:
            self._setup_model()

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).
        """
        return self.env

    def _setup_model(self) -> None:
        buffer_cls = PGRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.learning_rate,
            self.batch_size,
            self.n_users,
            np.eye(self.n_users),  # GSO parameter
            self.n_features_input,
            self.n_features_action,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self,
            env: SubprocVecEnv,
            callback: BaseCallback,
            rollout_buffer: PGRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, log_probs = self.policy.forward(obs_tensor)
                values = th.zeros(1)
            actions = actions.cpu().numpy()

            # 检查动作空间维度，适配不同的动作空间定义
            if isinstance(self.action_space, gym.spaces.Box):
                if len(self.action_space.shape) == 1:
                    # 对于1D动作空间，直接使用，不需要重塑
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                else:
                    # 对于2D动作空间，重塑为(n_envs, n_users, n_features_action)
                    actions_reshaped_for_clipping = actions.reshape(self.n_envs, self.n_users, self.n_features_action)
                    clipped_actions = np.clip(actions_reshaped_for_clipping, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            # Use original actions (flattened) for rollout_buffer.add
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        rollout_buffer.compute_returns(last_values=values, dones=dones)

        return True

    def set_logger(self, logger: Logger) -> None:
        self._logger = logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        return self._logger

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "device",
            "env",
            "eval_env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def _setup_learn(
            self,
            total_timesteps: int,
            eval_env: Optional[GymEnv],
            callback: MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
    ):
        self.start_time = time.time()

        if self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        if callback is not None:
            callback.init_callback(self)
        else:
            pass

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def train(self) -> None:
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "GNNBaseAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )
        if callback is not None:
            callback.on_training_start(locals(), globals())
        else:
            pass

        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, deterministic)

    @classmethod
    def load(
            cls,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ) -> "GNNBaseAlgorithm":

        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects,
        )

        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy" in params:
            if "log_std" in params["policy"]:
                del params["policy"]["log_std"]
                n_params = len(params["policy.optimizer"]["param_groups"][0]["params"])
                params["policy.optimizer"]["param_groups"][0]["params"] = [i for i in range(n_params - 1)]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            data["observation_space"] = env.observation_space
            data["action_space"] = env.action_space

            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            if force_reset and data is not None:
                data["_last_obs"] = None
        else:
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,
        )

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        model.set_parameters(params, exact_match=True, device=device)

        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is None:
                    continue
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        return model

    def get_parameters(self) -> Dict[str, Dict]:
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            params[name] = attr.state_dict()
        return params

    def set_parameters(
            self,
            load_path_or_dict: Union[str, Dict[str, Dict]],
            exact_match: bool = True,
            device: Union[th.device, str] = "auto",
    ) -> None:
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                attr.load_state_dict(params[name])
            else:
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            var_name = torch_var.split(".")[0]
            exclude.add(var_name)

        for param_name in exclude:
            data.pop(param_name, None)

        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


class GNNActorCriticAlgorithm(GNNBaseAlgorithm):
    """
    Actor Critic RL Algorithm parametrized by a Random Edge Graph Neural Network (REGNN).
    """

    def __init__(
            self,
            policy: Union[str, Type[GNNActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            ent_coef: float = 0.0,
            vf_coef: float = 0.25,
            max_grad_norm: float = 0.5,
            batch_size: int = 128,
            n_users: int = 10,
            n_features_input: int = 4,  # Modified: 3 features per node
            n_features_action: int = 1,  # Modified: 2 action features per node
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            n_epochs: int = 4,
    ):
        super(GNNActorCriticAlgorithm, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            n_users=n_users,
            n_features_input=n_features_input,
            n_features_action=n_features_action,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            n_epochs=n_epochs,
        )

        self.vf_coef = vf_coef
        self.gae_lambda = gae_lambda

    def _setup_model(self) -> None:
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.learning_rate,
            self.batch_size,
            self.n_users,
            np.eye(self.n_users),  # GSO parameter
            self.n_features_input,
            self.n_features_action,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self,
            env: SubprocVecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)

            actions = actions.cpu().numpy()
            
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)


            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            # print(self._last_obs.shape)
            rewards = rewards.flatten()
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            # out_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True