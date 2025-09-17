import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import time
# th.set_default_dtype(th.float32)

from stable_baselines3.common.type_aliases import GymEnv,MaybeCallback

# Base Algorithm
from GNNAlgorithm import GNNBaseAlgorithm
from GNNPolicies import GNNPolicy


class Reinforce(GNNBaseAlgorithm):
    """
    Reinforce Policy Gradient parametrized by a Random Edge Graph Neural Network (REGNN).

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
        batch_size: int = 128,
        n_users: int = 10,
        n_features_input: int = 2,
        n_features_action: int = 1,
        n_epochs: int = 1,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(Reinforce, self).__init__(
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
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            create_eval_env=create_eval_env,
            _init_setup_model=False,
            n_epochs=n_epochs,
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(Reinforce, self)._setup_model()


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        entropy_losses = []
        pg_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            batch_size = self.rollout_buffer.n_envs
            for rollout_data in self.rollout_buffer.get(batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                # REINFORCE loss
                returns = rollout_data.returns - rollout_data.returns.mean()
                # returns = rollout_data.returns
                policy_loss = (-returns*log_prob).mean()

                # Logging
                pg_losses.append(policy_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Adding an entropy bonus for now
                loss = policy_loss # + self.ent_coef * entropy_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                if self.max_grad_norm is not None:
                    params = list(self.policy.parameters()) # + list(self.policy.policy_net.parameters())
                    th.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_steps

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,  # Add callback argument
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "REINFORCE",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "Reinforce":

        return super(Reinforce, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,  # Pass callback
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
