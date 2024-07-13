import abc
import dataclasses
import logging
from typing import Callable, Iterable, Iterator, Mapping, Optional, Type, overload

import numpy as np
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import base_class, on_policy_algorithm, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util

class IQLearn(base.DemonstrationAlgorithm[types.Transitions]):
    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        soft_policy_algo: base_class.BaseAlgorithm,
        demo_minibatch_size: Optional[int] = None,
        train_timesteps: int = 10000,
        log_dir: types.AnyPath = "output/",
        # replay_buffer_capacity: Optional[int] = None,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        allow_variable_horizon: bool = False,
    ):

        self.demo_batch_size = demo_batch_size
        self.demo_minibatch_size = demo_minibatch_size or demo_batch_size
        if self.demo_batch_size % self.demo_minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self._demo_data_loader = None
        self.endless_expert_iterator = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self.soft_policy_algo = soft_policy_algo
        self.soft_policy_algo.set_expert_iterator(self.endless_expert_iterator)
        self.soft_policy_algo.set_expert_batch_size(self.demo_batch_size, self.demo_minibatch_size)

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.soft_policy_algo = soft_policy_algo
        self._log_dir = util.parse_path(log_dir)

        self._init_tensorboard = init_tensorboard
        if self._init_tensorboard:
            logging.info(f"building summary directory at {self._log_dir}")
            summary_dir = self._log_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(str(summary_dir))

        self.venv_buffering = wrappers.BufferingWrapper(self.venv)

        self.soft_policy_algo.set_env(self.venv)
        self.soft_policy_algo.set_logger(self.logger)

        self.train_timesteps = train_timesteps

        # if replay_buffer_capacity is None:
        #     replay_buffer_capacity = self.train_timesteps
        # self._replay_buffer = buffer.ReplayBuffer(
        #     replay_buffer_capacity,
        #     self.venv,
        # )

    @property
    def policy(self) -> policies.BasePolicy:
        policy = self.soft_policy_algo.policy
        assert policy is not None
        return policy

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
        )
        self.endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.soft_policy_algo.learn(total_timesteps, callback)



    # def _next_expert_batch(self) -> Mapping:
    #     assert self._endless_expert_iterator is not None
    #     return next(self._endless_expert_iterator)

    # def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
    #     if ndarray is not None:
    #         return th.as_tensor(ndarray, device=self.reward_train.device)
    #     return None

    