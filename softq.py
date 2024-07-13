import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Iterator
import dataclasses
from imitation.data import types

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

from stable_baselines3 import DQN
from imitation.algorithms import base

class SoftQ(DQN):
    """Soft Q-Learning.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        alpha: float = 0.05,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.alpha = alpha

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Soft Q update
                next_v_values = self._get_v_values(next_q_values)
                # Avoid potential broadcast issue
                next_v_values = next_v_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _get_v_values(self, q_values):
        return self.alpha*th.logsumexp(q_values/self.alpha, dim=1, keepdim=True)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        observation = th.tensor(observation, device=self.device, dtype=th.float32)
        with th.no_grad():
            q_values = self.q_net.forward(observation)
            v_values = self._get_v_values(q_values)
            action_probs = th.exp((q_values - v_values)/self.alpha)
            actions = th.multinomial(action_probs, num_samples=1).squeeze(1).cpu().numpy()

        return actions, state


class SoftQ_IQ(SoftQ):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        expert_batch_size: int = 32,
        expert_batch_iterator: Optional[Iterator[base.TransitionMapping]]=None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True
        
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.expert_batch_iterator = expert_batch_iterator
        self.expert_batch_size = expert_batch_size

        assert self.expert_batch_size == self.batch_size, "expert batch size must be equal to batch size"

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.device)
        return None
        
    def set_expert_iterator(self, expert_batch_iterator: Iterator[base.TransitionMapping]) -> None:
        self.expert_batch_iterator = expert_batch_iterator

    def set_expert_batch_size(self, expert_batch_size: int, expert_mini_batch_size: int) -> None:
        self.expert_batch_size = expert_batch_size
        self.expert_mini_batch_size = expert_mini_batch_size

    #Code from https://github.com/Div99/IQ-Learn
    def iq_loss(self, current_Q, current_v, next_v, batch, div='chi', loss_type='value_expert', grad_pen=False, regularize=False):
        gamma = self.gamma
        obs, next_obs, action, done, is_expert = batch

        loss_dict = {}
        # keep track of value of initial states
        # v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
        # loss_dict['v0'] = v0.item()

        #  calculate 1st term for IQ loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        y = (1 - done) * gamma * next_v
        reward = (current_Q - y)[is_expert]

        with th.no_grad():
            # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
            if div == "hellinger":
                phi_grad = 1/(1+reward)**2
            elif div == "kl":
                # original dual form for kl divergence (sub optimal)
                phi_grad = th.exp(-reward-1)
            elif div == "kl2":
                # biased dual form for kl divergence
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif div == "kl_fix":
                # our proposed unbiased form for fixing kl divergence
                phi_grad = th.exp(-reward)
            elif div == "js":
                # jensen–shannon
                phi_grad = th.exp(-reward)/(2 - th.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        loss_dict['softq_loss'] = loss.item()

        # calculate 2nd term for IQ loss, we show different sampling strategies
        if loss_type == "value_expert":
            # sample using only expert states (works offline)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (current_v - y)[is_expert].mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif loss_type == "value":
            # sample using expert and policy states (works online)
            # E_(ρ)[V(s) - γV(s')]
            value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        # elif loss_type == "v0":
        #     # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        #     # (1-γ)E_(ρ0)[V(s0)]
        #     v0_loss = (1 - gamma) * v0
        #     loss += v0_loss
        #     loss_dict['v0_loss'] = v0_loss.item()

        # alternative sampling strategies for the sake of completeness but are usually suboptimal in practice
        # elif args.method.loss == "value_policy":
        #     # sample using only policy states
        #     # E_(ρ)[V(s) - γV(s')]
        #     value_loss = (current_v - y)[~is_expert].mean()
        #     loss += value_loss
        #     loss_dict['value_policy_loss'] = value_loss.item()

        # elif args.method.loss == "value_mix":
        #     # sample by weighted combination of expert and policy states
        #     # E_(ρ)[Q(s,a) - γV(s')]
        #     w = args.method.mix_coeff
        #     value_loss = (w * (current_v - y)[is_expert] +
        #                   (1-w) * (current_v - y)[~is_expert]).mean()
        #     loss += value_loss
        #     loss_dict['value_loss'] = value_loss.item()

        else:
            raise ValueError(f'This sampling method is not implemented: {loss_type}')

        #TODO: add grad penalty term
        # if grad_pen:
        #     # add a gradient penalty to loss (Wasserstein_1 metric)
        #     gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
        #                                         action[is_expert.squeeze(1), ...],
        #                                         obs[~is_expert.squeeze(1), ...],
        #                                         action[~is_expert.squeeze(1), ...],
        #                                         lambda_gp)
        #     loss_dict['gp_loss'] = gp_loss.item()
        #     loss += gp_loss

        if div == "chi":  
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
            y = (1 - done) * gamma * next_v

            reward = current_Q - y
            chi2_loss = 1/(4 * self.alpha) * (reward**2)[is_expert].mean()
            loss += chi2_loss
            loss_dict['chi2_loss'] = chi2_loss.item()

        if regularize:
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
            y = (1 - done) * gamma * next_v

            reward = current_Q - y
            chi2_loss = 1/(4 * self.alpha) * (reward**2).mean()
            loss += chi2_loss
            loss_dict['regularize_loss'] = chi2_loss.item()

        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict


    def _setup_train_batch(self, replay_data, expert_data):
        expert_samples = dict(expert_data)
        gen_samples = dict(replay_data)

        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].cpu().detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)
        # Check dimensions.
        assert self.batch_size == len(expert_samples["obs"])
        assert self.batch_size == len(expert_samples["acts"])
        assert self.batch_size == len(expert_samples["next_obs"])
        assert self.batch_size == len(expert_samples["dones"])
        assert self.batch_size == len(gen_samples["obs"])
        assert self.batch_size == len(gen_samples["acts"])
        assert self.batch_size == len(gen_samples["next_obs"])
        assert self.batch_size == len(gen_samples["dones"])

        expert_samples["acts"] = np.expand_dims(expert_samples["acts"], axis=1)
        expert_samples["dones"] = np.expand_dims(expert_samples["dones"], axis=1)

        # print('Observation shape: ', expert_samples["obs"].shape, gen_samples["obs"].shape)
        # print('Action shape: ', expert_samples["acts"].shape, gen_samples["acts"].shape)
        # print('Next observation shape: ', expert_samples["next_obs"].shape, gen_samples["next_obs"].shape)
        # print('Done shape: ', expert_samples["dones"].shape, gen_samples["dones"].shape)


        for start in range(0, self.expert_batch_size, self.expert_mini_batch_size):
            end = start + self.expert_mini_batch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            # Concatenate rollouts, and label each row as expert or generator.
            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(self.expert_mini_batch_size, dtype=int),
                    np.zeros(self.expert_mini_batch_size, dtype=int),
                ],
            )

        # obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
        # acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
        # next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
        # dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
        # # notice that the labels use the convention that expert samples are
        # # labelled with 1 and generator samples with 0.
        # labels_expert_is_one = np.concatenate(
        #     [
        #         np.ones(self.expert_mini_batch_size, dtype=int),
        #         np.zeros(self.expert_mini_batch_size, dtype=int),
        #     ],
        # )

        return obs, acts, next_obs, dones, labels_expert_is_one

    def get_rewards(self, obs, acts, next_obs, dones):
        #TODO
        pass
        # # Compute rewards.
        # #R(s,a,s) = Q(s,a) - γ*V(s')
        # obs = th.tensor(obs, dtype=th.float32, device=self.device)
        # acts = th.tensor(acts, dtype=th.float32, device=self.device)
        # next_obs = th.tensor(next_obs, dtype=th.float32, device=self.device)
        # dones = th.tensor(dones, dtype=th.float32, device=self.device)

        # print('Observation shape: ', obs.shape)
        # print('Action shape: ', acts.shape)

        # with th.no_grad():
        #     q_values = self.q_net.forward(obs).gather(1, acts.long())
        #     next_q_values = self.q_net.forward(next_obs)
        #     next_v_values = self._get_v_values(next_q_values)
        #     rewards = q_values - (1 - dones)*self.gamma*next_v_values
        
        # return rewards

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            replay_dict = {
                "obs": replay_data.observations,
                "acts": replay_data.actions,
                "next_obs": replay_data.next_observations,
                "dones": replay_data.dones,
            }
            
            assert self.expert_batch_iterator is not None, 'Expert batch iterator is None'
            expert_dict = next(self.expert_batch_iterator)

            obs, acts, next_obs, dones, labels_expert_is_one = self._setup_train_batch(replay_dict, expert_dict)
            obs = self._torchify_array(obs)
            acts = self._torchify_array(acts)
            next_obs = self._torchify_array(next_obs)
            dones = self._torchify_array(dones)
            labels_expert_is_one = self._torchify_array(labels_expert_is_one)
            train_batch = (obs, next_obs, acts, dones, labels_expert_is_one)

            #Get next Q-values estimates
            next_q_values = self.q_net(next_obs)
            next_v_values = self._get_v_values(next_q_values)
            next_v_values = next_v_values.reshape(-1, 1)

            # Get current Q-values estimates
            current_q_values = self.q_net(obs)
            #Get curretn V-values estimates
            current_v_values = self._get_v_values(current_q_values)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=acts.long())

            loss, _ = self.iq_loss(current_q_values, current_v_values, next_v_values, train_batch)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
