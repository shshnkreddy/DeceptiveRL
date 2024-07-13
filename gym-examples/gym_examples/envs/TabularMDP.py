# import gym 
# from gym import spaces
from seals.base_envs import TabularModelPOMDP as TabularModelEnv
import numpy as np

# class RLEnv(TabularModelEnv):
#     def __init__(
#         self,
#         n_states: int,
#         n_actions: int,
#         horizon: int,
#         P,
#         R,
#         initial_state_dist, 
#         observation_matrix
#     ):
#         """
#         Args:
#             n_states: Number of states.
#             n_actions: Number of actions.
#             horizon: The horizon of the MDP, i.e. the episode length.
#             P: transition matrix. (S,A,S)
#             R: Reward Function (1-D)
#         """
#         super().__init__()
#         # this generator is ONLY for constructing the MDP, not for controlling
#         # random outcomes during rollouts
#         if(P.shape == (n_actions, n_states, n_states)):
#             self._transition_matrix = np.transpose(P, (1, 0, 2))
#         else:
#             self._transition_matrix = P

#         self._initial_state_dist = initial_state_dist
#         self._horizon = horizon

#         assert len(R.shape) == 1, 'Reward Matrix must be 1-D'
#         self._reward_matrix = R

#         self._observation_matrix = observation_matrix

#     @property
#     def observation_space(self):
#         return self.pomdp_observation_space

#     @property
#     def state_space(self):
#         return self.pomdp_state_space


#     @property
#     def observation_matrix(self):
#         return self._observation_matrix

#     @property
#     def transition_matrix(self):
#         return self._transition_matrix

#     @property
#     def reward_matrix(self):
#         return self._reward_matrix

#     @property
#     def initial_state_dist(self):
#         return self._initial_state_dist

#     @property
#     def horizon(self):
#         return self._horizon

class RLEnv(TabularModelEnv):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        horizon: int,
        P,
        R,
        initial_state_dist, 
        observation_matrix
    ):
        """
        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            horizon: The horizon of the MDP, i.e. the episode length.
            P: transition matrix. (S,A,S)
            R: Reward Function (1-D)
        """
        
        # this generator is ONLY for constructing the MDP, not for controlling
        # random outcomes during rollouts
        if(P.shape == (n_actions, n_states, n_states)):
            P = np.transpose(P, (1, 0, 2))

        # assert len(R.shape) == 1, 'Reward Matrix must be 1-D'

        super().__init__(transition_matrix=P, observation_matrix=observation_matrix, reward_matrix=R, horizon=horizon, initial_state_dist=initial_state_dist)
        self.n_states = n_states
        self.n_actions = n_actions
        # @property
        # def observation_space(self):
        #     return self.pomdp_observation_space

        # @property
        # def state_space(self):
        #     return self.pomdp_state_space


        # @property
        # def observation_matrix(self):
        #     return self._observation_matrix

        # @property
        # def transition_matrix(self):
        #     return self._transition_matrix

        # @property
        # def reward_matrix(self):
        #     return self._reward_matrix

        # @property
        # def initial_state_dist(self):
        #     return self._initial_state_dist

        # @property
        # def horizon(self):
        #     return self._horizon



