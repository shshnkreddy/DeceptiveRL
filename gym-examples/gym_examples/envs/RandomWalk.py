import gym
from gym import spaces
import numpy as np

class RandomWalkEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,))
        self.scale = 1
        self.reward_range = (0*self.scale, 0.5*self.scale)

    def reset(self):
        self.state = np.random.uniform(0, 1)
        self.steps = 0
        return np.array([self.state])

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:
            move = -np.random.uniform(0, 0.25)
            # move = -0.1
        else:
            move = np.random.uniform(0, 0.25)
            # move = 0.1

        self.state += move
        self.state = np.clip(self.state, 0, 1)
        reward = (0.5 - abs(self.state - 0.5))*self.scale
        done = self.steps == 49
        self.steps += 1
        return np.array([self.state]), reward, done, {}

    def render(self, mode='human'):
        pass
