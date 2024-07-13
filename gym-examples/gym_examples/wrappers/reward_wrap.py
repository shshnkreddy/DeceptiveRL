import gym

class reward_wrapper(gym.Wrapper):
    def __init__(self, env, lambda_=1, R_aug=None):
        super(reward_wrapper, self).__init__(env)
        self.R_aug = R_aug
        self.prev_obs = None
        self.lambda_ = lambda_

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        return self.prev_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.lambda_*reward
        if self.R_aug is not None:
            reward += self.R_aug(self.prev_obs, action)
        self.prev_obs = obs
        return obs, reward, done, info