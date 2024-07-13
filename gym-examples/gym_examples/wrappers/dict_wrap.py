import gym 

class dict_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, inp):
        return inp['obs']