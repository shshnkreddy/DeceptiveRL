import gym 

class hidden_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, inp):
        return inp['state']