from gym.envs.registration import register

register(
    id='gym_examples/TabMDP-v0',
    entry_point='gym_examples.envs:RLEnv',
)

register(
    id='gym_examples/RandomWalk-v0',
    entry_point='gym_examples.envs:RandomWalkEnv',
)
