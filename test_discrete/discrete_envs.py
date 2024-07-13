#import from parent directory
import sys
sys.path.append('..')
import mdptoolbox.example
from temporary_seed import temporary_seed
import numpy as np
import gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class discrete_env:
    def __init__(self, name, s_size, a_size, gamma=None, seed=0, P=None, R=None, alpha=None,  horizon=None):
        self.name = name
        self.seed = seed
        self.P = P
        self.R = R
        self.alpha = alpha
        self.gamma = gamma
        self.horizon = horizon
        self.s_size = s_size
        self.a_size = a_size
        
    def get_env(self):
        return self.s_size, self.a_size, self.P, self.R, self.alpha, self.gamma, self.horizon

class CyberBattle(discrete_env):

    def __init__(self, version, n_seeds_per_version=25, n_security_levels=2, seed=0, w1=0.8, w2=0.0, w3=0.2, noise=0.3, horizon=20, gamma=0.99):
        with temporary_seed(seed):
            config_data = np.load('../cybersim/config_data.npy').reshape(25, 10, 3)
            np.random.shuffle(config_data)
            config_data = config_data[:n_seeds_per_version]
            config_data = config_data.reshape(-1, n_seeds_per_version, 3)

            netw_configs = config_data[version]
            # states s, s+1 are the same network configuration with different security levels
            n_states = netw_configs.shape[0]*n_security_levels
            n_actions = netw_configs.shape[1]*n_security_levels
            P = np.ones((n_states, n_actions, n_states))*(noise/n_states)
            for s in range(n_states):
                for a in range(n_actions):
                    P[s, a, a] = (1-noise)/2
                    if(a%2==0):
                        P[s, a, a+1] = (1-noise)/2
                    else:
                        P[s, a, a-1] = (1-noise)/2
                
                    #make them sum up to 1, currently sum up to 99.99
                    P[s, a, np.random.randint(0, n_states)] += 1 - np.sum(P[s, a, :])
                

            R = np.zeros(n_states)
            R_matrix = netw_configs/np.linalg.norm(netw_configs, axis=0, keepdims=True)
            
            for s in range(0, n_states, n_security_levels):
                R[s+1] = np.dot(R_matrix[s//2], [w1, w2, w3])
                R[s] = np.dot(R_matrix[s//2], [w1, w2, w3]) - np.random.random()/5

            alpha = np.ones(n_states)/n_states
        
        super().__init__('cyber_battle', seed=seed, s_size=n_states, a_size=n_actions, P=P, R=R, alpha=alpha, gamma=gamma, horizon=horizon)


class RandomMDP(discrete_env):
    def __init__(self, seed=0, horizon=100, gamma=0.9, s_size=None, a_size=None):
        
        with temporary_seed(seed):
            if s_size is None:
                s_size = np.random.randint(28, 41)
            if a_size is None:
                a_size = np.random.randint(2, 16)
            
            P, R = mdptoolbox.example.rand(s_size, a_size)

            R = np.zeros(s_size)
            for s_ in range(s_size):
                R[s_] = 2*np.random.random() - 1

            alpha = np.zeros(s_size)
            alpha[0] = 1

        super().__init__('random', seed=seed, s_size=s_size, a_size=a_size, P=P, R=R, alpha=alpha, gamma=gamma, horizon=horizon)

class Classic2D(discrete_env):
    def __init__(self, seed=0, grid_size=4, horizon=150, gamma=0.9):
        s_size = grid_size**2
        a_size = 4

        alpha = np.ones(s_size)/s_size
        R = np.zeros(s_size)
        with temporary_seed(seed):
           for i in range(s_size):
                R[i] = np.random.rand()

        #0 -> up, 1 -> right, 2 -> down, 3 -> left
        P = np.zeros((s_size, a_size, s_size))
        for i in range(grid_size):
            for j in range(grid_size):
                curr_state = i*grid_size + j
                for a in range(a_size):
                    next_state = curr_state
                    if(a==0):
                        if(i!=0):
                            next_state = (i-1)*grid_size + j
                    elif(a==1):
                        if(j!=grid_size-1):
                            next_state = i*grid_size + j+1
                    elif(a==2):
                        if(i!=grid_size-1):
                            next_state = (i+1)*grid_size + j
                    elif(a==3):
                        if(j!=0):
                            next_state = i*grid_size + j-1

                    # print(i, j, a, next_state)
                    P[curr_state, a, next_state] = 1

        super().__init__('classic_2D', s_size, a_size, gamma, seed, P, R, alpha, horizon)
        

class FrozenLake(discrete_env):
    def __init__(self, seed=0, grid_size=4, horizon=150, gamma=0.9):
        
        map_desc = generate_random_map(size=grid_size, seed=seed)
        env = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=True) 
        s_size = grid_size**2
        a_size = 4

        alpha = np.ones(s_size)
        R = np.zeros(s_size)
        with temporary_seed(seed):
            for i in range(grid_size):
                for j in range(grid_size):
                    if(map_desc[i][j] == 'H'):
                        R[i*grid_size + j] = -1
                        alpha[i*grid_size + j] = 0
                    else:
                        R[i*grid_size + j] = np.random.rand()
        # R[-1] = 1

        print(R.reshape((grid_size, grid_size)))

        print(map_desc)

        P = np.zeros((s_size, a_size, s_size))
        P_env = env.env.P
        for s in range(s_size):
            for a in range(a_size):
                for i in P_env[s][a]:
                    P[s, a, i[1]] += i[0]

        #Last state is not absorbing 
        #Left
        P = self._make_last_state_non_absorbing(P, grid_size, s_size)

        P = np.swapaxes(P, 0, 1)

        alpha /= np.sum(alpha)
        
        super().__init__('Frozen_Lake', s_size, a_size, gamma, seed, P, R, alpha, horizon)

    def _make_last_state_non_absorbing(self, P, grid_size, s_size):
        i = grid_size - 1
        j = grid_size - 1

        P[s_size-1, :, :] = 0
        #Left 
        P[s_size-1, 0, i*grid_size + (j-1)] += 1/3
        P[s_size-1, 0, (i-1)*grid_size + j] += 1/3
        P[s_size-1, 0, (i)*grid_size + j] += 1/3

        #Down
        P[s_size-1, 1, (i)*grid_size + j] += 1/3
        P[s_size-1, 1, (i)*grid_size + j] += 1/3
        P[s_size-1, 1, (i)*grid_size + j-1] += 1/3

        #Right
        P[s_size-1, 2, i*grid_size + (j)] += 1/3
        P[s_size-1, 2, (i-1)*grid_size + j] += 1/3
        P[s_size-1, 2, (i)*grid_size + (j)] += 1/3

        #Up
        P[s_size-1, 3, (i-1)*grid_size + j] += 1/3
        P[s_size-1, 3, (i)*grid_size + j] += 1/3
        P[s_size-1, 3, (i)*grid_size + j-1] += 1/3

        return P

class FourRooms(discrete_env):
    def __init__(self, *, seed=0, room_size=3, n_doors=1, r_low=-1, r_high=1, r_std=0.3, transition_noise=0.3, horizon=500, gamma=0.99):
        assert n_doors <= room_size, 'n_doors must be less than or equal to room_size'
        
        self.room_size = room_size
        s_size = (2*self.room_size+1)**2
        a_size = 4
        #0 -> up, 1 -> right, 2 -> down, 3 -> left

        super().__init__('four_rooms', seed=seed, s_size=s_size, a_size=a_size, gamma=gamma, horizon=horizon)

       
        vert_doors = np.concatenate([self._random_without_replacement(0, self.room_size, n_doors, seed=seed), self._random_without_replacement(self.room_size+1, 2*self.room_size+1, n_doors, seed=seed+1)])
        
        #ensure different set of doors are picked for horizontal and vertical

        hor_doors = np.concatenate([self._random_without_replacement(0, self.room_size, n_doors, seed=seed+2), self._random_without_replacement(self.room_size+1, 2*self.room_size+1, n_doors, seed=seed+3)])
            
        print('Vertical doors: ', vert_doors)
        print('Horizontal doors: ', hor_doors)

        wall = np.zeros((2*self.room_size+1, 2*self.room_size+1), dtype=int) 
        wall[self.room_size][:] = 1
        wall.T[self.room_size][:] = 1

        wall[self.room_size][hor_doors] = 0
        wall.T[self.room_size][vert_doors] = 0

        self.P = self._get_transition_matrix(wall, transition_noise)
        self.R = self._get_reward_matrix(wall, r_low, r_high, r_std)
        self.alpha = self._get_init_state(wall)

        self.visualize_env(self.room_size, wall)

    def _random_without_replacement(self, low, high, n, step_size=1, seed=None):
        if(seed is None):
            seed = self.seed

        with temporary_seed(seed):
            nums = np.arange(low, high, step=step_size)
            return np.random.choice(nums, size=n, replace=False)

    def _get_transition_matrix(self, wall, transition_noise):
        P = np.zeros((self.s_size, self.a_size, self.s_size))

        # print('Wall: \n', wall)

        for i in range(2*self.room_size+1):
            for j in range(2*self.room_size+1):
                for a in range(self.a_size):
                    if(a==0): #up
                        cur_state = i*(2*self.room_size+1) + j
                        #Ensure it is within bounds
                        i_next = np.clip(i-1, 0, 2*self.room_size)
                        #Check if it is a wall
                        if(wall[i_next, j]):
                            i_next = i
                        next_state = i_next*(2*self.room_size+1) + j
                        
                        #Randomly move in a perpendicular direction
                        j_next = [np.clip(j-1, 0, 2*self.room_size), np.clip(j+1, 0, 2*self.room_size)]
                        perp_next_states = []
                        #Check if it is a wall
                        if(wall[i, j_next[0]]):
                            j_next[0] = j
                        if(wall[i, j_next[1]]):
                            j_next[1] = j
                        perp_next_states.append(i*(2*self.room_size+1) + j_next[0])
                        perp_next_states.append(i*(2*self.room_size+1) + j_next[1])

                        #Assign probabilities
                        P[cur_state, a, next_state] += 1 - transition_noise
                        P[cur_state, a, perp_next_states[0]] += transition_noise/2
                        P[cur_state, a, perp_next_states[1]] += transition_noise/2

                    elif(a==1): #right
                        cur_state = i*(2*self.room_size+1) + j
                        #Ensure it is within bounds
                        j_next = np.clip(j+1, 0, 2*self.room_size)
                        #Check if it is a wall
                        if(wall[i, j_next]):
                            j_next = j
                        next_state = i*(2*self.room_size+1) + j_next

                        #Randomly move in a perpendicular direction
                        i_next = [np.clip(i-1, 0, 2*self.room_size), np.clip(i+1, 0, 2*self.room_size)]
                        perp_next_states = []
                        #Check if it is a wall
                        if(wall[i_next[0], j]):
                            i_next[0] = i
                        if(wall[i_next[1], j]):
                            i_next[1] = i
                        perp_next_states.append(i_next[0]*(2*self.room_size+1) + j)
                        perp_next_states.append(i_next[1]*(2*self.room_size+1) + j)

                        #Assign probabilities
                        P[cur_state, a, next_state] += 1 - transition_noise
                        P[cur_state, a, perp_next_states[0]] += transition_noise/2
                        P[cur_state, a, perp_next_states[1]] += transition_noise/2

                    elif(a==2): #down
                        cur_state = i*(2*self.room_size+1) + j
                        #Ensure it is within bounds
                        i_next = np.clip(i+1, 0, 2*self.room_size)
                        #Check if it is a wall
                        if(wall[i_next, j]):
                            i_next = i
                        next_state = i_next*(2*self.room_size+1) + j

                        #Randomly move in a perpendicular direction
                        j_next = [np.clip(j-1, 0, 2*self.room_size), np.clip(j+1, 0, 2*self.room_size)]
                        perp_next_states = []
                        #Check if it is a wall
                        if(wall[i, j_next[0]]):
                            j_next[0] = j
                        if(wall[i, j_next[1]]):
                            j_next[1] = j
                        perp_next_states.append(i*(2*self.room_size+1) + j_next[0])
                        perp_next_states.append(i*(2*self.room_size+1) + j_next[1])

                        #Assign probabilities
                        P[cur_state, a, next_state] += 1 - transition_noise
                        P[cur_state, a, perp_next_states[0]] += transition_noise/2
                        P[cur_state, a, perp_next_states[1]] += transition_noise/2

                    elif(a==3): #left
                        cur_state = i*(2*self.room_size+1) + j
                        #Ensure it is within bounds
                        j_next = np.clip(j-1, 0, 2*self.room_size)
                        #Check if it is a wall
                        if(wall[i, j_next]):
                            j_next = j
                        next_state = i*(2*self.room_size+1) + j_next

                        #Randomly move in a perpendicular direction
                        i_next = [np.clip(i-1, 0, 2*self.room_size), np.clip(i+1, 0, 2*self.room_size)]
                        perp_next_states = []
                        #Check if it is a wall
                        if(wall[i_next[0], j]):
                            i_next[0] = i
                        if(wall[i_next[1], j]):
                            i_next[1] = i
                        perp_next_states.append(i_next[0]*(2*self.room_size+1) + j)
                        perp_next_states.append(i_next[1]*(2*self.room_size+1) + j)

                        #Assign probabilities
                        P[cur_state, a, next_state] += 1 - transition_noise
                        P[cur_state, a, perp_next_states[0]] += transition_noise/2
                        P[cur_state, a, perp_next_states[1]] += transition_noise/2

            
        return P

    def _get_reward_matrix(self, wall, r_low, r_high, r_std):
        R = np.zeros(self.s_size)

        with temporary_seed(self.seed):
            room_rewards = self._random_without_replacement(r_low, r_high, 4, step_size=0.1)

            for i in range(2*self.room_size+1):
                for j in range(2*self.room_size+1):
                    if(wall[i, j] == 1):
                        R[i*(2*self.room_size+1) + j] = 0
                        continue
                
                    if(i < self.room_size):
                        if(j < self.room_size):
                            R[i*(2*self.room_size+1) + j] = np.random.normal(room_rewards[0], r_std)
                        
                        elif(j > self.room_size):
                            R[i*(2*self.room_size+1) + j] = np.random.normal(room_rewards[1], r_std)

                    elif(i > self.room_size):
                        if(j < self.room_size):
                            R[i*(2*self.room_size+1) + j] = np.random.normal(room_rewards[2], r_std)
                        
                        elif(j > self.room_size):
                            R[i*(2*self.room_size+1) + j] = np.random.normal(room_rewards[3], r_std)

        return R

    def _get_init_state(self, wall):
        alpha = np.ones(self.s_size)*np.logical_not(wall.flatten())
        alpha /= np.sum(alpha)
        return alpha
        

    def visualize_env(self, room_size, wall):
        print('Visualizing the environment:')
        print('Raw Map:')
        # Print the top row of the environment
        print("W "+"W " * (2*room_size+1)+"W ")
        for i in range(0, 2*room_size+1):
            print("W ", end='')
            for j in range(0, 2*room_size+1):
                if(wall[i, j] == 1):
                    print("W ", end='')
                else: 
                    print("0 ", end='')
            print("W")

        print("W "+"W " * (2*room_size+1)+"W")

        print('Rewards: ')
        print(self.R.reshape(2*self.room_size+1, 2*self.room_size+1))

# class DeepSeaTreasure(discrete_env):
#     def __init__(self, seed=0, gamma=0.99, horizon=100):
#         self.rows = 10
#         self.cols = 11
#         s_size = self.rows * self.cols
#         a_size = 4
#         super().__init__('four_rooms', seed=seed, s_size=s_size, a_size=a_size, gamma=gamma, horizon=horizon)
        
#         self.a_size = 4
#         self.treasures = [(1, 7, 10), (4, 3, 15), (4, 9, 20), (7, 2, 25), (8, 6, 5)]
#         self.rewards = np.zeros((self.s_size, self.a_size, 2))
#         self.transitions = np.zeros((self.s_size, self.a_size, self.s_size))
        
#         for treasure in self.treasures:
#             state = self._state_index(treasure[0], treasure[1])
#             self.rewards[state, :, 1] = treasure[2]
            
#         for state in range(self.s_size):
#             row, col = self._state_position(state)
#             for action in range(self.a_size):
#                 next_state, prob = self._next_state_prob(state, action)
#                 self.transitions[state, action, next_state] = prob
#                 self.rewards[state, action, 0] = -1
                
#     def _state_index(self, row, col):
#         return (row-1) * self.cols + (col-1)
    
#     def _state_position(self, index):
#         row = index // self.cols + 1
#         col = index % self.cols + 1
#         return row, col
    
#     def _next_state_prob(self, state, action):
#         row, col = self._state_position(state)
#         if action == 0:  # move left
#             col = max(col - 1, 1)
#         elif action == 1:  # move right
#             col = min(col + 1, self.cols)
#         elif action == 2:  # move up
#             row = max(row - 1, 1)
#         else:  # move down
#             row = min(row + 1, self.rows)
#         next_state = self._state_index(row, col)
#         if next_state == state:
#             return next_state, 1.0
#         else:
#             return next_state, 0.0
            
#     def get_transition_probabilities(self):
#         return self.transitions
    
#     def get_reward_function(self):
#         return self.rewards

