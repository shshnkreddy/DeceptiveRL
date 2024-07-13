import sys
sys.path.append('../')
from discrete_envs import *
import gym
import numpy as np 
from imitation.algorithms.mce_irl import MCEIRL
from policy_randomization import *
from utils import *
from featurize import *
from imitation.rewards import reward_nets
from networks import *
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from seals import base_envs
import os
# from softq import SoftQ_IQ
from callbacks import *
# from iqlearn import IQLearn
from dqnfn import DQNFN
from callbacks import *
from stable_baselines3.common.monitor import Monitor

#Device is cuda if availabe else cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Env Configs 
env_name = 'FrozenLake_10x10'
room_size = 6
grid_size = 3
seeds = [4, 5]

#Policy Configs
lr_lambda = 0.5
n_iter = 750

#DP configs
dp_algo = 'qfn'
sigmas = [0.25, 0.5, 0.75, 1.0]

#Log path 
path_log = 'dp_results/'
exp_name = env_name+'_'+dp_algo+'/'

#Create log directory if it does not exist
path = path_log+exp_name
if not os.path.exists(path):
    os.makedirs(path)

#IRL configs
n_epochs_mce = int(1e3)

def main():
    for seed in seeds:
        print('Seed', seed)
        print()
        path_reward_functions = path+'reward_functions/'
        path_occ_measures = path+'occ_measures/'
        path_eval = path+'eval/'
        path_model = path+'model/'

        path_reward_functions = path_reward_functions+'seed_'+str(seed)+'/'
        path_occ_measures = path_occ_measures+'seed_'+str(seed)+'/'
        path_eval = path_eval+'seed_'+str(seed)+'/'
        path_model = path_model+'seed_'+str(seed)+'/'

        #Create these directories if they don't exist
        if not os.path.exists(path_reward_functions):
            os.makedirs(path_reward_functions)
        if not os.path.exists(path_occ_measures):
            os.makedirs(path_occ_measures)
        if not os.path.exists(path_eval):
            os.makedirs(path_eval)       

        #Setting up envs #TODO: Make it easier to change MDPs
        if(env_name == 'random'):
            env_ = RandomMDP(seed=seed, a_size=4)
        elif('FourRooms' in env_name):
            with temporary_seed(seed):
                n_doors = np.random.randint(1, room_size)
            env_ = FourRooms(seed=seed, room_size=room_size, n_doors=n_doors)
        elif('FrozenLake' in env_name):
            env_ = FrozenLake(seed=seed, grid_size=grid_size)
        
        s_size, a_size, P, R, alpha, gamma, horizon = env_.get_env()
        #Save reward function at path_reward_functions wthout info about seed
        np.save(path_reward_functions+'reward_function', R)

        observation_matrix = np.vstack([featurize(s, s_size) for s in range(s_size)])
        env = gym.make('gym_examples/TabMDP-v0', n_states=s_size, n_actions=a_size, horizon=horizon, P=P, R=R, initial_state_dist=alpha, observation_matrix=observation_matrix)

        kwargs = {'n_states':s_size, 'n_actions':a_size, 'horizon':horizon, 'P':P, 'R':R, 'initial_state_dist':alpha, 'observation_matrix':observation_matrix}
        obs_to_state_map = {}
        for i in range(observation_matrix.shape[0]):
            obs_to_state_map[str(observation_matrix[i])] = i

        n_envs = 1
        extended_eval_callback = ExtendedEvalCallback(Monitor(env), callback_on_new_best=None, eval_freq=max(int(1e3)//n_envs, 1), verbose=0)
        callback_list = [extended_eval_callback, InfoCallback(freq=int(1e4)//n_envs)]
        hidden_to_state_dict = {}
        for i in range(observation_matrix.shape[0]):
            hidden_to_state_dict[str(observation_matrix[i])] = i

        with temporary_seed(seed):
            rng = np.random.default_rng()

        venv1 = make_vec_env(
            'gym_examples/TabMDP-v0',
            n_envs=8,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], 
            env_make_kwargs=kwargs,
            rng=rng,
        )

        venv2 = make_vec_env(
            'gym_examples/TabMDP-v0',
            n_envs=8,
            post_wrappers=[lambda env, _: base_envs.ExposePOMDPStateWrapper(env), lambda env, _: RolloutInfoWrapper(env)], 
            env_make_kwargs=kwargs,
            rng=rng,
        )

        #Optimal Policy
        _, _, policy = soft_vi(env=env, lambda_=250, discount=gamma)
        _, occ_measure_star, occ_measure2d_star = get_occupancy_measures(env=env,pi=policy,discount=gamma)
        E_star = occ_measure_star.T@R
        print('E_star', E_star)
        #save occ_measure_2d_star at path_occ_measures without info about seed 
        np.save(path_occ_measures+'occ_measure_2d_star', occ_measure2d_star)
        
        if('FrozenLake' in env_name):
            E_hat = 0
        else:
            pi_hat = np.ones((s_size, a_size))/a_size
            _, occ_measure_hat, occ_measure_hat2d = get_occupancy_measures(env=env,pi=pi_hat,discount=gamma)
            E_hat = occ_measure_hat.T@R
            np.save(path_occ_measures+'occ_measure_hat2d', occ_measure_hat2d)
        print('E_hat', E_hat)

        pi_star_stoch, _, occ_measure_stoch2d_star = MaxEnt(env, alpha, gamma, E_star, method='true', lr = lr_lambda, n_iter=2000)
        #save occ_measure_stoch2d_star at path_occ_measures without info about seed 
        np.save(path_occ_measures+'occ_measure_stoch2d_star', occ_measure_stoch2d_star)

        for sigma in sigmas:
            print('\nsigma:', sigma, flush=True)
            # print()
            #add info about sigma to all paths
            path_eval = path_eval+'sigma_'+str(sigma)+'/'
            path_model = path_model+'sigma_'+str(sigma)+'/'
            path_occ_measures = path_occ_measures+'sigma_'+str(sigma)+'/'
            path_reward_functions = path_reward_functions+'sigma_'+str(sigma)+'/'
            if not os.path.exists(path_eval):
                os.makedirs(path_eval)
            if not os.path.exists(path_model):
                os.makedirs(path_model)
            if not os.path.exists(path_occ_measures):
                os.makedirs(path_occ_measures)
            if not os.path.exists(path_reward_functions):
                os.makedirs(path_reward_functions)


            learner = DQNFN("MlpPolicy", env, a_size, sigma=sigma, obs_to_state_map=hidden_to_state_dict)
            #check if file exists
            if(os.path.exists(path_model+'model_DQNFN.zip')):
                learner.set_parameters(path_model+'model_DQNFN.zip')
            else:
                learner.learn(total_timesteps=int(2*1e5), callback=callback_list)
                learner.save(path_model+'model_DQNFN')

            actions = learner.predict(observation_matrix, deterministic=True)[0]
            del learner

            pi_dp = np.zeros((s_size, a_size), dtype=np.float32)
            for i in range(s_size):
                pi_dp[i][actions[i]] = 1.0
            
            _, occ_measure_dp, occ_measure_dp2d = get_occupancy_measures(env=env,pi=pi_dp,discount=gamma)
            np.save(path_occ_measures+'occ_measure_dp2d', occ_measure_dp2d)
            E_dp = occ_measure_dp.T@R
            # E_dp = max(E_dp, E_hat)            
            print('E_dp',E_dp)
            # E_dp = (E_dp-E_hat)/(E_star-E_hat)
            # exp_reward, _ = evaluate_policy(learner, env, n_eval_episodes=10)
            # print('exp_reward', exp_reward)
            print('Mean Rewards:', extended_eval_callback.mean_rewards)
            np.save(path_eval+'mean_rewards', extended_eval_callback.mean_rewards)
            np.save(path_eval+'E_dp', [(E_dp-E_hat)/(E_star-E_hat)])

            # continue

            #MCE IRL for DP
            reward_net = reward_nets.BasicRewardNet(
                env.observation_space,
                env.action_space,
                hid_sizes=[256],
                use_action=False,
                use_done=False,
                use_next_state=False,
            )

            mce_irl = MCEIRL(
                occ_measure_dp,
                env,
                reward_net,
                log_interval=250,
                discount=gamma,
                optimizer_kwargs={"lr": 0.01},
                rng=rng,
            )
            occ_measure, R_rec = mce_irl.train(max_iter=n_epochs_mce)
            #save R_rec at path_reward_functions without info about randomization rate and beta and info about mce
            np.save(path_reward_functions+'R_rec_mce_dp', R_rec)
            
            #optimal policy in R_rec
            Q, V, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_rec)
            _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
            #save occ_measure2d at path_occ_measures without info about randomization rate and beta and info about mce
            np.save(path_occ_measures+'occ_measure2d_mce_dp', occ_measure2d)
            E_rec = occ_measure.T@R
            # E_rec = max(E_rec, E_hat)
            print('E_rec_mce_dp', E_rec)
            eval_dp = (E_rec-E_hat)/(E_star-E_hat)
            np.save(path_eval+'eval_dp', [eval_dp])

            #KL
            pi_beta, occ_measure2d, R_new = Det_KL(env, P, R, s_size, a_size, alpha, gamma, E_dp, pi_star_stoch, n_iter=n_iter, lr=lr_lambda)
            occ_measure_kl = np.sum(occ_measure2d, axis=1)
            np.save(path_occ_measures+'occ_measure_kl2d', occ_measure2d)
            #MCE IRL for KL
            reward_net = reward_nets.BasicRewardNet(
                env.observation_space,
                env.action_space,
                hid_sizes=[256],
                use_action=False,
                use_done=False,
                use_next_state=False,
            )

            mce_irl = MCEIRL(
                occ_measure_kl,
                env,
                reward_net,
                log_interval=250,
                discount=gamma,
                optimizer_kwargs={"lr": 0.01},
                rng=rng,
            )
            occ_measure_mce, R_rec = mce_irl.train(max_iter=n_epochs_mce)
            #save R_rec at path_reward_functions without info about randomization rate and beta and info about mce
            np.save(path_reward_functions+'R_rec_mce_kl', R_rec)
            #optimal policy in R_rec
            Q, V, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_rec)
            _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
            #save occ_measure2d at path_occ_measures without info about randomization rate and beta and info about mce
            np.save(path_occ_measures+'occ_measure2d_mce_kl', occ_measure2d)
            E_rec = occ_measure.T@R
            # E_rec = max(E_rec, E_hat)
            print('E_rec_mce_kl', E_rec)
            eval_kl=(E_rec-E_hat)/(E_star-E_hat)
            np.save(path_eval+'eval_kl', [eval_kl])

            #WD
            _, occ_measures2d, _ = Det_WD(env, P, R, s_size, a_size, alpha, gamma, E_dp, occ_measure_stoch2d_star, lr=lr_lambda, n_iter=n_iter)
            occ_measure_wd = np.sum(occ_measures2d, axis=1)
            np.save(path_occ_measures+'occ_measure_wd2d', occ_measures2d)

            #MCE IRL for WD
            reward_net = reward_nets.BasicRewardNet(
                env.observation_space,
                env.action_space,
                hid_sizes=[256],
                use_action=False,
                use_done=False,
                use_next_state=False,
            )

            mce_irl = MCEIRL(
                occ_measure_wd,
                env,
                reward_net,
                log_interval=250,
                discount=gamma,
                optimizer_kwargs={"lr": 0.01},
                rng=rng,
            )

            occ_measure, R_rec = mce_irl.train(max_iter=n_epochs_mce)
            #save R_rec at path_reward_functions without info about randomization rate and beta and info about mce
            np.save(path_reward_functions+'R_rec_mce_wd', R_rec)
            #optimal policy in R_rec
            Q, V, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_rec)
            _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
            #save occ_measure2d at path_occ_measures without info about randomization rate and beta and info about mce
            np.save(path_occ_measures+'occ_measure2d_mce_wd', occ_measure2d)
            E_rec = occ_measure.T@R
            # E_rec = max(E_rec, E_hat)
            print('E_rec_mce_wd', E_rec)
            eval_wd=(E_rec-E_hat)/(E_star-E_hat)
            np.save(path_eval+'eval_wd', [eval_wd])
                
            #remove info about sigma from all paths
            path_reward_functions = path_reward_functions.replace('sigma_'+str(sigma)+'/', '')
            path_occ_measures = path_occ_measures.replace('sigma_'+str(sigma)+'/', '')
            path_eval = path_eval.replace('sigma_'+str(sigma)+'/', '')
            path_model = path_model.replace('sigma_'+str(sigma)+'/', '')
        
        #remove info about seed from all paths
        path_reward_functions = path_reward_functions.replace('seed_'+str(seed)+'/', '')
        path_occ_measures = path_occ_measures.replace('seed_'+str(seed)+'/', '')
        path_eval = path_eval.replace('seed_'+str(seed)+'/', '')
        path_model = path_model.replace('seed_'+str(seed)+'/', '')



if __name__ == '__main__':
    main()

             


            

            



        
