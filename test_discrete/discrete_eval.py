import sys
sys.path.append('../')
import gym
import numpy as np 
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from policy_randomization import *
from utils import *
from featurize import *
from networks import *
from imitation.rewards import reward_nets
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.mce_irl import mce_occupancy_measures, MCEIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from temporary_seed import temporary_seed
from discrete_envs import RandomMDP, FourRooms, FrozenLake, Classic2D, CyberBattle
import os
from seals import base_envs
from softq import SoftQ_IQ
from callbacks import *
from iqlearn import IQLearn

#Device is cuda if availabe else cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Env Configs 
grid_size = 10  
room_size = 4
envs = ['FourRooms4x4']
seeds = [0, 1, 2, 3, 4]
# n_trajs_list = [10]
n_trajs_list = [-1]
#Policy Configs
policies = ['WD']
# policies = ['KL', 'WD', 'WDNN', 'f_div_kl', 'f_div_rkl', 'f_div_hellinger', 'f_div_pearson', 'f_div_tv', 'f_div_js']
n_modes = 1

betas_config = [0.01, 0.1, 1]
action_encoding = False
lr_lambda = 0.5
n_iter = 750

randomization = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#IRL configs
use_model_free = False 
use_true_om = True
n_epochs_mce = int(1e3)
n_epochs_iq = int(2*1e5)
strat=None
n_picks=1
n_clusters=2

#Log Path
path_log = 'discrete_results/'

def main():
    for env_name in envs:
        print('Env', env_name)
        for policy_name in policies:
            print('Policy', policy_name)
            exp_name = env_name+'_'+policy_name+'/'
            path = path_log+exp_name
            
            if(policy_name=='MaxEnt'):
                betas=[1]

            else:
                betas = betas_config
            #create this directory if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            for seed in seeds:
                print('Seed', seed)
                print()
                path_reward_functions = path+'reward_functions/'
                path_occ_measures = path+'occ_measures/'
                path_eval = path+'eval/'
                #add info about seed to all above paths and add a '/' at the end
                path_reward_functions = path_reward_functions+'seed_'+str(seed)+'/'
                path_occ_measures = path_occ_measures+'seed_'+str(seed)+'/'
                path_eval = path_eval+'seed_'+str(seed)+'/'

                #Create these directories if they don't exist
                if not os.path.exists(path_reward_functions):
                    os.makedirs(path_reward_functions)
                if not os.path.exists(path_occ_measures):
                    os.makedirs(path_occ_measures)
                if not os.path.exists(path_eval):
                    os.makedirs(path_eval)       
            
                if(env_name == 'random'):
                    env_ = RandomMDP(seed=seed)
                elif('FourRooms' in env_name):
                    with temporary_seed(seed):
                        # n_doors = np.random.randint(1, room_size, n_doors=room_size//2)
                        n_doors = room_size//2
                    env_ = FourRooms(seed=seed, room_size=room_size, n_doors=n_doors)
                elif('FrozenLake' in env_name):
                    env_ = FrozenLake(seed=seed, grid_size=grid_size)
                elif('Classic2D' in env_name):
                    env_ = Classic2D(seed=seed, grid_size=grid_size)
                elif('CyberBattle' in env_name):
                    env_ = CyberBattle(version=seed, horizon=100, n_seeds_per_version=25)
                
                s_size, a_size, P, R, alpha, gamma, horizon = env_.get_env()
                if(P.shape == (s_size, a_size, s_size)):
                    P = np.transpose(P, (1, 0, 2))
                #Save reward function at path_reward_functions wthout info about seed
                np.save(path_reward_functions+'reward_function', R)

                observation_matrix = np.vstack([featurize(s, s_size) for s in range(s_size)])
                env = gym.make('gym_examples/TabMDP-v0', n_states=s_size, n_actions=a_size, horizon=horizon, P=P, R=R, initial_state_dist=alpha, observation_matrix=observation_matrix)

                kwargs = {'n_states':s_size, 'n_actions':a_size, 'horizon':horizon, 'P':P, 'R':R, 'initial_state_dist':alpha, 'observation_matrix':observation_matrix}
                obs_to_state_map = {}
                for i in range(observation_matrix.shape[0]):
                    obs_to_state_map[str(observation_matrix[i])] = i

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
                
                pi_star_stoch, _, occ_measure_stoch2d_star = MaxEnt(env, alpha, gamma, E_star, method='true', lr = lr_lambda, n_iter=2000)
                #save occ_measure_stoch2d_star at path_occ_measures without info about seed 
                np.save(path_occ_measures+'occ_measure_stoch2d_star', occ_measure_stoch2d_star)

                pi_hat = np.ones((s_size, a_size))/a_size
                _, occ_measure_hat, occ_measure_hat2d = get_occupancy_measures(env=env,pi=pi_hat,discount=gamma)

                if('FrozenLake' in env_name):
                    E_hat = 0
                else:
                    E_hat = occ_measure_hat.T@R
                    # assert False, 'No policy found'
                # print('E_hat', E_hat)
                # save occ_measure_hat2d at path_occ_measures without info about seed
                np.save(path_occ_measures+'occ_measure_hat2d', occ_measure_hat2d)

                #Get Baseline for WD
                if('WD' in policy_name):
                    reward_net_wds = []
                    exp_baselines = []
                    for i in range(n_modes):

                        cntr = 0
                        if('NN' in policy_name):
                            layers_data = [(32, nn.ReLU())]
                        else:
                            layers_data = None
                        _, reward_net_wd, exp_baseline = Det_WD_baseline(env, P, R, s_size, a_size, alpha, gamma, occ_measure_stoch2d_star, lr_wd=0.1, n_iter=50, n_iter2=150, action_encoding=action_encoding, layers_data=layers_data)
                        while(exp_baseline>E_hat and cntr<10):
                            _, reward_net_wd, exp_baseline = Det_WD_baseline(env, P, R, s_size, a_size, alpha, gamma, occ_measure_stoch2d_star, lr_wd=0.1, n_iter=50, n_iter2=150, action_encoding=action_encoding, layers_data=layers_data)
                            cntr += 1

                        reward_net_wds.append(reward_net_wd)
                        exp_baselines.append(exp_baseline)

                    exp_baselines = np.array(exp_baselines)
                    exp_baseline = np.max(exp_baselines)
                    # assert exp_baseline<E_hat, 'Baseline reward is greater than minimum reward'

                elif('f_div' in policy_name):
                    dtype = policy_name.split('_')[2]
                    anti_r, exp_baseline = get_f_div_baseline(env, P, R, s_size, a_size, alpha, gamma, occ_measure_stoch2d_star, n_iter=100, f_div=dtype)

                elif('KL' in policy_name):
                    anti_r = -np.log(occ_measure_stoch2d_star + 1e-8) 
                    _, _, policy_kl = soft_vi(env=env, lambda_=250, discount=gamma, reward=anti_r)
                    _, occ_measure_kl, _ = get_occupancy_measures(env=env,pi=policy_kl,discount=gamma)
                    exp_baseline = occ_measure_kl.T@R      

                elif('MaxEnt' in policy_name):
                    exp_baseline = E_hat

                if(exp_baseline>E_hat):
                    print('Seed Skipped as Baseline is larger than E_hat')
                    continue
                
                if(strat is not None):
                    E_hat = exp_baseline
                
                print('E_hat', E_hat)

                for r in randomization:
                    #Remove info about r from all paths in the same variable
                    #add info about r to all paths in the same variable
                    path_reward_functions = path_reward_functions+'r_'+str(r)+'/'
                    path_occ_measures = path_occ_measures+'r_'+str(r)+'/'
                    path_eval = path_eval+'r_'+str(r)+'/'
                    if not os.path.exists(path_eval):
                        os.makedirs(path_eval)
                    if not os.path.exists(path_reward_functions):
                        os.makedirs(path_reward_functions)
                    if not os.path.exists(path_occ_measures):
                        os.makedirs(path_occ_measures)
                    
                    E_min = E_hat + (E_star-E_hat)*r
                    print('\nr', r)
                    print(flush=True)
                    print('E_min', E_min)

                    for beta in betas:
                        print('\nbeta', beta)
                        print()

                        #add info about beta to all paths in the same variable
                        path_reward_functions = path_reward_functions+'beta_'+str(beta)+'/'
                        path_occ_measures = path_occ_measures+'beta_'+str(beta)+'/'
                        path_eval = path_eval+'beta_'+str(beta)+'/'

                        if not os.path.exists(path_eval):
                            os.makedirs(path_eval)
                        if not os.path.exists(path_reward_functions):
                            os.makedirs(path_reward_functions)
                        if not os.path.exists(path_occ_measures):   
                            os.makedirs(path_occ_measures)

                        #Randomizing the policy
                        if(policy_name=='MaxEnt'):
                            if(r<0.1):
                                pi_beta = pi_hat
                                occ_measures1d_beta = occ_measure_hat
                                occ_measures2d_beta = occ_measure_hat2d
                            else:
                                pi_beta, occ_measures1d_beta, occ_measures2d_beta = MaxEnt(env, alpha, gamma, E_min, method='true', n_iter=n_iter)

                        elif(policy_name=='KL'):
                            pi_beta, _, R_new = Det_KL(env, P, R, s_size, a_size, alpha, gamma, E_min, pi_star_stoch, n_iter=n_iter, lr=lr_lambda)
                            pi_beta = BE(R_new, P, gamma, s_size, a_size, horizon, pi_beta, beta=beta)
                            _, occ_measures1d_beta, occ_measures2d_beta = get_occupancy_measures(env, pi=pi_beta, discount=gamma)
                        
                        elif('WD' in policy_name):
                            if('NN' in policy_name):
                                layers_data = [(32, nn.ReLU())]
                            else:
                                layers_data = None
                            occ_measures_2d_betas = []
                            for i in range(n_modes):
                                pi_beta, occ_measures2d_beta, R_new = Det_WD(env, P, R, s_size, a_size, alpha, gamma, E_min, occ_measure_stoch2d_star, lr=lr_lambda, n_iter=n_iter, action_encoding=action_encoding, exp_baseline=exp_baselines[i], reward_net=reward_net_wds[i], layers_data=layers_data)
                                pi_beta = BE(R_new, P, gamma, s_size, a_size, horizon, pi_beta, beta=beta)
                                _, occ_measures1d_beta, occ_measures2d_beta = get_occupancy_measures(env, pi=pi_beta, discount=gamma)
                                occ_measures_2d_betas.append(occ_measures2d_beta)
                                #save occ_measure 2d
                                np.save(path_occ_measures+'occ_measure2d_'+str(i), occ_measures2d_beta)

                            occ_measures_2d_betas = np.array(occ_measures_2d_betas)
                            occ_measures2d_beta = np.mean(occ_measures_2d_betas, axis=0)
                        
                        elif('f_div' in policy_name):
                            dtype = policy_name.split('_')[2]
                            pi_beta, occ_measures2d_beta, R_new = f_div(env, P, R, s_size, a_size, alpha, gamma, occ_measure_stoch2d_star, E_min, n_iter=100, n_iter_base=100, f_div=dtype, binary_search=True, lr=0.5, anti_r=anti_r, exp_baseline=exp_baseline)
                            pi_beta = BE(R_new, P, gamma, s_size, a_size, horizon, pi_beta, beta=beta)
                            _, occ_measures1d_beta, occ_measures2d_beta = get_occupancy_measures(env, pi=pi_beta, discount=gamma)

                        if(strat is not None):
                            if('FrozenLake' in env_name):
                                grid_size_ = grid_size
                            elif 'FourRooms' in env_name:
                                grid_size_ = 2*room_size+1
                            else:
                                grid_size_ = None
                            
                            rng_ = np.random.default_rng()
                            occ_measures1d_beta = filter_occ(occ_measures1d_beta, s_size, grid_size=grid_size_, n_clusters=n_clusters, n_picks=n_picks, strat=strat, rng=rng_)
                        #save occ_measures2d_beta at path_occ_measures without info about randomization rate and beta
                        np.save(path_occ_measures+'occ_measures2d_beta', occ_measures2d_beta)

                        #Evaluating the randomization policy
                        E_beta = max(get_E(s_size, a_size, occ_measures2d_beta, R, P), E_hat)
                        exp_rand_ = [(E_beta-E_hat)/(E_star-E_hat)]
                        #write exp_rand_ at path_eval without info about randomization rate and beta
                        np.save(path_eval+'exp_rand_', exp_rand_)

                        # exp_rand.append((E_beta-E_hat)/(E_star-E_hat))
                        print('E_beta', E_beta)

                        #Generating demonstrations
                        for n_trajs in n_trajs_list:
                            # exp_rand = []
                            exp_mce = []
                            # exp_gail = []
                            # exp_airl = []
                            exp_iq = []
                            #print ntrajs and an empty line after
                            print('\nn_trajs', n_trajs)
                            print()

                            path_reward_functions = path_reward_functions+'n_trajs_'+str(n_trajs)+'/'
                            path_occ_measures = path_occ_measures+'n_trajs_'+str(n_trajs)+'/'
                            path_eval = path_eval+'n_trajs_'+str(n_trajs)+'/'
                            if not os.path.exists(path_eval):
                                os.makedirs(path_eval)
                            if not os.path.exists(path_reward_functions):
                                os.makedirs(path_reward_functions)
                            if not os.path.exists(path_occ_measures):
                                os.makedirs(path_occ_measures)

                            
                            #MCE IRL
                            reward_net = reward_nets.BasicRewardNet(
                                env.observation_space,
                                env.action_space,
                                hid_sizes=[256],
                                use_action=False,
                                use_done=False,
                                use_next_state=False,
                            )

                            if(use_true_om==True):
                                # pi_beta_ = np.expand_dims(pi_beta, axis=0)
                                # pi_beta_ = np.repeat(pi_beta_, horizon, axis=0)
                                # _, om = mce_occupancy_measures(env, pi=pi_beta_, discount = gamma)

                                mce_irl = MCEIRL(
                                    occ_measures1d_beta,
                                    env,
                                    reward_net,
                                    log_interval=250,
                                    discount=gamma,
                                    optimizer_kwargs={"lr": 0.01},
                                    rng=rng,
                                )    

                            else:
                                #Generating demonstrations for MCE
                                policy_net1 = detPolicyNet(pi_beta, a_size, obs_to_state_map, hidden=True)
                                demos_mce = rollout.rollout(
                                    policy_net1,
                                    venv2,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=n_trajs), 
                                    rng=rng,
                                )

                                mce_irl = MCEIRL(
                                    demos_mce,
                                    env,
                                    reward_net,
                                    log_interval=250,
                                    discount=gamma,
                                    optimizer_kwargs={"lr": 0.01},
                                    rng=rng,
                                )           
                            
                            occ_measure, R_rec = mce_irl.train(max_iter=n_epochs_mce)
                            #save R_rec at path_reward_functions without info about randomization rate and beta and info about mce
                            np.save(path_reward_functions+'R_rec_mce', R_rec)
                            #optimal policy in R_rec
                            Q, V, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_rec)
                            _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
                            #save occ_measure2d at path_occ_measures without info about randomization rate and beta and info about mce
                            np.save(path_occ_measures+'occ_measure2d_mce', occ_measure2d)
                            E_rec = occ_measure.T@R
                            E_rec = max(E_rec, E_hat)
                            print('E_rec_mce', E_rec)
                            exp_mce.append((E_rec-E_hat)/(E_star-E_hat))

                            if(use_model_free==True):
                                #Generating demonstrations for GAIL/AIRL
                                policy_net = detPolicyNet(pi_beta, a_size, obs_to_state_map)
                                demos_ail = rollout.rollout(
                                    policy_net,
                                    venv1,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=n_trajs), 
                                    rng=rng,
                                )
                                

                                callback_list = [InfoCallback(freq=int(1e5)//8)]

                                #IQLearn
                                learner = SoftQ_IQ(
                                    policy='MlpPolicy',
                                    env=venv1,
                                    verbose=0,
                                    gamma=gamma,
                                    batch_size=64,
                                    expert_batch_size=64,
                                )
                                
                                IQ_trainer = IQLearn(
                                    venv=venv1,
                                    demonstrations=demos_ail,
                                    demo_batch_size=64,
                                    soft_policy_algo=learner, 
                                )

                                with HiddenPrints():
                                    IQ_trainer.train(n_epochs_iq, callback=callback_list)
                                with torch.no_grad():
                                    q_vals = learner.q_net.forward(torch.tensor(observation_matrix, device = learner.device))
                                    v_vals = learner._get_v_values(q_vals).cpu().numpy()
                                    q_vals = q_vals.cpu().numpy()

                                R_rec = np.zeros((s_size, a_size))
                                for s in range(s_size):
                                    for a in range(a_size):
                                        R_rec[s,a] = q_vals[s,a] - gamma*np.sum(v_vals*P[a,s])
                                np.save(path_reward_functions+'R_rec_iq', R_rec)
                                # exit(0)

                                #Optimal Policy in R_rec
                                Q, V, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_rec)
                                _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
                                #save occ_measure2d at path_occ_measures without info about randomization rate and beta and info about airl
                                np.save(path_occ_measures+'occ_measure2d_iq', occ_measure2d)
                                E_rec = occ_measure.T@R
                                E_rec = max(E_rec, E_hat)
                                print('E_rec_iq', E_rec)
                                exp_iq.append((E_rec-E_hat)/(E_star-E_hat))
                                


                            
                            #Saving exp_rand, exp_mce, exp_gail, exp_airl, exp_rand as numpy array in the folder path_eval
                            # np.save(path_eval + 'exp_rand.npy', exp_rand)
                            np.save(path_eval + 'exp_mce.npy', exp_mce)
                            if(use_model_free):
                                # np.save(path_eval + 'exp_gail.npy', exp_gail)
                                # np.save(path_eval + 'exp_airl.npy', exp_airl)
                                np.save(path_eval + 'exp_iq.npy', exp_iq)

                            #Remove info about n_trajs from all paths in the same variable if they exist
                            path_reward_functions = path_reward_functions.replace('n_trajs_'+str(n_trajs)+'/','')
                            path_occ_measures = path_occ_measures.replace('n_trajs_'+str(n_trajs)+'/','')
                            path_eval = path_eval.replace('n_trajs_'+str(n_trajs)+'/','')

                        #Remove info about beta from all paths in the same variable if they exist (use replace)
                        path_reward_functions = path_reward_functions.replace('beta_'+str(beta)+'/', '')
                        path_occ_measures = path_occ_measures.replace('beta_'+str(beta)+'/', '')
                        path_eval = path_eval.replace('beta_'+str(beta)+'/', '')

                    #Remove info about r from all paths in the same variable if they exist (use replace)
                    path_reward_functions = path_reward_functions.replace('r_'+str(r)+'/', '')
                    path_occ_measures = path_occ_measures.replace('r_'+str(r)+'/', '')
                    path_eval = path_eval.replace('r_'+str(r)+'/', '')
                
                #Remove info about seed from all paths in the same variable if they exist (use replace)
                path_reward_functions = path_reward_functions.replace('seed_'+str(seed)+'/', '')
                path_occ_measures = path_occ_measures.replace('seed_'+str(seed)+'/', '')
                path_eval = path_eval.replace('seed_'+str(seed)+'/', '')        

if __name__ == '__main__':
    main()               
                











