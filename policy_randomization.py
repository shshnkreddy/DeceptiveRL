# from cvxopt import matrix, solvers
import numpy as np
from utils import get_E, soft_vi, get_occupancy_measures, get_pi, normalize
# import mosek
from seals.base_envs import TabularModelPOMDP as TabularModelEnv
from networks import *
from stable_baselines3.common.evaluation import evaluate_policy
from gym_examples.wrappers import hidden_wrapper
import torch.nn.functional as F

def MaxEnt(env: TabularModelEnv, alpha, gamma, E_min, H_min=None, lr=0.5, lr_mu=0.1, n_iter=100, method='true', binary_search=True):
    lambda_ = np.random.random()
    mu_= np.random.random()

    if(binary_search==True):
        l = 0
        r = 250
        lambda_ = (l+r)/2

    # lambda_ = -1
    print_rate = n_iter // 10
    R = env.reward_matrix
    horizon = env.horizon

    for i in range(n_iter):    
        if H_min is None:
            mu_ = 1.0
        Q, V, policy = soft_vi(env=env, lambda_=lambda_, discount=gamma, mu_=mu_) 

        if(method=='roll'):
            policy_net = detPolicyNet(pi=np.mean(policy, axis=0), hidden=True, n_actions=env.n_actions)
            exp_reward, _ = evaluate_policy(policy_net, hidden_wrapper(env), n_eval_episodes=10)

        elif(method=='true'):    
            _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
            exp_reward = occ_measure.T@R
            exp_entropy = np.sum(-occ_measure2d*np.log(occ_measure2d/np.expand_dims(occ_measure, axis=1)))

        if(i%print_rate==0):
            if(H_min is not None):
                print(f'Reward: {exp_reward} Lambda: {lambda_} Entropy: {exp_entropy} Mu:{mu_}',flush=True)
            else:
                print(f'Reward: {exp_reward} Lambda: {lambda_}', flush=True)

        grad_lambda = exp_reward-E_min 
        if(binary_search==True):
            if(exp_reward<E_min):
                l = lambda_
            else:
                r = lambda_
            lambda_ = (l+r)/2
        else:
            
            lambda_ -= lr*grad_lambda

        if(np.abs(grad_lambda) < 1e-4):
            print(f'Reward: {exp_reward} Lambda: {lambda_}', flush=True)
            break

        if(H_min is not None):
            if(exp_entropy<H_min):       
                grad_mu = exp_entropy-H_min
            else:
                grad_mu = 0
            mu_ -= lr_mu*grad_mu
    
    _, occ_measure, occ_measure2d = get_occupancy_measures(env=env,pi=policy,discount=gamma)
    policy = np.mean(policy, axis=0)
    return policy, occ_measure, occ_measure2d


def Det_KL(env, P, R, s_size, a_size, alpha, gamma, E_min, pi_star, lr=1, n_iter=1000, binary_search=True):
    lambda_ = np.random.random()
    print_rate = n_iter // 10
    # print_rate = 1

    if(binary_search==True):
        l = 0
        r = 250
        lambda_ = (l+r)/2
    
    if len(R.shape) == 1:
        R = np.expand_dims(R, axis=1)
        R = np.repeat(R, a_size, axis=1)
    
    for i in range(n_iter):
        R_new = lambda_*R - np.log(pi_star+1e-8)

        Q, V, policy = soft_vi(env=env, reward=R_new, lambda_=250, discount=gamma) #Line 4 in Alg. 1, Line 5 in Alg. 3
        _, occ_measure, x = get_occupancy_measures(env=env,pi=policy,discount=gamma)
        
        exp_reward = get_E(s_size, a_size, x, R, P = P)

        grad_lambda = exp_reward-E_min 
        
        if(i%print_rate==0):
            print(f'Reward: {exp_reward} Lambda: {lambda_} Grad Lambda: {grad_lambda}',flush=True)

        #Alg 3
        if(binary_search==True):
            if(exp_reward<E_min):
                l = lambda_
            else:
                r = lambda_
            lambda_ = (l+r)/2
        #Alg 1
        else:
            lambda_ -= lr*grad_lambda

        if(np.abs(grad_lambda) < 1e-4):
            print(f'Reward: {exp_reward} Lambda: {lambda_} Grad Lambda: {grad_lambda}',flush=True)
            break

    pi = get_pi(x, s_size, a_size)

    return pi, x, R_new

def Det_WD(env, P, R, s_size, a_size, alpha, gamma, E_min, x_star, lr=0.01, lr_wd=0.1, n_iter=750, n_iter2=250, n_iter_base1=50, n_iter_base2=150, action_encoding=True, binary_search=True, update_rewards=False, scale=10, exp_baseline=None, reward_net=None, layers_data=[(32, nn.ReLU())]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lambda_ = np.random.random()
    if(binary_search==True):
        l = 0
        r = 250
        lambda_ = (l+r)/2

    if(update_rewards==True):
        lambda_ = 0

    x_star = np.reshape(x_star, (s_size,a_size))
    
    observation_matrix = torch.tensor(env.observation_matrix, device=device, dtype=torch.float32)
    if(action_encoding==True):
        #create one hot encoding of actions and combine with observation matrix
        action_matrix = F.one_hot(torch.arange(a_size, device=device))
        observation_matrix_ = torch.vstack([torch.cat((observation_matrix[i], action_matrix[j])) for i in range(s_size) for j in range(a_size)])
        observation_matrix = observation_matrix_
    
    #Generate the anti-reward
    if(reward_net is None):
        _, reward_net, exp_baseline = Det_WD_baseline(env, P, R, s_size, a_size, alpha, gamma, x_star, lr_wd=lr_wd, n_iter=n_iter_base1, n_iter2=n_iter_base2, action_encoding=action_encoding, layers_data=layers_data)
        cntr = 0
        while(exp_baseline>E_min and cntr<10):
            _, reward_net, exp_baseline = Det_WD_baseline(env, P, R, s_size, a_size, alpha, gamma, x_star, lr_wd=lr_wd, n_iter=n_iter_base1, n_iter2=n_iter_base2, action_encoding=action_encoding, layers_data=layers_data)
            cntr += 1

    assert exp_baseline<E_min, 'Baseline reward is greater than minimum reward'

    if(update_rewards==True):
        optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr_wd)
    
    R_aug = reward_net(observation_matrix)
    R_aug = scale*F.normalize(R_aug.T).T
    R_aug = R_aug.detach().cpu().numpy()
            
    if(action_encoding==True):
        R_aug = R_aug.reshape((s_size,a_size))
        if(len(R.shape) == 1):
            R = np.expand_dims(R, axis=1)
            R = np.repeat(R, a_size, axis=1)
    
    else:
        R_aug = R_aug.T[0]

    x_star_tensor = torch.tensor(x_star, device=device, dtype=torch.float32)
    
    # print_rate = 1
    print_rate = n_iter//10

    for i in range(n_iter+1):
        R_new = lambda_*R + R_aug
        
        _, _, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_new) #Line 4 in Alg. 1, Line 5 in Alg. 3
        _, x_1d, x = get_occupancy_measures(env=env,pi=policy,discount=gamma)
        if(action_encoding==True):
            exp_reward = x.flatten().T@R.flatten()
        else:
            exp_reward = x_1d.T@R

        grad_lambda = exp_reward-E_min

        #Depricated 
        if(update_rewards==True):
            # reset_model_weights(layer=reward_net)

            x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
            for j in range(n_iter2):  
                optimizer.zero_grad()
                R_aug = reward_net(observation_matrix)
                R_aug = scale*F.normalize(R_aug.T).T
                if(action_encoding==True):
                    R_aug = R_aug.reshape((s_size, a_size))
                
                loss = -torch.sum((x_tensor-x_star_tensor)*R_aug)
                loss.backward()
                optimizer.step()


        if(i%print_rate==0):
            print(f'Reward: {exp_reward} Lambda: {lambda_} Grad Lambda: {grad_lambda}',flush=True)

        #Alg 3
        if(binary_search==True):
            if(grad_lambda > 0):
                r = lambda_
            else:
                l = lambda_
            lambda_ = (l+r)/2
        #Alg 1
        else:
            lambda_ -= lr*grad_lambda

        if(np.abs(grad_lambda) < 1e-4):
            print(f'Reward: {exp_reward} Lambda: {lambda_} Grad Lambda: {grad_lambda}',flush=True)
            break
        
    pi = get_pi(x, s_size, a_size)
    return pi, x, R_new

#Generates the WD based anti-reward
def Det_WD_baseline(env, P, R, s_size, a_size, alpha, gamma, x_star, lr_wd=0.1, n_iter=100, n_iter2=250, action_encoding=True, scale=10, layers_data=[(32, nn.ReLU())]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_star = np.reshape(x_star, (s_size,a_size))
    
    observation_matrix = torch.tensor(env.observation_matrix, device=device, dtype=torch.float32)
    if(action_encoding==True):
        #create one hot encoding of actions and combine with observation matrix
        action_matrix = F.one_hot(torch.arange(a_size, device=device))
        observation_matrix_ = torch.vstack([torch.cat((observation_matrix[i], action_matrix[j])) for i in range(s_size) for j in range(a_size)])
        observation_matrix = observation_matrix_.to(device)
        
    obs_dim = observation_matrix.shape[1]
    
    reward_net = WDNNReward(obs_dim, layers_data=layers_data)
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr_wd)
    R_aug = reward_net(observation_matrix)
    R_aug = scale*F.normalize(R_aug.T).T
    if(action_encoding==True):
        R_aug = R_aug.reshape((s_size,a_size))
        if(len(R.shape) == 1):
            R = np.expand_dims(R, axis=1)
            R = np.repeat(R, a_size, axis=1)

    x_star_tensor = torch.tensor(x_star, device=device, dtype=torch.float32)
    print_rate = n_iter//10

    for i in range(n_iter+1):
        R_aug = R_aug.detach().cpu().numpy()
        if(action_encoding == False):
            R_aug = R_aug.T[0]

        Q, V, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_aug) #Line 5 in Alg 2
        _, x_1d, x = get_occupancy_measures(env=env,pi=policy,discount=gamma)
        if(action_encoding==True):
            exp_reward = x.flatten().T@R.flatten()
        else:
            exp_reward = x_1d.T@R

        x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
        for j in range(n_iter2):  
            optimizer.zero_grad()
            R_aug = reward_net(observation_matrix)
            R_aug = scale*F.normalize(R_aug.T).T
            if(action_encoding==True):
                R_aug = R_aug.reshape((s_size, a_size))
            
            #Maximze the Wasserstein distance between the occupancy measure and occupancy measure of \pi^* 
            loss = -torch.sum((x_tensor-x_star_tensor)*R_aug) #Line 4 in Alg 2
            loss.backward()
            optimizer.step()

    print(f'Baseline Reward: {exp_reward}', flush=True)
    
    return R_aug, reward_net, exp_reward

def f_div(env, P, R, s_size, a_size, alpha, gamma, x_star, E_min, n_iter=100, n_iter_base=100, f_div='kl', binary_search=True, lr=0.5, anti_r=None, exp_baseline=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_star = np.reshape(x_star, (s_size,a_size))
    # observation_matrix = torch.tensor(env.observation_matrix, device=device, dtype=torch.float32)
    
    if(anti_r is None):
        anti_r, exp_baseline = get_f_div_baseline(env, P, R, s_size, a_size, alpha, gamma, x_star, n_iter=n_iter_base, f_div=f_div)
        
    assert exp_baseline<E_min, 'Baseline reward is greater than minimum reward'

    if(binary_search==True):
        l = 0
        r = 250
        lambda_ = (l+r)/2
    
    else:
        lambda_ = np.random.random()

    if(len(R.shape) == 1):
        R = np.expand_dims(R, axis=1)
        R = np.repeat(R, a_size, axis=1)

    print_rate = n_iter // 10
    for i in range(n_iter+1):

        R_new = lambda_*R + anti_r
        _, _, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=R_new) #Line 4 in Alg. 1, Line 5 in Alg. 3
        _, x_1d, x = get_occupancy_measures(env=env,pi=policy,discount=gamma)
        exp_reward = x.flatten().T@R.flatten()

        grad_lambda = exp_reward-E_min
        if(binary_search==True):
            if(grad_lambda > 0):
                r = lambda_
            else:
                l = lambda_
            lambda_ = (l+r)/2
        else:
            lambda_ -= lr*grad_lambda

        if(i%print_rate==0):
            print(f'Reward: {exp_reward} Lambda: {lambda_} Grad Lambda: {grad_lambda}',flush=True)

        if(np.abs(grad_lambda) < 1e-4):
            print(f'Reward: {exp_reward} Lambda: {lambda_} Grad Lambda: {grad_lambda}',flush=True)
            break

    pi = get_pi(x, s_size, a_size)
    return pi, x, R_new

#Generate f_div based anti-reward
def get_f_div_baseline(env, P, R, s_size, a_size, alpha, gamma, x_star, n_iter=100, f_div=f_div, eps=1e-8):
    #Random init of policy
    pi_hat = np.ones((s_size, a_size))/a_size
    _, _, x = get_occupancy_measures(env=env,pi=pi_hat,discount=gamma)

    for i in range(n_iter+1):
        ratio = (x+eps)/(x_star+eps)

        #Closed form solutions given in Table 2 (Line 4 in Alg 2)
        if(f_div=='kl'):
            anti_r = ratio
    
        elif(f_div=='rkl'):
            anti_r = -(1+np.log(1/ratio))

        elif(f_div=='hellinger'):
            anti_r = np.sqrt(ratio)-1

        elif(f_div=='pearson'):
            anti_r = 2*(1-1/ratio)

        elif(f_div=='tv'):
            anti_r = 0.5*np.sign(1-1/ratio)
        
        elif(f_div=='js'):
            anti_r = np.log(0.5*(1+ratio))

        _, _, policy = soft_vi(env=env, lambda_=250, discount=gamma, reward=anti_r)  #Line 5 in Alg 2
        _, x_1d, x = get_occupancy_measures(env=env,pi=policy,discount=gamma)

    anti_r = normalize(anti_r)
    exp_reward = x_1d.T@R
    print(f'Baseline Reward: {exp_reward}', flush=True)

    return anti_r, exp_reward
        