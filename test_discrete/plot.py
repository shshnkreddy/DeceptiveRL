#import from parent dir
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import simpson
from scipy.stats import pearsonr
from epic import deshape_pearson_distance, fully_connected_random_canonical_reward
from utils import get_Hw

class PlotObj:
    def __init__(self, env_name, save_path, read_dir=None):
        self.env_name = env_name
        if(read_dir is None):
            self.read_dir = './discrete_results/' + env_name + '_' 
        else:
            self.read_dir = read_dir + env_name + '_'
        self.save_path = save_path
        self.randomization = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


        #create save_apth if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def get_eval_metrics(self, seed, random_algo, beta, n_traj, model_free=True):
        
        eval_path = self.read_dir + str(random_algo) + f'/eval/seed_{seed}/'
        exp_mce = []
        exp_iq = []
        exp_rand = []

        for r in self.randomization:
            eval_path_ = eval_path + f'r_{r}/' + f'beta_{beta}/' 
            exp_rand.append(np.load(eval_path_ + 'exp_rand_.npy')[0])
            eval_path_ += f'n_trajs_{n_traj}/'
            exp_mce.append(np.load(eval_path_ + 'exp_mce.npy')[0])
            if(model_free):
                # exp_gail.append(np.load(eval_path_ + 'exp_gail.npy')[0])
                # exp_airl.append(np.load(eval_path_ + 'exp_airl.npy')[0])
                exp_iq.append(np.load(eval_path_ + 'exp_iq.npy')[0])
        if(model_free):
            return exp_rand, exp_iq
        
        return exp_rand, exp_mce
    
    def get_rewards(self, seed, random_algo, beta, ntrajs, model_free=True):
        reward_path = self.read_dir + str(random_algo) + f'/reward_functions/seed_{seed}/'
        r_gt = np.load(reward_path + 'reward_function.npy')

        occ_path = self.read_dir + str(random_algo) + f'/occ_measures/seed_{seed}/'
        occ_star_2d_star = np.load(occ_path + 'occ_measure_2d_star.npy')
        a_size = occ_star_2d_star.shape[1]

        r_mces = []
        r_iqs = []
        for r in self.randomization:
            reward_path_ = reward_path + f'r_{r}/' + f'beta_{beta}/' + f'n_trajs_{ntrajs}/'
            r_mce = np.load(reward_path_ + 'R_rec_mce.npy')
            r_mces.append(r_mce)
            if(model_free):
                r_iq = np.load(reward_path_ + 'R_rec_iq.npy')
                r_iqs.append(np.mean(r_iq, axis=1))
                # r_gail = np.load(reward_path_ + 'reward_function_gail.npy')
                # r_airl = np.load(reward_path_ + 'reward_function_airl.npy')
        if(model_free):
            return r_gt, r_iqs, a_size
        return r_gt, r_mces, a_size, 
    
    def get_entropy(self, seed, random_algo, beta, ntrajs):
        occ_path = self.read_dir + str(random_algo) + f'/occ_measures/seed_{seed}/'

        H_beta = []
        for r in self.randomization:
            occ_path_ = occ_path + f'r_{r}/' + f'beta_{beta}/' #+ f'n_trajs_{ntrajs}/'
            occ_measure = np.load(occ_path_ + 'occ_measures2d_beta.npy')
            H_beta.append(get_Hw(occ_measure))

        return H_beta


    def get_constraint_metrics(self, random_algo, seeds, betas, ntrajs=10, model_free=True):
        exp_rands = {}
        for seed in seeds:
            for beta in betas:
                exp_rand, _, _, _ = self.get_eval_metrics(seed, random_algo, beta, ntrajs, model_free)
                exp_rands[(seed, beta)] = np.array(exp_rand)

        return exp_rands

    def plot_eval_metrics(self, seed, exp_rand, exp_mce, exp_gail, exp_airl, random_algo, beta, n_traj):
        
        # randomization = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
        randomization = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        r_range = len(randomization)

        plt.clf()
        plt.cla()

        plt.plot(randomization, label='Constraint')
        plt.plot(exp_rand, label='Randomization Algorithm')
        plt.plot(exp_mce, label='MCE')
        plt.plot(exp_gail, label='GAIL')
        plt.plot(exp_airl, label='AIRL')
        
        plt.xlabel('Randomization Parameter')
        plt.ylabel('Fraction of Reward Obtained')
        plt.xticks(np.arange(0, r_range), randomization)
        plt.legend()

        #add info about the experiment to save path
        save_path = self.save_path+f'{self.env_name}_'
        #add info about seed
        save_path += f'_seed_{seed}_'
        save_path += f'{random_algo}_'
        save_path += f'beta_{beta}_'
        save_path += f'n_traj_{n_traj}'

        

        plt.savefig(save_path+'_eval.jpg')
        
    def plot_constraint_metrics(self, exp_rands):
        
        # randomization = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1])
        randomization = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        r_range = len(randomization)

        plt.clf()
        plt.cla()

        #iterate over exp_rands dictionary
        for (seed, beta) in exp_rands.keys():
            plt.plot(exp_rands[(seed, beta)]-randomization, label=f'seed_{seed}_beta_{beta}')
        
        plt.ylim(-1.0, 1.0)
        plt.ylabel('Constraint Violation')
        plt.xlabel('Randomization Parameter')
        plt.xticks(np.arange(0, r_range), randomization)
        plt.legend()
        plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_constraint.jpg')

    def get_collated_metrics_mce(self, seeds, random_algo, betas, model_free=False, ntrajs=-1):
        exp_rands_list = {}
        exp_mce_list = {}
        # if(model_free):
        #     exp_iq_list = {}

        if(random_algo == 'MaxEnt'):
            betas = [1]

        for beta in betas:
            exp_rands_list[beta] = np.zeros(11)
            exp_mce_list[beta] = np.zeros(11)
            # if(model_free):
            #     exp_iq_list[beta] = np.zeros(11)
        
        for seed in seeds:
            for beta in betas:
                # if(model_free):
                #     exp_rand, exp_iq = self.get_eval_metrics(seed=seed, beta=beta, n_traj=ntrajs, random_algo=random_algo, model_free=model_free)
                # else:
                exp_rand, exp_mce = self.get_eval_metrics(seed=seed, beta=beta, n_traj=ntrajs, random_algo=random_algo, model_free=model_free)
                
                if(len(exp_rand) == 6):
                    print('Seed', seed)
                exp_rands_list[beta] = np.vstack((exp_rands_list[beta], np.clip(np.array(exp_rand), 0, 1)))       
                exp_mce_list[beta] = np.vstack((exp_mce_list[beta], np.clip(np.array(exp_mce), 0, 1)))
                # if(model_free):
                #     exp_iq_list[beta] = np.vstack((exp_iq_list[beta], np.clip(np.array(exp_iq), 0, 1)))

        for beta in betas:
            exp_rands_list[beta] = exp_rands_list[beta][1:]
            exp_mce_list[beta] = exp_mce_list[beta][1:]
            # if(model_free):
            #     exp_iq_list[beta] = exp_iq_list[beta][1:]

        # if(model_free):
        #     return exp_rands_list, exp_iq_list

        return exp_rands_list, exp_mce_list
    
    def plot_collated_results(self, exp_rand_list, exp_mce_list, betas, random_algo):
        if random_algo == 'MaxEnt':
            betas = [1]
            random_algo_name = 'MEIR'
        else:
            random_algo_name = 'MM_' + random_algo
        
        for beta in betas:
            plt.clf()
            plt.cla()
            plt.plot(self.randomization, label='Exp Reward Threshold', color = 'blue')
            plt.plot(exp_rand_list[beta].mean(axis=0), label=random_algo_name, color='green')
            plt.plot(exp_mce_list[beta].mean(axis=0), label='MCE IRL', color='red')
            plt.xlabel('E_min')
            plt.ylabel('Fraction of Reward Obtained')
            plt.xticks(np.arange(0, 11), self.randomization)
            # plt.xticks(np.linspace(0, 1, 11))
            # plt.ylim(0.0, 1.0)
            plt.yticks(np.linspace(0, 1, 11))
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_{beta}_collated_eval.jpg')

    def plot_collated_dist_results(self, dists, dist_type, random_algo, dshort):
        plt.clf()
        plt.cla()
        plt.plot(dists, color = 'red')
        plt.xlabel('E_min')
        plt.ylabel(dist_type)
        plt.xticks(np.arange(0, 11), self.randomization)
        # plt.xticks(np.linspace(0, 1, 11))
        plt.ylim(0.0, 1.0)
        # plt.yticks(np.linspace(0, 1, 11))
        # plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_collated_{dshort}.jpg')

        
    def plot_constraint_collated_results(self, exp_rand_list, betas, random_algo):
        
        for beta in betas:
            plt.clf()
            plt.cla()
            plt.boxplot(exp_rand_list[beta]-self.randomization, positions=self.randomization, showfliers=False, widths=0.05)
            plt.xlabel('Reward Threshold')
            plt.ylabel('Constraint Violation')
            plt.xlim(0, 1)
            # plt.xticks(np.arange(0, 11), self.randomization)
            # plt.xticks(np.linspace(0, 1, 11))
            plt.ylim(-0.5, 0.5)
            plt.yticks(np.linspace(-0.5, 0.5, 11))
            # plt.yticks(np.linspace(-1, 1, 11))
            # plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_{beta}_collated_constraint.jpg')

    def get_excess_metrics(self, exp_rand_list, exp_mce_list, betas, random_algo):
        if random_algo == 'MaxEnt':
            betas = [1]

        excess_const = {}
        excess_rand = {}
        for beta in betas:
            excess_rand[beta] = np.zeros(11)
            excess_const[beta] = np.zeros(11)
        
        for beta in betas:
            excess_rand[beta] = simpson(exp_mce_list[beta].mean(axis=0) - exp_rand_list[beta].mean(axis=0), self.randomization)
            excess_const[beta] = simpson(exp_mce_list[beta].mean(axis=0) - self.randomization, self.randomization)
            # excess_rand[beta] = exp_mce_list[beta].mean(axis=0) - exp_rand_list[beta].mean(axis=0)
            # excess_const[beta] = exp_mce_list[beta].mean(axis=0) - self.randomization

        return excess_const, excess_rand

    def get_collated_pearson_correlation(self, seeds, random_algo, betas, average=False, model_free=False, ntrajs=-1):
        pearson_correlation = {}
        if(random_algo == 'MaxEnt'):
            betas = [1]

        for beta in betas:
            if(average==True):    
                pearson_correlation[beta] = np.zeros(10)

            else:
                pearson_correlation[beta] = np.zeros(11)

        for seed in seeds:
            for beta in betas:
                #should change the var name to r_tilde
                r_gt, r_mces, _ = self.get_rewards(seed=seed, beta=beta, random_algo=random_algo, ntrajs=ntrajs, model_free=model_free)
                
                if(average==True):
                    corr_arrray = np.array([np.abs(pearsonr(r_gt, r_mce)[0]) for r_mce in r_mces[1:]])
                else:
                    corr_arrray = np.array([np.abs(pearsonr(r_gt, r_mce)[0]) for r_mce in r_mces])
                pearson_correlation[beta] = np.vstack((pearson_correlation[beta], corr_arrray))

        for beta in betas:
            if(average==True):
                pearson_correlation[beta] = np.mean(pearson_correlation[beta][1:])
            else:
                pearson_correlation[beta] = np.mean(pearson_correlation[beta][1:], axis=0)
        
        return pearson_correlation
    
    def get_collated_epic_distance(self, seeds, random_algo, betas, gamma, average=False, model_free=False, ntrajs=-1):
        epic_distance = {}
        if(random_algo == 'MaxEnt'):
            betas = [1]

        for beta in betas:
            if(average==True):
                epic_distance[beta] = np.zeros(10)

            else:
                epic_distance[beta] = np.zeros(11)
            
        
        for seed in seeds:

            for beta in betas:
                    
                r_gt, r_mces, a_size = self.get_rewards(seed=seed, beta=beta, random_algo=random_algo, ntrajs=ntrajs, model_free=model_free)
                s_size = r_gt.shape[0]
                r_gt = np.expand_dims(r_gt, axis=(1,2))
                r_gt = np.repeat(r_gt, a_size, axis=1)
                r_gt = np.repeat(r_gt, s_size, axis=2)

                for i in range(len(r_mces)):
                    r_mces[i] = np.expand_dims(r_mces[i], axis=(1,2))
                    r_mces[i] = np.repeat(r_mces[i], a_size, axis=1)
                    r_mces[i] = np.repeat(r_mces[i], s_size, axis=2)
                
                if(average==True):
                    epic_distance[beta] = np.vstack((epic_distance[beta], np.array([deshape_pearson_distance(r_gt, r_mce, gamma, fully_connected_random_canonical_reward) for r_mce in r_mces[1:]])))

                else:
                    epic_distance[beta] = np.vstack((epic_distance[beta], np.array([deshape_pearson_distance(r_gt, r_mce, gamma, fully_connected_random_canonical_reward) for r_mce in r_mces])))
        for beta in betas:
            if(average==True):
                epic_distance[beta] = np.mean(epic_distance[beta][1:])

            else:
                epic_distance[beta] = np.mean(epic_distance[beta][1:], axis=0)

        return epic_distance

    def get_collated_entropy(self, seeds, random_algo, betas):
        hw = {}
        if(random_algo == 'MaxEnt'):
            betas = [1]

        for beta in betas:
            hw[beta] = np.zeros(11)
        
        for seed in seeds:
            for beta in betas:
                H_beta = self.get_entropy(seed=seed, beta=beta, random_algo=random_algo, ntrajs=-1)
                hw[beta] = np.vstack((hw[beta], np.array(H_beta)))

        for beta in betas:
            hw[beta] = np.mean(hw[beta][1:], axis=0)

        return hw
    
    def get_dp_metrics(self, dp_algo, seed, sigmas):
        read_path = self.read_dir + str(dp_algo) + f'/eval/seed_{seed}/'
        E_dp = []
        eval_dp = []
        eval_kl = []
        eval_wd = []

        for sigma in sigmas:
            read_path_sigma = read_path + f'sigma_{sigma}/'
            E_dp.append(np.load(read_path_sigma + 'E_dp.npy')[0])
            eval_dp.append(np.load(read_path_sigma + 'eval_dp.npy')[0])
            eval_kl.append(np.load(read_path_sigma + 'eval_kl.npy')[0])
            eval_wd.append(np.load(read_path_sigma + 'eval_wd.npy')[0])


        return np.array(E_dp), np.array(eval_dp), np.array(eval_kl), np.array(eval_wd)

    def get_collated_dp_metrics(self, dp_algo, seeds, sigmas):
        
        E_dp = np.zeros(len(sigmas))
        eval_dp = np.zeros(len(sigmas))
        eval_kl = np.zeros(len(sigmas))
        eval_wd = np.zeros(len(sigmas))
        
        for seed in seeds:
            E_dp_, eval_dp_, eval_kl_, eval_wd_ = self.get_dp_metrics(dp_algo, seed, sigmas)
            # print(E_dp.shape, E_dp_.shape)
            E_dp = np.vstack((E_dp, E_dp_))
            eval_dp = np.vstack((eval_dp, eval_dp_))
            eval_kl = np.vstack((eval_kl, eval_kl_))
            eval_wd = np.vstack((eval_wd, eval_wd_))
        E_dp = np.mean(E_dp[1:], axis=0)
        eval_dp = np.mean(eval_dp[1:], axis=0)
        eval_kl = np.mean(eval_kl[1:], axis=0)
        eval_wd = np.mean(eval_wd[1:], axis=0)

        return E_dp, eval_dp, eval_kl, eval_wd
    
    def get_dp_pearson_correlation(self, dp_algo, seed, sigmas):
        read_path = self.read_dir + str(dp_algo) + f'/reward_functions/seed_{seed}/'
        pearson_dp = []
        pearson_kl = []
        pearson_wd = []

        r_gt = np.load(read_path + 'reward_function.npy')
        for sigma in sigmas:
            read_path_sigma = read_path + f'sigma_{sigma}/'
            r_mce_dp = np.load(read_path_sigma + 'R_rec_mce_dp.npy')
            r_mce_kl = np.load(read_path_sigma + 'R_rec_mce_kl.npy')
            r_mce_wd = np.load(read_path_sigma + 'R_rec_mce_wd.npy')

            pearson_dp.append(pearsonr(r_gt, r_mce_dp)[0])
            pearson_kl.append(pearsonr(r_gt, r_mce_kl)[0])
            pearson_wd.append(pearsonr(r_gt, r_mce_wd)[0])

        return np.array(pearson_dp), np.array(pearson_kl), np.array(pearson_wd)
        
    def get_collated_dp_pearson_correlation(self, dp_algo, seeds, sigmas):
        pearson_dp = np.zeros(len(sigmas))
        pearson_kl = np.zeros(len(sigmas))
        pearson_wd = np.zeros(len(sigmas))

        for seed in seeds:
            pearson_dp_, pearson_kl_, pearson_wd_ = self.get_dp_pearson_correlation(dp_algo, seed, sigmas)
            pearson_dp = np.vstack((pearson_dp, pearson_dp_))
            pearson_kl = np.vstack((pearson_kl, pearson_kl_))
            pearson_wd = np.vstack((pearson_wd, pearson_wd_))

        pearson_dp = np.mean(pearson_dp[1:])
        pearson_kl = np.mean(pearson_kl[1:])
        pearson_wd = np.mean(pearson_wd[1:])

        return pearson_dp, pearson_kl, pearson_wd
    
    def get_epic_dp(self, dp_alog, seed, sigmas, a_size=4, gamma=0.99):
        read_path = self.read_dir + str(dp_algo) + f'/reward_functions/seed_{seed}/'
        epic_dp = []
        epic_kl = []
        epic_wd = []

        r_gt = np.load(read_path + 'reward_function.npy')
        s_size = r_gt.shape[0]
        r_gt = np.expand_dims(r_gt, axis=(1,2))
        r_gt = np.repeat(r_gt, a_size, axis=1)
        r_gt = np.repeat(r_gt, s_size, axis=2)

        for sigma in sigmas:
            read_path_sigma = read_path + f'sigma_{sigma}/'
            r_mce_dp = np.load(read_path_sigma + 'R_rec_mce_dp.npy')
            r_mce_dp = np.expand_dims(r_mce_dp, axis=(1,2))
            r_mce_dp = np.repeat(r_mce_dp, a_size, axis=1)
            r_mce_dp = np.repeat(r_mce_dp, s_size, axis=2)

            r_mce_kl = np.load(read_path_sigma + 'R_rec_mce_kl.npy')
            r_mce_kl = np.expand_dims(r_mce_kl, axis=(1,2))
            r_mce_kl = np.repeat(r_mce_kl, a_size, axis=1)
            r_mce_kl = np.repeat(r_mce_kl, s_size, axis=2)

            r_mce_wd = np.load(read_path_sigma + 'R_rec_mce_wd.npy')
            r_mce_wd = np.expand_dims(r_mce_wd, axis=(1,2))
            r_mce_wd = np.repeat(r_mce_wd, a_size, axis=1)
            r_mce_wd = np.repeat(r_mce_wd, s_size, axis=2)

            epic_dp.append(deshape_pearson_distance(r_gt, r_mce_dp, gamma, fully_connected_random_canonical_reward))
            epic_kl.append(deshape_pearson_distance(r_gt, r_mce_kl, gamma, fully_connected_random_canonical_reward))
            epic_wd.append(deshape_pearson_distance(r_gt, r_mce_wd, gamma, fully_connected_random_canonical_reward))

        return np.array(epic_dp), np.array(epic_kl), np.array(epic_wd)
    
    def get_collated_dp_epic(self, dp_algo, seeds, sigmas, a_size=4, gamma=0.99):
        epic_dp = np.zeros(len(sigmas))
        epic_kl = np.zeros(len(sigmas))
        epic_wd = np.zeros(len(sigmas))

        for seed in seeds:
            epic_dp_, epic_kl_, epic_wd_ = self.get_epic_dp(dp_algo, seed, sigmas, a_size, gamma)
            epic_dp = np.vstack((epic_dp, epic_dp_))
            epic_kl = np.vstack((epic_kl, epic_kl_))
            epic_wd = np.vstack((epic_wd, epic_wd_))

        epic_dp = np.mean(epic_dp[1:])
        epic_kl = np.mean(epic_kl[1:])
        epic_wd = np.mean(epic_wd[1:])

        return epic_dp, epic_kl, epic_wd
    
    def plot_hw(self, hw_max_ent, hw_kl, hw_wd):
        plt.clf()
        plt.cla()
        plt.plot(hw_max_ent[1], label='MEIR', color='blue')
        plt.plot(hw_kl[0.01], label='MMBE_KL beta=0.01', color='red', linestyle='dotted')
        plt.plot(hw_kl[0.05], label='MMBE_KL beta=0.05', color='red', linestyle='dashed')
        plt.plot(hw_kl[0.1], label='MMBE_KL beta=0.1', color='red', linestyle='dashdot')
        plt.plot(hw_wd[0.01], label='MMBE_WD beta=0.01', color='green', linestyle='dotted')
        plt.plot(hw_wd[0.05], label='MMBE_WD beta=0.05', color='green', linestyle='dashed')
        plt.plot(hw_wd[0.1], label='MMBE_WD beta=0.1', color='green', linestyle='dashdot')
        plt.xlabel('E_min')
        plt.ylabel('Causal Entropy')
        plt.xticks(np.arange(0, 11), self.randomization)
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(self.save_path+f'{self.env_name}_hw.jpg')

    def plot_dp_eval(self, E_dp, eval_dp, eval_kl, eval_wd):
        plt.clf()
        plt.cla()
        plt.plot(E_dp, label='E_DP')
        plt.plot(eval_dp, label='Eval_DP')
        plt.plot(eval_kl, label='Eval_KL')
        plt.plot(eval_wd, label='Eval_WD')
        plt.xlabel('Sigma')
        plt.ylabel('Fraction of Reward Obtained')
        plt.xticks(np.arange(0, len(sigmas)), sigmas)
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(self.save_path+f'{self.env_name}_{dp_algo}_eval.jpg')

    
    def plot_eval(self, map_eval, env_title, random_algo, algo_name, beta):
        plt.clf()
        plt.cla()
        plt.plot(map_eval[(self.env_name, random_algo, beta)][0], label=algo_name, marker='.', linestyle='dotted')
        # plt.plot(map_eval[(self.env_name, 'MaxEnt', 1)][0], label='MEIR', marker='x', linestyle='dashed')
        plt.plot(self.randomization, label='Reward Threshold', marker='*', linestyle='solid', color='blue')
        plt.plot(map_eval[(self.env_name, 'MaxEnt', 1)][1], label = 'IRL_MEIR', marker='o', linestyle='dashdot', color='red')
        plt.plot(map_eval[(self.env_name, random_algo, beta)][1], label = 'IRL_'+algo_name, marker='^', linestyle='solid', color='purple')
        plt.xlabel('$E_{min}$', fontsize=18)
        plt.ylabel('Fraction of Reward Obtained', fontsize=18)
        plt.xticks(np.arange(0, 11), self.randomization)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right', fontsize=15)
        # plt.title(env_title)
        # plt.grid()
        plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_{beta}_eval.jpg', dpi=1200, bbox_inches="tight")

    def plot_pear(self, map_pear, env_title, random_algo, algo_name, beta):
        plt.clf()
        plt.cla()
        # plt.plot(map_eval[(self.env_name, random_algo, beta)][0], label=algo_name, marker='.', linestyle='dotted')
        # plt.plot(map_eval[(self.env_name, 'MaxEnt', 1)][0], label='MEIR', marker='x', linestyle='dashed')
        # plt.plot(self.randomization, label='Reward Threshold', marker='*', linestyle='solid', color='blue')
        plt.plot(map_pear[(self.env_name, 'MaxEnt', 1)], label = 'MEIR', marker='o', linestyle='dashdot', color='red')
        plt.plot(map_pear[(self.env_name, random_algo, beta)], label = 'MM', marker='^', linestyle='solid', color='purple')
        plt.xlabel('$E_{min}$', fontsize=15)
        plt.ylabel('Pearson Correlation', fontsize=15)
        plt.xticks(np.arange(0, 11), self.randomization)
        # plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right', fontsize=15)
        # plt.title(env_title)
        # plt.grid()
        plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_{beta}_pearc.jpg', dpi=1200)

    def plot_epic(self, map_epic, env_title, random_algo, algo_name, beta):
        plt.clf()
        plt.cla()
        plt.plot(map_epic[(self.env_name, random_algo, beta)][0], label=algo_name, marker='.', linestyle='dotted')
        # plt.plot(map_eval[(self.env_name, 'MaxEnt', 1)][0], label='MEIR', marker='x', linestyle='dashed')
        # plt.plot(self.randomization, label='Reward Threshold', marker='*', linestyle='solid', color='blue')
        plt.plot(map_epic[(self.env_name, 'MaxEnt', 1)], label = 'MEIR', marker='o', linestyle='dashdot', color='red')
        plt.plot(map_epic[(self.env_name, random_algo, beta)], label = 'MM', marker='^', linestyle='solid', color='purple')
        plt.xlabel('$E_{min}$', fontsize=20)
        plt.ylabel('EPIC Distance', fontsize=20)
        plt.xticks(np.arange(0, 11), self.randomization)
        # plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right', fontsize=15)
        # plt.title(env_title)
        plt.savefig(self.save_path+f'{self.env_name}_{random_algo}_{beta}_epic.jpg', dpi=1200)


if __name__ == '__main__':
    #Max Ent Comparison
    dict_ = {
            # 'random': [0, 1, 2, 3, 4],
            # 'FrozenLake_5x5': [ 0, 3, 12, 39, 46],
            # 'FrozenLake_5x5': [ 0, 1, 2, 3, 4],
            'FrozenLake_10x10': [ 0, 1, 2, 3, 4], 
            # # 'FourRooms_4x4': [5, 6, 7, 8, 9],
            # 'FourRooms_4x4': [0, 1, 2, 3, 5],
            # 'FourRooms_6x6': [0, 1, 2, 3, 4]
            # 'CyberBattle': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
    
    model_free = False
    ntrajs = -1
    betas = [-1]
    average = True
    
    eval_map = {}
    epic_map = {}
    pear_map = {}
    
    #iterate over dict_
    for env_name in dict_.keys():
        print('Env:', env_name)
        seeds = dict_[env_name]
        
        random_algo_list = ['MaxEnt', 'WD']

        gamma = 0.99
        
        plot_obj = PlotObj(env_name=env_name, save_path='./plots_paper/')
        hw_kl = None
        hw_wd = None
        hw_max_ent = None

        for random_algo in random_algo_list:
            if(random_algo == 'MaxEnt'):
                betas_ = [1]
            else:
                betas_ = betas
            exp_rands_list, exp_mce_list = plot_obj.get_collated_metrics_mce(seeds, random_algo, betas, model_free=-model_free, ntrajs=ntrajs)
            for beta in betas_:
                eval_map[(env_name, random_algo, beta)] = [np.mean(exp_rands_list[beta], axis=0), np.mean(exp_mce_list[beta], axis=0)]


            pearson_correlation = plot_obj.get_collated_pearson_correlation(seeds, random_algo, betas, average=average, model_free=model_free, ntrajs=ntrajs)
            print('Pearson Correlation', pearson_correlation)
            if(random_algo == 'MaxEnt'):
                beta = 1
            else:
                beta = -1
            if(average==False):
                for beta in betas_:
                    pear_map[(env_name, random_algo, beta)] = pearson_correlation[beta]
            print('Pearson Correlation', pearson_correlation)
            epic_distance = plot_obj.get_collated_epic_distance(seeds, random_algo, betas, gamma=gamma, average=average, model_free=model_free, ntrajs=ntrajs)
            print('EPIC Distance', epic_distance)
            if(average==False):
                for beta in betas_:
                    epic_map[(env_name, random_algo, beta)] = epic_distance[beta]
        
        for random_algo in random_algo_list:
            if(random_algo == 'MaxEnt'):
                betas_ = [1]
            else:
                betas_ = betas
            for beta in betas_:
                plot_obj.plot_eval(eval_map, env_name, random_algo, 'MMBE', beta)
                if(average==False):
                    plot_obj.plot_pear(pear_map, env_name, random_algo, 'MM', beta)
                    plot_obj.plot_epic(epic_map, env_name, random_algo, 'MM', beta)

    #DP Comparison
    dict_ = {
            'random' : [1,2,3,4,5], 
            'FourRooms_4x4' : [1,2,3,4,5],
            'FrozenLake_5x5' : [1,2,3,4],
    }

    sigmas = [0.5, 1.0, 1.5, 2.0]
    dp_algo = 'qfn'
   
    for env_name in dict_.keys():
        print('Env:', env_name)
        if('Frozen' in env_name):
            sigmas = [0.5, 1.0]
        plot_obj = PlotObj(env_name=env_name, save_path='./dp_plots/', read_dir='./dp_results/')
        E_dp, eval_dp, eval_kl, eval_wd = plot_obj.get_collated_dp_metrics(dp_algo, dict_[env_name], sigmas)
        print('E_dp', E_dp)
        print('eval_dp', eval_dp)
        print('eval_kl', eval_kl)
        print('eval_wd', eval_wd)
        e2_dp = simpson(eval_dp-E_dp, sigmas)
        e2_kl = simpson(eval_kl-E_dp, sigmas)
        e2_wd = simpson(eval_wd-E_dp, sigmas)
        print('E2_dp', e2_dp)
        print('E2_kl', e2_kl)
        print('E2_wd', e2_wd)
        pearson_dp, pearson_kl, pearson_wd = plot_obj.get_collated_dp_pearson_correlation(dp_algo, dict_[env_name], sigmas)
        print('Pearson DP', pearson_dp)
        print('Pearson KL', pearson_kl)
        print('Pearson WD', pearson_wd)

        epic_dp, epic_kl, epic_wd = plot_obj.get_collated_dp_epic(dp_algo, dict_[env_name], sigmas)
        print('EPIC DP', epic_dp)
        print('EPIC KL', epic_kl)
        print('EPIC WD', epic_wd)
        




