import numpy as np
from scipy.stats import entropy
import scipy.special
from seals.base_envs import  TabularModelPOMDP as TabularModelEnv
from typing import Optional, Tuple
from imitation.data.rollout import discounted_sum
from scipy.spatial.distance import jensenshannon
import sys
import os
from sklearn.cluster import KMeans

def check_sol(x, A, alpha):
    # print((A @ x).shape, alpha.shape)
    
    res1 = np.squeeze(A @ x) - alpha
    res1 = np.mean(res1**2)
    
    res2 = np.where(x < 0)
    flag = True
    if(len(res2[0]) > 0):
        flag = False
    
    print('MSE in Ax = alpha', res1)
    print('Valid Solution x(s,a) > 0: ', flag)
    print()
    return res1, res2

def get_pi(x, s_size, a_size):
    x = np.reshape(x, (s_size,a_size))
    pi = np.zeros((s_size,a_size))
    
    for s in range(s_size):
        d = np.sum(x[s])
        if(d == 0):
            pi[s] = np.zeros(a_size)
            pi[s, 0] = 1
            continue

        for a in range(a_size):
            pi[s,a] = x[s,a] / d

    return pi

def get_v(pi, P, R, gamma = 0.9, n_iter = 1000, eps = 1e-3):

    a_size, s_size, _ = P.shape
    V = np.zeros(s_size)

    conv = False
    
    if(len(pi.shape) > 1):
        det = False
    else:
        det = True

    i = 0
    while(i < n_iter and conv == False):
        conv = True
        i += 1
        for s in range(s_size): 
            exp_reward = 0
            if(det):
                a_ = pi[s]
                
                for s_ in range(s_size):
                    if(len(R.shape) == 3):
                        exp_reward += P[a_, s, s_] * (R[a_, s, s_] + gamma * V[s_])
                    elif (len(R.shape) == 2):
                        exp_reward += P[a_, s, s_] * (R[s, a_] + gamma * V[s_])
                    else:
                        exp_reward += P[a_, s, s_] * (R[s] + gamma * V[s_])
            
            else:
                for a in range(a_size):
                    for s_ in range(s_size):
                        if(len(R.shape) == 3):
                            exp_reward += pi[s, a] * P[a, s, s_] * (R[a, s, s_] + gamma * V[s_])
                        elif (len(R.shape) == 2):
                            exp_reward += pi[s, a] * P[a, s, s_] * (R[s,a] + gamma * V[s_])
                        else:
                            exp_reward += pi[s, a] * P[a, s, s_] * (R[s] + gamma * V[s_])
                        
            
            # print(exp_reward)  
            if(np.abs(V[s]-exp_reward)>eps):
                conv = False
            V[s] = exp_reward
    
    if(i == n_iter):
        print('Value Iteration failed to converge')
    return V

    
def get_E(s_size, a_size, x, R, P = None):
    
    x = np.reshape(x, (s_size,a_size))
    
    e = 0
    # print(R, P, x)
    for s in range(s_size):
        for a in range(a_size):
            if(len(R.shape) > 2):
                for s_ in range(s_size):
                    e += P[a, s, s_] * R[a, s, s_] * x[s,a]

            elif(len(R.shape) == 2):
                e += R[s,a] * x[s,a]
            
            else:
                e += R[s] * x[s,a]

    return e

def get_Hw(x, alpha=None):
    hw = 0
    x += 1e-12
    
    if(alpha is None):
        c = 1
    else:
        c = np.sum(alpha)

    pi_ = get_pi(x, x.shape[0], x.shape[1])
    hw = -np.sum(x*np.log(pi_))    
    
    return hw/c

# def logsumexp(x):
#     c = x.max()
#     return c + np.log(np.sum(np.exp(x - c)))

# def get_random_policy(s_size,a_size):
#     pi = np.random.random_sample((s_size, a_size))
#     for s in range(s_size):
#         pi[s] = soft_max(pi[s])
#     return pi

def KL(pi1, pi2):
    s_size, a_size = pi1.shape
    return np.mean(np.vstack([entropy(pi1[s], pi2[s]) for s in range(s_size)]))

# def JS(pi1, pi2):
#     s_size, a_size = pi1.shape
#     return np.mean(np.vstack([np.sqrt((entropy(pi1[s], pi2[s]) + entropy(pi2[s], pi1[s]))/2) for s in range(s_size)]))

def soft_vi(
    env: TabularModelEnv,
    *,
    reward: Optional[np.ndarray] = None,
    discount: float = 1.0,
    lambda_: 1.0,
    mu_ = 1.0, 
    R_aug = None #shape (s,a)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs the soft Bellman backup for a finite-horizon MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env: a tabular, known-dynamics MDP.
        reward: a reward matrix. Defaults to env.reward_matrix.
        discount: discount rate.

    Returns:
        (V, Q, \pi) corresponding to the soft values, Q-values and MCE policy.
        V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
        \pi is a 3d array, indexed \pi[t,s,a].

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix

    # Initialization
    # indexed as V[t,s]
    V = np.full((horizon, n_states), -np.inf)
    # indexed as Q[t,s,a]
    Q = np.zeros((horizon, n_states, n_actions))

    if(len(reward.shape) == 1):
        broad_R = reward[:, None]
    else:
        broad_R = reward

    if R_aug is None:
        R_aug = np.zeros_like(broad_R)

    if(len(R_aug.shape) == 1):
        R_aug = R_aug[:, None]

    broad_R = (lambda_*broad_R + R_aug)/(mu_)
    # print(broad_R.shape)
    # print(broad_R.reshape(5, 5))
    # print(broad_R.shape, R_aug.shape)

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = broad_R
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
        next_values_s_a = T @ V[t + 1, :]
        Q[t, :, :] = broad_R + discount*next_values_s_a
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi

# def QLearning(
#     env: TabularModelEnv,
#     *,
#     reward: Optional[np.ndarray] = None,
#     discount = 1.0,
#     lambda_ = 1.0, 
#     KL_max = False,
#     pi_star = None
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     r"""Performs the soft Bellman backup for a finite-horizon MDP.

#     Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
#     (9.3) from Ziebart (2010).

#     Args:
#         env: a tabular, known-dynamics MDP.
#         reward: a reward matrix. Defaults to env.reward_matrix.
#         discount: discount rate.

#     Returns:
#         (V, Q, \pi) corresponding to the soft values, Q-values and MCE policy.
#         V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
#         \pi is a 3d array, indexed \pi[t,s,a].

#     Raises:
#         ValueError: if ``env.horizon`` is None (infinite horizon).
#     """
#     # shorthand
#     horizon = env.horizon
#     if horizon is None:
#         raise ValueError("Only finite-horizon environments are supported.")
#     n_states = env.n_states
#     n_actions = env.n_actions
#     T = env.transition_matrix
#     if reward is None:
#         reward = env.reward_matrix

#     if KL_max == True:
#         log_pi_star = -np.log(pi_star)

#     # Initialization
#     # indexed as V[t,s]
#     V = np.full((horizon, n_states), -np.inf)
#     # indexed as Q[t,s,a]
#     Q = np.zeros((horizon, n_states, n_actions))

    
#     broad_R = lambda_ * reward[:, None]
#     # print(broad_R)

#     # Base case: final timestep
#     # final Q(s,a) is just reward
#     Q[horizon - 1, :, :] = broad_R
#     # V(s) is always normalising constant
#     V[horizon - 1, :] = np.max(Q[horizon - 1, :, :], axis=1)

#     # Recursive case
#     for t in reversed(range(horizon - 1)):
#         next_values_s_a = T @ V[t + 1, :]
#         Q[t, :, :] = broad_R + discount*next_values_s_a

#         if KL_max == True:
#             Q[t, :, :] += log_pi_star*KL

#         V[t, :] = np.max(Q[t, :, :], axis=1)

#     pi_ = np.argmax(Q, axis=2)
#     pi = np.zeros((horizon, n_states, n_actions))
#     for t in range(horizon):
#         for s in range(n_states):
#             pi[t,s,pi_[t,s]] = 1

#     return pi, Q

def Q_evaluation(R, P, gamma, s_size, a_size, horizon, pi):
    if(P.shape == (a_size, s_size, s_size)):
        P = np.transpose(P, (1,0,2))

    if(len(R.shape) == 1):
        R = np.expand_dims(R, axis=1)
        R = np.repeat(R, a_size, axis=1)

    if(pi.shape == (s_size, a_size)):
        pi = np.expand_dims(pi, axis=0)
        pi = np.repeat(pi, axis=0, repeats=horizon)


    Q = np.zeros((horizon, s_size, a_size))
    V = np.zeros((horizon, s_size))
    Q[horizon-1, :, :] = R #shape of R=(s,a)
    V[horizon-1, :] = np.sum(Q[horizon-1, :, :]*pi[horizon-1, :, :], axis=1)

    for t in reversed(range(horizon - 1)):
        next_values_s_a = P @ V[t+1, :]
        Q[t, :, :] = R + gamma*next_values_s_a
        V[t, :] = np.sum(Q[t,:,:]*pi[t,:,:], axis=1)

    # Q = np.mean(Q, axis=0)
    # V = np.mean(V, axis=0)

    return Q, V


def get_occupancy_measures(
    env: TabularModelEnv,
    *,
    reward: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    discount: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state visitation frequency Ds for each state s under a given policy pi.

    You can get pi from `mce_partition_fh`.

    Args:
        env: a tabular MDP.
        reward: reward matrix. Defaults is env.reward_matrix.
        pi: policy to simulate. Defaults to soft-optimal policy w.r.t reward
            matrix.
        discount: rate to discount the cumulative occupancy measure D.

    Returns:
        Tuple of ``D`` (ndarray) and ``Dcum`` (ndarray). ``D`` is of shape
        ``(env.horizon, env.n_states)`` and records the probability of being in a
        given state at a given timestep. ``Dcum`` is of shape ``(env.n_states,)``
        and records the expected discounted number of times each state is visited.
    """
    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix
    
    if(pi.shape == (n_states, n_actions)):
        pi = np.expand_dims(pi, axis=0)
        pi = np.repeat(pi, axis=0, repeats=horizon)

    D = np.zeros((horizon + 1, n_states))
    D2 = np.zeros((horizon, n_states, n_actions))
    D[0, :] = env.initial_state_dist
    

    for t in range(horizon):
        for a in range(n_actions):
            D2[t, :, a] = D[t] * pi[t, :, a]
            D[t+1, :] += D2[t, :, a] @ T[:, a, :]
            

    Dcum = discounted_sum(D[:-1], discount)
    Dcum2 = discounted_sum(D2, discount)
    assert isinstance(Dcum, np.ndarray) and isinstance(Dcum2, np.ndarray)
    return D, Dcum, Dcum2

def BE(R, P, gamma, s_size, a_size, horizon, pi, beta=0.1):
    if(beta == -1):
        return pi
    Q, _ = Q_evaluation(R, P, gamma, s_size, a_size, horizon, pi)
    Q = np.mean(Q, axis=0)
    pi_soft = scipy.special.softmax(Q/beta, axis=1)
    
    return pi_soft

def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)

def eps_greedy(pi, eps):
    """
    Epsilon-greedy policy
    """
    if(len(pi.shape) == 3):
        horizon, n_states, n_actions = pi.shape
        pi_new = np.ones_like(pi)*eps / n_actions
        pi_new[np.arange(horizon), np.arange(n_states), np.argmax(pi, axis=2)] += 1 - eps

    else:
        n_states, n_actions = pi.shape
        pi_new = np.ones_like(pi)*eps / n_actions
        pi_new[np.arange(n_states), np.argmax(pi, axis=1)] += 1 - eps
    return pi_new

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def normalize(x):
    return x/max(np.linalg.norm(x), 1e-8)

def filter_occ(x, s_size, a_size=None, grid_size=None, n_clusters=2, percentile=70, strat='max', n_picks=1, rng=None):
    if(len(x.shape) == 2):
        x = np.sum(x, axis=1)
    _filter = np.percentile(x, percentile, axis=0)
    x_occ = np.zeros_like(x)
    x_occ[x >= _filter] = 1
    
    if(grid_size is not None):
        x = np.reshape(x, (grid_size, grid_size))
        x_occ = np.reshape(x_occ, (grid_size, grid_size))
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                if(x_occ[i,j] == 1):
                    coords.append([i,j])

        if(n_clusters == 0):
            centres = np.arange(1, min(x_occ.sum(), 5))
            n_clusters = np.random.choice(centres)

        #do kmeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
        clusters = kmeans.cluster_centers_
        labels = kmeans.labels_

        labels_rank = np.zeros(n_clusters)
        
        for _coords in coords:
            labels_rank[labels[coords.index(_coords)]] = max(labels_rank[labels[coords.index(_coords)]], x[_coords[0], _coords[1]])
        
        idxs = np.argsort(labels_rank)[::-1]
        
        x_filter = []
        for c in idxs: 
            x_filter_ = np.zeros((grid_size, grid_size))
            coords_idx = np.where(labels == c)

            coords_cluster = np.array(coords)[coords_idx]
            
            for coords_ in coords_cluster:
                x_filter_[coords_[0], coords_[1]] = x[coords_[0], coords_[1]]
            
            x_filter.append(x_filter_.flatten())

        x_filter = np.array(x_filter)
        if(strat == 'max'):
            x_filter = np.sum(x_filter[0:n_picks], axis=0)

        elif(strat == 'random'):
            assert rng is not None
            x_filter = np.sum(rng.choice(x_filter, n_picks, replace=False, axis=0), axis=0)
        
        return x_filter
    
    else:
        if(strat=='random'):
            assert rng is not None
            idxs = np.where(x_occ > 0)[0]
            idxs = rng.choice(idxs, n_picks, replace=False)
            x_filter = np.zeros_like(x)
            x_filter[idxs] = x[idxs]

        elif(strat=='max'):
            idxs = np.argsort(x)[::-1]
            x_filter = np.zeros_like(x)
            x_filter[idxs[0:n_picks]] = x[idxs[0:n_picks]]

        return x_filter

        
            
            



            


