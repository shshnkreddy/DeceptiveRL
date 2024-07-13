import torch 
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F

class detPolicyNet(torch.nn.Module):
    def __init__(self, pi, n_actions, obs_to_state_map=None, hidden=False):
        super(detPolicyNet, self).__init__()
        self.pi = pi
        self.obs_to_state_map = obs_to_state_map
        self.n_actions = n_actions
        self.actions_array = np.arange(self.n_actions)
        self.hidden = hidden

    def forward(self, obs, states, dones):
        batch_size = obs.shape[0]
        actions = np.zeros(batch_size, dtype = int)
        if(self.hidden==False):
            for i in range(batch_size):
                state = self.obs_to_state_map[str(obs[i])]
                actions[i] = np.random.choice(self.actions_array, 1, p=self.pi[state])

        else:
            for i in range(batch_size):
                state = int(obs[i])
                actions[i] = np.random.choice(self.actions_array, 1, p=self.pi[state])
            # actions[i] = self.pi[states[i]]
        # print(actions)
        return actions, states
    __call__ = forward

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        actions = self.forward(obs)
        return actions, None

class NNPolicy(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, layers_data=None):
        super(NNPolicy, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.layers = nn.ModuleList()

        input_size = obs_dim
        if layers_data is not None:
            for size, activation in layers_data:
                self.layers.append(nn.Linear(input_size, size))
                input_size = size  # For the next layer
                if activation is not None:
                    assert isinstance(activation, torch.nn.Module), \
                        "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                    self.layers.append(activation)

        self.layers.append(nn.Linear(input_size, action_dim))
        
        self.softmax = nn.Softmax(dim=1)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        print('Architecture:', self.layers)

    def forward(self, obs):
        for layer in self.layers:
            obs = layer(obs)
        action = self.softmax(obs)
        return action

class WDNNReward(torch.nn.Module):
    def __init__(self, obs_dim, layers_data):
        super(WDNNReward, self).__init__()
        
        self.obs_dim = obs_dim
        
        self.layers = nn.ModuleList()

        input_size = self.obs_dim

        if layers_data is not None:
            for size, activation in layers_data:
                self.layers.append(spectral_norm(nn.Linear(input_size, size)))
                input_size = size  # For the next layer
                if activation is not None:
                    assert isinstance(activation, torch.nn.Module), \
                        "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                    self.layers.append(activation)

        self.layers.append(spectral_norm(nn.Linear(input_size, 1)))
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        print('Architecture:', self.layers)

    def forward(self, obs):
        for layer in self.layers:
            obs = layer(obs)
        return obs

class DoubleQArgmax(torch.nn.Module):
    def __init__(self, net1, net2, lambda_, device):
        super(DoubleQArgmax, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.lambda_ = lambda_
        self.device = device
        
    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            values1 = self.net1(obs)
            # values1 = F.normalize(values1, dim=1)
            values2 = self.net2(obs)
            # values2 = F.normalize(values2, dim=1)
            
            actions = torch.argmax(self.lambda_*values1 + values2, dim=1, keepdim=False)
        #action = argmax of values1 + lambda2 * values2
        actions = actions.cpu().numpy()
        
        return actions
        __call__ = forward

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        actions = self.forward(obs)
        return actions, None


