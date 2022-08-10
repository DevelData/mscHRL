import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    """
    
    """
    
    def __init__(self, 
                 critic_lr, 
                 input_dims, 
                 num_actions, 
                 name="critic_network", 
                 fully_connected_dims_1=256, 
                 fully_connected_dims_2=256, 
                 checkpoint_dir="./network_checkpoints/"):
        
        super(CriticNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.name = name
        self.fully_connected_dims_1 = fully_connected_dims_1
        self.fully_connected_dims_2 = fully_connected_dims_2
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_sac")
        
        # NN architecture
        self.fc1 = nn.Linear(in_features=input_dims[0] + num_actions, 
                             out_features=self.fully_connected_dims_1)
        self.fc2 = nn.Linear(in_features=self.fully_connected_dims_1, 
                             out_features=self.fully_connected_dims_2)
        self.q_val = nn.Linear(in_features=self.fully_connected_dims_2, 
                               out_features=1)
        
        # Optimizer and device settings
        self.optimizer = optim.Adam(params=self.parameters(), 
                                    lr=critic_lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        
        
    def forward(self, state, action):
        action_value = F.relu(self.fc1(T.cat([state, action]), dim=1))
        action_value = F.relu(self.fc2(action_value))
        q = self.q_val(action_value)

        return q


    def save_checkpoint(self):
        """

        """
        T.save(self.state_dict(), self.checkpoint_file)

        return

    def load_checkpoint(self):
        """

        """
        self.load_state_dict(T.load(self.checkpoint_file))

        return
        

        
class ValueNetwork(nn.Module):
    """
    
    """
    
    def __init__(self, 
                 value_lr, 
                 input_dims, 
                 name="value_network", 
                 fully_connected_dims_1=256, 
                 fully_connected_dims_2=256, 
                 checkpoint_dir="./network_checkpoints/"):
        
        super(ValueNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fully_connected_dims_1 = fully_connected_dims_1
        self.fully_connected_dims_2 = fully_connected_dims_2
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_sac")
        
        # NN architecture
        # Monitor the unpacking
        self.fc1 = nn.Linear(in_features=self.input_dims[0], 
                             out_features=self.fully_connected_dims_1)
        self.fc2 = nn.Linear(in_features=self.fully_connected_dims_1, 
                             out_features=self.fully_connected_dims_2)
        self.value = nn.Linear(in_features=self.fully_connected_dims_2, 
                               out_features=1)
        
        # Optimizer and device settings
        self.optimizer = optim.Adam(self.parameters(), lr=value_lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        
        
    def forward(self, state):
        """

        """

        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        value = self.value(state_value)

        return value


    def save_checkpoint(self):
        """

        """
        T.save(self.state_dict(), self.checkpoint_file)

        return


    def load_checkpoint(self):
        """

        """
        self.load_state_dict(T.load(self.checkpoint_file))

        return
    
    
    
class ActorNetwork(nn.Module):
    """
    
    """
    
    def __init__(self, 
                 actor_lr,
                 input_dims,
                 max_action,
                 num_actions=2,
                 name="actor_network",
                 fully_connected_dims_1=256,
                 fully_connected_dims_2=256,
                 checkpoint_dir="./network_checkpoints/"):
        
        super(ActorNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.name = name
        self.max_action = max_action
        self.num_actions = num_actions
        self.fully_connected_dims_1 = fully_connected_dims_1
        self.fully_connected_dims_2 = fully_connected_dims_2
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_sac")
        # To prevent taking log of 0
        self.reparameterization_noise = 1e-6
        
        # NN architecture
        # Monitor the unpacking
        self.fc1 = nn.Linear(in_features=self.input_dims[0], 
                             out_features=self.fully_connected_dims_1)
        self.fc2 = nn.Linear(in_features=self.fully_connected_dims_1, 
                             out_features=self.fully_connected_dims_2)
        self.mu = nn.Linear(in_features=self.fully_connected_dims_2, 
                            out_features=self.num_actions)
        self.sigma = nn.Linear(in_features=self.fully_connected_dims_2, 
                               out_features=self.num_actions)
        
        # Optimizer and device settings
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        
    
    def forward(self, state):
        """
        
        """
        
        probability = F.relu(self.fc1(state))
        probability = F.relu(self.fc2(probability))
        mu = self.mu(probability)
        sigma = self.sigma(probability)
        
        # Authors of original paper used values between -20 and 2 for clamping.
        # Could do the same, but substitute 0 with really small value.
        # Try tutorial implementation first
        sigma = T.clamp(sigma, min=self.reparameterization_noise, max=1)
        
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        #print("mu:", mu)
        #print("sigma:", sigma)
        #print("probabilities:", probabilities)
        
        if reparameterize:
            # Sample from distribution with noise
            action_samples = probabilities.rsample()
        else:
            # Sample from distribution without noise
            action_samples = probabilities.sample()
            
        action = T.tanh(action_samples) * T.tensor(self.max_action).to(self.device)
        log_probability = probabilities.log_prob(action_samples)
        log_probability = log_probability - T.log(1 - action.pow(2) + self.reparameterization_noise)
        log_probability = log_probability.sum(1, keepdim=True)
        
        return action, log_probability
    
    
    def save_checkpoint(self):
        """

        """
        T.save(self.state_dict(), self.checkpoint_file)

        return


    def load_checkpoint(self):
        """

        """
        self.load_state_dict(T.load(self.checkpoint_file))

        return