# networks/actor_network.py

from networks.general_network import GeneralNetwork
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal



class ActorNetwork(GeneralNetwork):
    """
    Actor network for SAC. Used in the worker module.
    """

    def __init__(self, 
                 env_name, 
                 learning_rate, 
                 name, 
                 checkpoint_dir, 
                 input_dims, 
                 fc1_size, 
                 fc2_size, 
                 output_dims,
                 max_action):

        super(ActorNetwork, self).__init__(env_name,
                                           learning_rate,
                                           name, 
                                           checkpoint_dir, 
                                           input_dims, 
                                           fc1_size, 
                                           fc2_size, 
                                           output_dims)
        self.max_action = max_action
        # To prevent taking log of 0
        self.reparameterization_noise = 1e-6

        
    def forward(self, input_array):
        """
        """

        probability = F.relu(self.fc1(input_array))
        probability = F.relu(self.fc2(probability))
        mu_sigma = self.output(probability)
        mu = mu_sigma[:, :mu_sigma.shape[1]//2] # Could be source of error
        sigma = mu_sigma[:, mu_sigma.shape[1]//2:] # Could be source of error

        sigma = T.clamp(sigma, min=self.reparameterization_noise, max=1)

        return mu, sigma


    def sample_distribution(self, states_array, skills_array=None, reparameterize=True):
        """
        state_array: (N x observation_elems) tensor (device initialized)
        skills_array: (N x skill_dims) tensor (device initialized)
        """
        if skills_array is not None:
            input_array = T.cat([states_array, skills_array], dim=1)
        else:
            input_array = states_array

        mu, sigma = self.forward(input_array) 
        probabilities = Normal(mu, sigma)
        
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
