from general_network import GeneralNetwork
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal



class SchedulerNetwork(GeneralNetwork):
    """
    
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
                 option_interval, 
                 gamma):

        super(SchedulerNetwork, self).__init__(env_name, 
                                               learning_rate, 
                                               name, 
                                               checkpoint_dir, 
                                               input_dims, 
                                               fc1_size, 
                                               fc2_size, 
                                               output_dims)
        self.option_interval = option_interval
        self.gamma = gamma
        # To prevent samples with zero standard deviation (non-differentiable)
        self.reparameterization_noise = 1e-6

        # Additional layer for calculation of standard deviation
        #self.sigma = nn.Linear(in_features=self.fc2_size, 
        #                       out_features=self.output_dims)


    def forward(self, state):
        """
        mu_sigma dimensions are incorrect, even in SAC ActorNetwork
        """

        # Don't forget to send the state to the Tensor device in other methods
        #state = T.tensor(state).to(self.device)
        processed_state = F.relu(self.fc1(state))
        processed_state = F.relu(self.fc2(processed_state))
        
        # Trying this instead of separate layers 
        mu_sigma = self.output(processed_state)
        mu = mu_sigma[:, :mu_sigma.shape[1]//2] # Could be source of error
        sigma = mu_sigma[:, mu_sigma.shape[1]//2:] # Could be source of error
        #sigma = self.sigma(processed_state)

        # Probably should clamp - max value is picked from SAC tutorial
        sigma = T.clamp(sigma, min=self.reparameterization_noise, max=1)

        return mu, sigma

    
    def sample_skill(self, state, reparameterize=True):
        """
        mu dims: batch_size x skill_dims
        sigma dims = batch_size x skill_dims
        
        """

        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # Find out why it was in the original tutorial
        # Something about sampling with noise
        if reparameterize:
            # Sample from distribution with noise
            skill_samples = probabilities.rsample()
        else:
            # Sample from distribution without noise
            skill_samples = probabilities.sample()

        # Skill elements have a value between -1 and 1 in paper
        skills = T.tanh(skill_samples)

        # This is to prevent zeros from being passed into the log function
        # Potential source of error
        #skill_samples.data[skill_samples.data.eq(0)] = self.reparameterization_noise

        # In the appendix - still some questions around the derivation.
        # Needed for change of variable
        log_probability = probabilities.log_prob(skill_samples)
        log_probability = log_probability - T.log(1 - skills.pow(2) + self.reparameterization_noise)
        log_probability = log_probability.sum(1, keepdim=True)

        return skills, log_probability

    
    # Check this out - FOCUS!!!
    def objective_rewards(self, log_prob, reward_array, batch_idx, horizon_discount=False):
        """
        Computes the loss value for the Scheduler.
        reward_array: from replay buffer of scheduler
        log_prob: from output of sample_normal of scheduler
        batch_idx: batch indices of the samples selected
        """

        discount_factors = np.full(batch_idx.shape, self.gamma)
        discount_factors = np.power(discount_factors, batch_idx)
        
        if horizon_discount:
            rewards = reward_array * discount_factors
            rewards = T.tensor(rewards).to(self.device)
            
        else:
            rewards = T.tensor(reward_array).to(self.device)

        final_reward = log_prob * rewards

        return final_reward.mean()