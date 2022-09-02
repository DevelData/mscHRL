import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class GeneralNetwork(nn.Module):
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
                 output_dims):
        
        super(GeneralNetwork, self).__init__()
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.name = name
        self.checkpoint_dir = checkpoint_dir + self.env_name
        self.input_dims = input_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.output_dims = output_dims

        # File in which to save the parameters
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        # NN architecture
        self.fc1 = nn.Linear(in_features=input_dims, 
                             out_features=self.fc1_size)
        self.fc2 = nn.Linear(in_features=self.fc1_size, 
                             out_features=self.fc2_size)
        self.output = nn.Linear(in_features=self.fc2_size, 
                                out_features=self.output_dims)

        # Optimizer and device settings
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def save_checkpoint(self):
        """
        """

        T.save(self.state_dict, self.checkpoint_file)

        return

    
    def load_checkpoint(self, custom_state_dict=None):
        """
        """

        if custom_state_dict is not None:
            self.load_state_dict(T.load(custom_state_dict))
        else:
            self.load_state_dict(T.load(self.checkpoint_file))

        return



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
                 gamma, 
                 option_gamma):

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
        self.option_gamma = option_gamma
        # To prevent samples with zero standard deviation (non-differentiable)
        self.reparameterization_noise = 1e-6

        # Array for discounting in the loss function
        self.option_interval_discount = np.full(self.option_interval, self.option_gamma)
        self.option_interval_discount = np.power(self.option_interval_discount, [i for i in range(self.option_interval)])
        self.option_interval_discount = T.tensor(self.option_interval_discount).to(self.device)

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
        processed_state = F.relu(self.fc2(state))
        
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


    def post_interval_reward(self, log_probs, reward_array, expected_value=True):
        """
        Calculates the reward for the scheduler after the option interval (K).
        1. Find out the dimensions for log_prob_actions
            Same as reward - log_prob is actually log likelihood.
        
        log_probs: from actor network of worker module
            Type: numpy array
            Size: 1 x option_interval
        reward_array: from environment
            Type: numpy array
            Size: 1 x option_interval
        """

        if expected_value:
            rewards = reward_array * self.option_interval_discount
            final_reward = log_probs * rewards
            return final_reward.mean().item()

        else:
            return reward_array.sum().item()
        
    
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



class DiscriminatorNetwork(GeneralNetwork):
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
                 output_dims):
        
        super(DiscriminatorNetwork, self).__init__(self, 
                                                   env_name,
                                                   learning_rate, 
                                                   name, 
                                                   checkpoint_dir, 
                                                   input_dims, 
                                                   fc1_size, 
                                                   fc2_size, 
                                                   output_dims)

    def forward(self, input_array, use_tanh=False):
        """
        input_array can be a combination of actions, states or next states.
        Make sure to preprocess the inputs before passing them through. Also,
        modify input_dims before using this function.
        """
        # Have to concatenate the inputs and output something that matches 
        # the dimensions of the skill.

        processed_input = F.relu(self.fc1(input_array))
        processed_input = F.relu(self.fc2(processed_input))
        predicted_skill = self.output(processed_input)

        if use_tanh:
            predicted_skill = T.tanh(predicted_skill)

        return predicted_skill


    def compute_loss(self, input_array, skill, use_tanh):
        """
        Check feature extractor table to know what to input in input_array.
        """

        predicted_skill = self.forward(input_array=input_array, use_tanh=use_tanh)

        return -1 * F.mse_loss(predicted_skill, skill)



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


    def sample_distribution(self, states_array, skills_array, reparameterize=True):
        """
        state_array: (N x observation_elems) tensor (device initialized)
        skills_array: (N x skill_dims) tensor (device initialized)
        """

        input_array = T.cat([states_array, skills_array], dim=1)

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



class ValueNetwork(GeneralNetwork):
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
                 output_dims):

        super(ValueNetwork, self).__init__(env_name, 
                                           learning_rate, 
                                           name, 
                                           checkpoint_dir, 
                                           input_dims, 
                                           fc1_size, 
                                           fc2_size, 
                                           output_dims)

        
    def forward(self, state):
        """
        """

        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state))
        value = self.output(state_value)

        return value



class CriticNetwork(GeneralNetwork):
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
                 output_dims):

        super(CriticNetwork, self).__init__(env_name, 
                                            learning_rate, 
                                            name, 
                                            checkpoint_dir, 
                                            input_dims, 
                                            fc1_size, 
                                            fc2_size, 
                                            output_dims)


    def forward(self, state_array, action_array):
        """
        """

        state_action_array = T.cat([state_array, action_array], dim=1)
        action_value = F.relu(self.fc1(state_action_array))
        action_value = F.relu(self.fc2(action_value))
        q_value = self.output(action_value)

        return q_value