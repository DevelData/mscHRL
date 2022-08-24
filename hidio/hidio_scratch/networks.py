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
                 skill_dims,
                 option_interval, 
                 gamma, 
                 option_gamma, 
                 max_action, 
                 episode_length):

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
        self.skill_dims = skill_dims # Not sure about using this yet
        self.option_gamma = option_gamma
        self.max_action = max_action
        self.episode_length = episode_length
        # To prevent samples with zero standard deviation (non-differentiable)
        self.reparameterization_noise = 1e-6

        # Array for discounting in the loss function
        self.option_interval_discounting = np.full(self.option_interval, self.option_gamma)
        self.option_interval_discounting = np.power(self.option_interval_discounting, [i for i in range(self.option_interval)])

        # Additional layer for calculation of standard deviation
        #self.sigma = nn.Linear(in_features=self.fc2_size, 
        #                       out_features=self.output_dims)


    def forward(self, state):

        # Don't forget to send the state to the Tensor device in other methods
        #state = T.tensor(state).to(self.device)
        processed_state = F.relu(self.fc1(state))
        processed_state = F.relu(self.fc2(state))
        
        # Trying this instead of separate layers 
        mu_sigma = self.output(processed_state)
        print("Shape of mu_sigma:", mu_sigma.shape)
        mu = mu_sigma[:, 0]
        sigma = mu_sigma[:, 1]
        #sigma = self.sigma(processed_state)

        # To clamp or not to clamp ??????????????
        # Probably should clamp - max value is picked from SAC tutorial
        sigma = T.clamp(sigma, min=self.reparameterization_noise, max=1)

        return mu, sigma

    
    def sample_normal(self, state, reparameterize=True):
        """
        
        """

        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # Find out why it was in the original tutorial
        # Something about sampling with noise
        if reparameterize:
            # Sample from distribution with noise
            action_samples = probabilities.rsample()
        else:
            # Sample from distribution without noise
            action_samples = probabilities.sample()

        # Is the tanh necessary ??
        # OpenAI Gym website is down
        # Perhaps needed for symmetric action space - confirm if continuous 
        # actions are symmetric
        actions = T.tanh(action_samples) * T.tensor(self.max_action).to(self.device)
        actions = actions.reshape(self.skill_dims, -1)

        # This I do not understand at all - FIGURE THIS OUT!!!
        #log_probability = probabilities.log_prob(action_samples)
        #log_probability = log_probability - T.log(1 - action.pow(2) + self.reparameterization_noise)
        #log_probability = log_probability.sum(1, keepdim=True)

        return actions


    def post_interval_reward(self, reward_array, option_interval_discounted=False):
        """
        Calculates the reward for the scheduler after the option interval (K).
        """

        if option_interval_discounted:
            return (reward_array * self.option_interval_discounting).sum()
        else:
            return reward_array.sum()
        
    
    def objective_rewards(self, )
        


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
                 output_dims, 
                 normal_mu, 
                 normal_std, 
                 skill_dims,
                 batch_size):
        #self, env_name, learning_rate, dimensions, normal_mu, normal_std, name, checkpoint_dir, fc1_size=64, fc2_size=64):
        
        super(DiscriminatorNetwork, self).__init__(self, 
                                                   env_name,
                                                   learning_rate, 
                                                   name, 
                                                   checkpoint_dir, 
                                                   input_dims, 
                                                   fc1_size, 
                                                   fc2_size, 
                                                   output_dims)
        # Used for sampling - output distribution is assumed to be Gaussian
        self.normal_mu = normal_mu
        self.normal_std = normal_std
        self.skill_dims = skill_dims
        self.batch_size = batch_size


    def forward(self, input_array):
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

        return predicted_skill


    def compute_loss(self, input_array, skill):

        predicted_skill = self.forward(input_array=input_array)\
                              .reshape(self.batch_size, self.skill_dims, -1)

        return F.mse_loss(predicted_skill, skill).pow(2)



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
                 num_actions, 
                 max_action):

        super(ActorNetwork, self).__init__(env_name,
                                           learning_rate,
                                           name, 
                                           checkpoint_dir, 
                                           input_dims, 
                                           fc1_size, 
                                           fc2_size, 
                                           output_dims)
        self.num_actions = num_actions
        self.max_action = max_action

        # To prevent taking log of 0
        self.reparameterization_noise = 1e-6

        
    def forward(self, state):
        """
        """

        probability = F.relu(self.fc1(state))
        probability = F.relu(self.fc2(probability))
        mu_sigma = mu_sigma = self.output(probability)
        print("Shape of mu_sigma:", mu_sigma.shape)
        mu = mu_sigma[:, 0]
        sigma = mu_sigma[:, 1]

        sigma = T.clamp(sigma, min=self.reparameterization_noise, max=1)

        return mu, sigma

    def choose_action(self, state, reparameterize=True):
        """
        """

        mu, sigma = self.forward(state)
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