import os
from hidio.hidio_scratch.networks import ActorNetwork, CriticNetwork, ValueNetwork
import numpy as np
import torch as T
import torch.nn.functional as F
from replay_buffer import SchedulerBuffer, WorkerReplayBuffer
from networks import SchedulerNetwork, DiscriminatorNetwork


class WorkerAgent(object):
    """
    For SAC
    """

    def __init__(self, env, 
                       max_memory_size, 
                       reward_scale, 
                       batch_size, 
                       polyak_coeff, 
                       checkpoint_dir, 
                       option_interval, 
                       skill_dims, 
                       gamma, 
                       learning_rate):

        self.max_memory_size = max_memory_size
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.polyak_coeff = polyak_coeff
        self.checkpoint_dir = checkpoint_dir
        self.env_name = env
        self.option_interval = option_interval
        self.skill_dims = skill_dims
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.env_name = env.spec.id
        self.num_actions = env.action_space.shape[0]
        self.observation_space_dims = env.observation_space.shape[0]

        self.memory = WorkerReplayBuffer(option_interval=self.option_interval, 
                                         memory_size=self.max_memory_size, 
                                         state_dims=self.observation_space_dims, 
                                         skill_dims=self.skill_dims, 
                                         num_actions=self.num_actions)
        self.actor_network = ActorNetwork(env_name=self.env_name, 
                                          learning_rate=self.learning_rate, 
                                          name="actor_network", 
                                          checkpoint_dir=self.checkpoint_dir, 
                                          input_dims=self.observation_space_dims + self.skill_dims, 
                                          fc1_size=256, 
                                          fc2_size=256, 
                                          output_dims=self.num_actions,
                                          max_action=env.action_space.high)
        self.critic_network_1 = CriticNetwork(env_name=self.env_name, 
                                              learning_rate=self.learning_rate, 
                                              name="critic_network_1", 
                                              checkpoint_dir=self.checkpoint_dir, 
                                              input_dims=self.observation_space_dims + self.num_actions, 
                                              fc1_size=256, 
                                              fc2_size=256, 
                                              output_dims=1)
        self.critic_network_2 = CriticNetwork(env_name=self.env_name, 
                                              learning_rate=self.learning_rate, 
                                              name="critic_network_2", 
                                              checkpoint_dir=self.checkpoint_dir, 
                                              input_dims=self.observation_space_dims + self.num_actions, 
                                              fc1_size=256, 
                                              fc2_size=256, 
                                              output_dims=1)
        self.value_network = ValueNetwork(env_name=self.env_name, 
                                          learning_rate=self.learning_rate, 
                                          name="value_network", 
                                          checkpoint_dir=self.checkpoint_dir, 
                                          input_dims=self.observation_space_dims, 
                                          fc1_size=256, 
                                          fc2_size=256, 
                                          output_dims=1)
        self.target_value_network = ValueNetwork(env_name=self.env_name, 
                                                 learning_rate=self.learning_rate, 
                                                 name="value_network", 
                                                 checkpoint_dir=self.checkpoint_dir, 
                                                 input_dims=self.observation_space_dims, 
                                                 fc1_size=256, 
                                                 fc2_size=256, 
                                                 output_dims=1)

        self.update_target_value_network_params(polyak_coeff)

    
    def update_target_value_network_params(self, polyak_coeff):
        """
        
        """

        if polyak_coeff is None:
            polyak_coeff = self.polyak_coeff

        target_value_network_state_dict = dict(self.target_value_network.named_parameters())
        value_network_state_dict = dict(self.value_network.named_parameters())

        for key in value_network_state_dict:
            value_network_state_dict[key] = (polyak_coeff * value_network_state_dict[key].clone()) + ((1-polyak_coeff) * target_value_network_state_dict[key].clone())

        self.target_value_network.load_state_dict(value_network_state_dict)

        return

    
    def remember(self, state_array, action_array, next_state_array, skill, reward_array):
        """
        
        """

        self.memory.store_transitions(state_array, action_array, next_state_array, skill, reward_array)

        return


    def save_models(self):
        """
        
        """

        print("############--Saving worker models--############")
        self.actor_network.save_checkpoint()
        self.critic_network_1.save_checkpoint()
        self.critic_network_2.save_checkpoint()
        self.target_value_network.save_checkpoint()
        self.value_network.save_checkpoint()

        return

    
    def load_models(self):
        """
        
        """        

        print("############--Loading worker models--############")
        self.actor_network.load_checkpoint()
        self.critic_network_1.load_checkpoint()
        self.critic_network_2.load_checkpoint()
        self.target_value_network.load_checkpoint()
        self.value_network.load_checkpoint()

        return


    def transfer_network_params(self, source_network, target_network):
        """
        Transfers network parameters from source_network to the target_network.
        """

        pass


    def learn(self, discriminator_output):
        """
        
        """

        # Why??
        #if self.memory.memory_counter < self.batch_size:
        #    return

        loss = 0
        actions_array, next_states_sample, skills_sample = self.memory.sample_buffer(self.batch_size)
        actions_array = T.tensor(actions_array, dtype=T.float32).to(self.actor_network.device)
        next_states_sample = T.tensor(next_states_sample, dtype=T.float32).to(self.actor_network.device)
        skills_sample = T.tensor(skills_sample, dtype=T.float32).to(self.actor_network.device)

        for i in range(self.option_interval):


        
        return


class Agent(object):
    """
    For HRL
    """
    
    def __init__(self, env, q_feature):


        pass

