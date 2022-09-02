import os
from hidio.hidio_scratch.networks import ActorNetwork, CriticNetwork, ValueNetwork
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import SchedulerBuffer, WorkerReplayBuffer
from networks import SchedulerNetwork, DiscriminatorNetwork


# Needs entropy modulation term alpha
class WorkerAgent(object):
    """
    For SAC
    """

    def __init__(self, 
                 env, 
                 max_memory_size, 
                 reward_scale, 
                 batch_size, 
                 polyak_coeff, 
                 checkpoint_dir, 
                 option_interval, 
                 skill_dims, 
                 gamma,
                 alpha,
                 use_auto_entropy_adjustment, 
                 learning_rate=10**-4, 
                 beta=0.01):

        self.max_memory_size = max_memory_size
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.polyak_coeff = polyak_coeff
        self.checkpoint_dir = checkpoint_dir
        self.option_interval = option_interval
        self.skill_dims = skill_dims
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.beta = beta

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
                                                 name="target_value_network", 
                                                 checkpoint_dir=self.checkpoint_dir, 
                                                 input_dims=self.observation_space_dims, 
                                                 fc1_size=256, 
                                                 fc2_size=256, 
                                                 output_dims=1)

        # Entropy adjustment factor (alpha)
        self.alpha = alpha
        self.use_auto_entropy_adjustment = use_auto_entropy_adjustment
        self.target_entropy = -T.prod(T.Tensor([env.action_space.high - env.action_space.low])).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor_network.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Align parameters of value_network and target_value_network
        self.update_target_value_network_params(polyak_coeff=1)

    
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


    def adjust_alpha(self, log_prob):
        """
        Automatically adjusts the value of alpha to provide a good balance of
        exploration and exploitation.
        """

        if self.use_auto_entropy_adjustment:
            alpha_loss = (self.log_alpha * (-log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        return

    
    def remember(self, state_array, action_array, next_state_array, skill, reward_array):
        """
        
        """

        self.memory.store_transitions(state_array, action_array, next_state_array, skill, reward_array)

        return

    
    def choose_action(self, state, skill):
        """
        state: numpy array of dims (n_elems,)
        skill: numpy array of dims (m_elems,)
        """

        state = T.tensor(state).reshape(1,-1).to(self.actor_network.device)
        skill = T.tensor(skill).reshape(1,-1).to(self.actor_network.device)
        action, log_probs = self.actor_network.sample_distribution(states_array=state, skills_array=skill, reparameterize=False)

        action = action.cpu().detach().numpy().squeeze(axis=0)
        log_probs = log_probs.cpu().detach().numpy().squeeze(axis=0)

        return action, log_probs


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

    
    def compute_q_val(self, states, skills, reparameterize):
        """
        
        """

        sampled_actions, log_probs = self.actor_network.sample_distribution(states_array=states, 
                                                                            skills_array=skills, 
                                                                            reparameterize=reparameterize)
        log_probs = log_probs.view(-1)
        q1_policy = self.critic_network_1.forward(state_array=states, action_array=sampled_actions)
        q2_policy = self.critic_network_2.forward(state_array=states, action_array=sampled_actions)
        critic_value = T.min(q1_policy, q2_policy).view(-1)

        return critic_value, log_probs


    def transfer_network_params(self, source_network, target_network):
        """
        Transfers network parameters from source_network to the target_network.
        """

        pass


    def learn(self, discriminator_output):
        """
        
        """

        # To allow build up of memory in replay buffer
        #if self.memory.memory_counter < self.batch_size:
        #    return

        total_reward = 0

        states_sample, actions_sample, next_states_sample, skills_sample, done_sample = self.memory.sample_buffer(self.batch_size)
        states_sample = T.tensor(states_sample, dtype=T.float32).to(self.actor_network.device)
        actions_sample = T.tensor(actions_sample, dtype=T.float32).to(self.actor_network.device)
        next_states_sample = T.tensor(next_states_sample, dtype=T.float32).to(self.actor_network.device)
        skills_sample = T.tensor(skills_sample, dtype=T.float32).to(self.actor_network.device)
        done_sample = T.tensor(done_sample).to(self.actor_network.device)

        for i in range(self.option_interval):
            # Samples from each interaction between worker and environment
            states = states_sample[:, i, :]
            actions = actions_sample[:, i, :]
            next_states = next_states_sample[:, i, :]
            skills = skills_sample
            done = done_sample[:, i, :].view(-1)

            # Initializing the value networks
            value_network_value = self.value_network.forward(states).view(-1)
            target_value_network_value = self.target_value_network.forward(next_states).view(-1)
            target_value_network_value[done] = 0.0

            # Update value_network
            critic_value, log_probs = self.compute_q_val(states=states, skills=skills, reparameterize=False)
            self.value_network.optimizer.zero_grad()
            value_target = critic_value - log_probs
            value_loss = 0.5 * F.mse_loss(value_network_value, value_target)
            value_loss.backward(retain_graph=True)
            self.value_network.optimizer.step()

            # Should the actor_network losses be backpropagated along with another final update?
            # The presence of an entropy weighting factor would suggest yes
            # What is entropy in the actor loss?
                # log_probs term is the entropy factor
            # Where is the entropy weighting factor added?
                # Multiplied by the log_probs term
            # Entropy weighting term seems to be extremely important for good performance
            critic_value, log_probs = self.compute_q_val(states=states, skills=skills, reparameterize=True)
            actor_loss = T.mean(log_probs - critic_value)
            self.actor_network.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_network.optimizer.step()

            # Reward calculation
            reward = discriminator_output - self.beta * log_probs
            total_reward = total_reward + reward

            # Critic network updates
            self.critic_network_1.optimizer.zero_grad()
            self.critic_network_2.optimizer.zero_grad()
            q_hat = (self.reward_scale * reward) + (self.gamma * target_value_network_value)
            q1_old_policy = self.critic_network_1.forward(state_array=states, action_array=actions).view(-1)
            q2_old_policy = self.critic_network_2.forward(state_array=states, action_array=actions).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward()
            self.critic_network_1.optimizer.step()
            self.critic_network_2.optimizer.step()

            # Update target_value_network
            self.update_target_value_network_params()

            # Update alpha
            self.adjust_alpha(log_prob=log_probs)
            
        return total_reward



class Agent(object):
    """
    For HRL
    """
    
    def __init__(self, 
                 env, 
                 feature, 
                 skill_dims, 
                 learning_rate, 
                 memory_size, 
                 checkpoint_dir, 
                 option_interval, 
                 gamma, 
                 option_gamma, 
                 episode_length, 
                 reward_scale, 
                 batch_size, 
                 polyak_coeff, 
                 beta, 
                 alpha, 
                 use_auto_entropy_adjustment, 
                 min_entropy_target, 
                 w_alpha, 
                 w_auto_entropy_adjustment):

        # Environment description attributes
        self.env_name = env.spec.id
        self.num_actions = env.action_space.shape[0]
        self.observation_space_dims = env.observation_space.shape[0]

        # Entropy modulating terms
        self.alpha = alpha
        self.use_auto_entropy_adjustment = use_auto_entropy_adjustment
        self.min_entropy_target = min_entropy_target 
        self.target_entropy = -T.prod(T.Tensor([env.action_space.high - env.action_space.low])).item()
        self.log_alpha = T.ones(1, requires_grad=True, device=self.actor_network.device) * self.min_entropy_target
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        self.w_alpha = w_alpha
        self.w_auto_entropy_adjustment = w_auto_entropy_adjustment


        self.skill_dims = skill_dims
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.checkpoint_dir = checkpoint_dir
        self.option_interval = option_interval
        self.gamma = gamma
        self.option_gamma = option_gamma
        self.episode_length = episode_length
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.polyak_coeff = polyak_coeff
        self.beta = beta

        self.feature = feature
        if (self.feature == "state") or (self.feature == "stateDiff"):
            self.discriminator_input_size = self.observation_space_dims
        
        elif (self.feature == "action") or (self.feature == "stateAction"):
            self.discriminator_input_size = self.observation_space_dims + self.num_actions

        elif self.feature == "stateConcat":
            self.discriminator_input_size = self.observation_space_dims * self.option_interval

        elif self.feature == "actionConcat":
            self.discriminator_input_size = self.observation_space_dims + (self.num_actions * self.option_interval)

        else:
            raise ValueError("Input values can only be 'state', 'action', 'stateDiff', 'stateAction', 'stateConcat', 'actionConcat'")

        
        self.worker = WorkerAgent(env=env, 
                                  max_memory_size=self.memory_size, 
                                  reward_scale=self.reward_scale, 
                                  batch_size=self.batch_size, 
                                  polyak_coeff=self.polyak_coeff, 
                                  checkpoint_dir=self.checkpoint_dir, 
                                  option_interval=self.option_interval, 
                                  skill_dims=self.skill_dims, 
                                  gamma=self.gamma, 
                                  learning_rate=self.learning_rate, 
                                  beta=self.beta, 
                                  alpha=self.w_alpha,
                                  use_auto_entropy_adjustment=self.w_auto_entropy_adjustment)
        self.scheduler = SchedulerNetwork(env_name=self.env_name, 
                                          learning_rate=self.learning_rate, 
                                          name="scheduler_network", 
                                          checkpoint_dir=self.checkpoint_dir, 
                                          input_dims=self.observation_space_dims, 
                                          fc1_size=256, 
                                          fc2_size=256, 
                                          output_dims=self.skill_dims, 
                                          option_interval=self.option_interval, 
                                          gamma=self.gamma, 
                                          option_gamma=self.option_gamma)
        self.scheduler_memory = SchedulerBuffer(memory_size=self.memory_size, 
                                                skill_dims=self.skill_dims, 
                                                state_dims=self.observation_space_dims, 
                                                num_actions=self.num_actions)
        self.discriminator = DiscriminatorNetwork(env_name=self.env_name, 
                                                  learning_rate=self.learning_rate, 
                                                  name="discriminator_network", 
                                                  checkpoint_dir=self.checkpoint_dir, 
                                                  input_dims=self.discriminator_input_size, 
                                                  fc1_size=64, 
                                                  fc2_size=64, 
                                                  output_dims=self.skill_dims)
        self.value_network = ValueNetwork(env_name=self.env_name, 
                                          learning_rate=self.learning_rate, 
                                          name="scheduler_value_network", 
                                          checkpoint_dir=self.checkpoint_dir, 
                                          input_dims=self.observation_space_dims, 
                                          fc1_size=256, 
                                          fc2_size=256, 
                                          output_dims=1)
        self.target_value_network = ValueNetwork(env_name=self.env_name, 
                                                 learning_rate=self.learning_rate, 
                                                 name="scheduler_target_value_network", 
                                                 checkpoint_dir=self.checkpoint_dir, 
                                                 input_dims=self.observation_space_dims, 
                                                 fc1_size=256, 
                                                 fc2_size=256, 
                                                 output_dims=1)
        self.critic_network_1 = CriticNetwork(env_name=self.env_name, 
                                              learning_rate=self.learning_rate, 
                                              name="scheduler_critic_network_1", 
                                              checkpoint_dir=self.checkpoint_dir, 
                                              input_dims=self.observation_space_dims + self.skill_dims, 
                                              fc1_size=256, 
                                              fc2_size=256, 
                                              output_dims=1)
        self.critic_network_2 = CriticNetwork(env_name=self.env_name, 
                                              learning_rate=self.learning_rate, 
                                              name="scheduler_critic_network_2", 
                                              checkpoint_dir=self.checkpoint_dir, 
                                              input_dims=self.observation_space_dims + self.skill_dims, 
                                              fc1_size=256, 
                                              fc2_size=256, 
                                              output_dims=1)
        
        # Align parameters of value_network and target_value_network
        self.update_target_value_network_params(polyak_coeff=1)


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

    
    def adjust_alpha(self, log_prob):
        """
        Used to adjust the entropy modulating term alpha.
        """

        if self.use_auto_entropy_adjustment:
            alpha_loss = (self.log_alpha * (-log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        return


    def generate_skill(self, state):
        """
        state: numpy array of (n_elems,)
        """

        state = T.tensor(state).reshape(1, -1).to(self.scheduler.device)
        skill, _ = self.scheduler.sample_skill(state=state, reparameterize=False)

        return skill.cpu().detach().numpy().squeeze(axis=0)


    def remember(self, state, skill, next_state, reward):
        """
        state: numpy array of (n_elems,)
        skill: numpy array of (m_elems,)
        next_state: numpy array of (n_elems,)
        reward: float
        """

        self.scheduler_memory.store_transitions(state, skill, next_state, reward)
        
        return


    def save_models(self):
        """
        
        """

        print("************--Saving scheduler and discriminator--************")
        self.scheduler.save_checkpoint()
        self.value_network.save_checkpoint()
        self.target_value_network.save_checkpoint()
        self.critic_network_1.save_checkpoint()
        self.critic_network_2.save_checkpoint()
        self.discriminator.save_checkpoint()
        self.worker.save_models()

        return

    
    def load_models(self):
        """
        """

        print("************--Loading scheduler and discriminator--************")
        self.scheduler.load_checkpoint()
        self.value_network.load_checkpoint()
        self.target_value_network.load_checkpoint()
        self.critic_network_1.load_checkpoint()
        self.critic_network_2.load_checkpoint()
        self.discriminator.load_checkpoint()
        self.worker.load_models()

        return


    def learn(self):
        """
        
        """

        states_sample, skill_sample, next_states_sample, rewards_sample, batch = self.scheduler_memory.sample_buffer(self.batch_size)
        states_sample = T.tensor(states_sample, dtype=T.float32).to(self.scheduler.device)
        skill_sample = T.tensor(skill_sample, dtype=T.float32).to(self.scheduler.device)
        next_states_sample = T.tensor(next_states_sample, dtype=T.float32).to(self.scheduler.device)
        rewards_sample = T.tensor(rewards_sample, dtype=T.float32).to(self.scheduler.device)

        # Update target_value_network
        self.update_target_value_network_params()
        
        # Update alpha
        self.adjust_alpha(log_prob=-------------)

        return

