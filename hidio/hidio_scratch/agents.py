import os
import copy
from networks import ActorNetwork, CriticNetwork, ValueNetwork, SchedulerNetwork, DiscriminatorNetwork
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import SchedulerBuffer, WorkerReplayBuffer
from pathlib import Path


# Needs entropy modulation term alpha
# Make checkpoint_dir if it doesn't exist
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
                 feature,
                 use_tanh,
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

        # Environment variables
        self.env_name = env.spec.id
        self.num_actions = env.action_space.shape[0]
        self.observation_space_dims = env.observation_space.shape[0]

        # Discriminator variables
        self.feature = feature
        self.use_tanh = use_tanh
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


        # Networks
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
                                          output_dims=self.num_actions * 2,
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
        self.discriminator = DiscriminatorNetwork(env_name=self.env_name, 
                                                  learning_rate=self.learning_rate, 
                                                  name="discriminator_network", 
                                                  checkpoint_dir=self.checkpoint_dir, 
                                                  input_dims=self.discriminator_input_size, 
                                                  fc1_size=64, 
                                                  fc2_size=64, 
                                                  output_dims=self.skill_dims)

        # Entropy adjustment factor (alpha)
        self.alpha = alpha
        self.use_auto_entropy_adjustment = use_auto_entropy_adjustment
        self.target_entropy = -T.prod(T.Tensor([env.action_space.high - env.action_space.low])).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor_network.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Align parameters of value_network and target_value_network
        self.update_target_value_network_params(polyak_coeff=1)

    
    def update_target_value_network_params(self, polyak_coeff=None):
        """
        
        """

        if polyak_coeff is None:
            polyak_coeff = self.polyak_coeff

        target_value_network_state_dict = dict(self.target_value_network.named_parameters())
        value_network_state_dict = dict(self.value_network.named_parameters())

        for key in value_network_state_dict:
            value_network_state_dict[key] = ((1-polyak_coeff) * value_network_state_dict[key].clone()) + (polyak_coeff * target_value_network_state_dict[key].clone())

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


    def discriminator_loss(self, 
                           initial_state,
                           state,
                           action, 
                           next_state, 
                           next_state_array, 
                           action_array,
                           skill):
        """
        
        """

        if self.feature == "state":
            input_array = state
        
        elif self.feature == "action":
            input_array = T.cat([initial_state, action], dim=1)

        elif self.feature == "stateDiff":
            input_array = next_state - state

        elif self.feature == "stateAction":
            input_array = T.cat([action, next_state], dim=1)

        elif self.feature == "stateConcat":
            input_array = next_state_array.reshape(self.batch_size, -1)

        elif self.feature == "actionConcat":
            input_array = T.cat([initial_state, action_array.reshape(self.batch_size, -1)], dim=1)

        discriminator_output = self.discriminator.compute_loss(input_array=input_array, skill=skill, use_tanh=self.use_tanh)

        return discriminator_output

    
    def remember(self, state_array, action_array, next_state_array, skill, reward_array, done_array):
        """
        
        """

        self.memory.store_transitions(state_array, action_array, next_state_array, skill, reward_array, done_array)

        return

    
    def choose_action(self, state, skill):
        """
        state: numpy array of dims (n_elems,)
        skill: numpy array of dims (m_elems,)
        """

        state = T.tensor(state, dtype=T.float32).reshape(1,-1).to(self.actor_network.device)
        skill = T.tensor(skill, dtype=T.float32).reshape(1,-1).to(self.actor_network.device)
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
        self.discriminator.save_checkpoint()

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
        self.discriminator.load_checkpoint()

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


    def learn(self, external_batch=None):
        """
        
        """

        # To allow build up of memory in replay buffer
        if self.memory.memory_counter < self.batch_size:
            return

        total_reward = 0

        states_sample, actions_sample, next_states_sample, skills_sample, done_sample = self.memory.sample_buffer(batch_size=self.batch_size, external_batch=external_batch)
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
            done = done_sample[:, i].view(-1)

            # Initializing the value networks
            value_network_value = self.value_network.forward(states).view(-1)
            target_value_network_value = self.target_value_network.forward(next_states).view(-1)
            target_value_network_value[done] = 0.0

            # Update value_network
            critic_value, log_probs = self.compute_q_val(states=states, skills=skills, reparameterize=False)
            self.value_network.optimizer.zero_grad()
            value_target = critic_value - self.alpha * log_probs
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
            actor_loss = T.mean(self.alpha * log_probs - critic_value)
            self.actor_network.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_network.optimizer.step()

            # Discriminator loss
            discriminator_output = self.discriminator_loss(initial_state=states_sample[:, 0, :], 
                                                           state=states, 
                                                           action=actions, 
                                                           next_state=next_states, 
                                                           next_state_array=next_states_sample,
                                                           action_array=actions_sample, 
                                                           skill=skills)
            discriminator_loss = -1 * discriminator_output
            self.discriminator.optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            self.discriminator.optimizer.step()

            # Reward calculation
            reward = (discriminator_output + self.beta * log_probs).sum()
            total_reward = total_reward + reward.detach()

            # Critic network updates
            self.critic_network_1.optimizer.zero_grad()
            self.critic_network_2.optimizer.zero_grad()
            q_hat = ((self.reward_scale * reward) + (self.gamma * target_value_network_value)).detach().view(-1)
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
            
        return total_reward / self.option_interval



class Agent(object):
    """
    For HRL
    """
    
    def __init__(self, 
                 env,
                 skill_dims, 
                 learning_rate, 
                 memory_size, 
                 checkpoint_dir, 
                 option_interval, 
                 gamma, 
                 option_gamma,
                 reward_scale, 
                 batch_size, 
                 polyak_coeff, 
                 beta, 
                 alpha, 
                 use_auto_entropy_adjustment, 
                 min_entropy_target, 
                 w_alpha, 
                 w_auto_entropy_adjustment, 
                 use_tanh, 
                 feature):

        # Environment description attributes
        self.env_name = env.spec.id
        self.num_actions = env.action_space.shape[0]
        self.observation_space_dims = env.observation_space.shape[0]

        self.skill_dims = skill_dims
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.checkpoint_dir = checkpoint_dir
        self.option_interval = option_interval
        self.gamma = gamma
        self.option_gamma = option_gamma
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.polyak_coeff = polyak_coeff
        self.beta = beta
        self.use_tanh = use_tanh
        self.feature = feature

        # Entropy modulating terms for worker
        self.w_alpha = w_alpha
        self.w_auto_entropy_adjustment = w_auto_entropy_adjustment

        # Networks
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
                                  use_auto_entropy_adjustment=self.w_auto_entropy_adjustment, 
                                  use_tanh=self.use_tanh, 
                                  feature=self.feature)
        self.scheduler = SchedulerNetwork(env_name=self.env_name, 
                                          learning_rate=self.learning_rate, 
                                          name="scheduler_network", 
                                          checkpoint_dir=self.checkpoint_dir, 
                                          input_dims=self.observation_space_dims, 
                                          fc1_size=256, 
                                          fc2_size=256, 
                                          output_dims=self.skill_dims * 2, 
                                          option_interval=self.option_interval, 
                                          gamma=self.gamma)
        self.scheduler_memory = SchedulerBuffer(memory_size=self.memory_size, 
                                                skill_dims=self.skill_dims, 
                                                state_dims=self.observation_space_dims)
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

        # Create the directories for saving model parameters, if they don't exist
        Path(self.scheduler.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Entropy modulating terms
        self.alpha = alpha
        self.use_auto_entropy_adjustment = use_auto_entropy_adjustment
        self.min_entropy_target = min_entropy_target 
        self.target_entropy = -T.prod(T.Tensor([env.action_space.high - env.action_space.low])).item()
        self.log_alpha = T.tensor(self.min_entropy_target, dtype=T.float32, requires_grad=True, device=self.scheduler.device)
        #T.ones(1, requires_grad=True, device=self.scheduler.device) * self.min_entropy_target
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        
        # Array for discounting in the loss function
        self.option_interval_discount = np.full(self.option_interval, self.option_gamma)
        self.option_interval_discount = np.power(self.option_interval_discount, [i for i in range(self.option_interval)])
        
        # Align parameters of value_network and target_value_network
        self.update_target_value_network_params(polyak_coeff=1)


    def update_target_value_network_params(self, polyak_coeff=None):
        """
        
        """

        if polyak_coeff is None:
            polyak_coeff = self.polyak_coeff

        target_value_network_state_dict = dict(self.target_value_network.named_parameters())
        value_network_state_dict = dict(self.value_network.named_parameters())

        for key in value_network_state_dict:
            value_network_state_dict[key] = ((1-polyak_coeff) * value_network_state_dict[key].clone()) + (polyak_coeff * target_value_network_state_dict[key].clone())

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

        state = T.tensor(state, dtype=T.float32).reshape(1, -1).to(self.scheduler.device)
        skill, _ = self.scheduler.sample_skill(state=state, reparameterize=False)

        return skill.cpu().detach().numpy().squeeze(axis=0)


    def remember(self, state, skill, next_state, reward, done):
        """
        state: numpy array of (n_elems,)
        skill: numpy array of (m_elems,)
        next_state: numpy array of (n_elems,)
        reward: float
        """

        self.scheduler_memory.store_transitions(state, skill, next_state, reward, done)
        
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
        self.worker.load_models()

        return

    
    def compute_q_val(self, states, reparameterize):
        """
        """

        sampled_skills, log_probs = self.scheduler.sample_skill(state=states, 
                                                                reparameterize=reparameterize)
        log_probs = log_probs.view(-1)
        q1_policy = self.critic_network_1.forward(state_array=states, action_array=sampled_skills)
        q2_policy = self.critic_network_2.forward(state_array=states, action_array=sampled_skills)
        critic_value = T.min(q1_policy, q2_policy).view(-1)

        return critic_value, log_probs


    def post_interval_reward(self, actor_log_probs, reward_array, expected_value=True):
        """
        Calculates the reward for the scheduler after the option interval (K).
        
        log_probs: from actor network of worker module
            Type: numpy array
            Size: 1 x option_interval
        reward_array: from environment
            Type: numpy array
            Size: 1 x option_interval
        """

        if expected_value:
            rewards = reward_array * self.option_interval_discount
            final_reward = actor_log_probs * rewards
            return final_reward.mean().item()

        else:
            return reward_array.sum().item()

    
    def generate_empty_arrays(self):
        """
        Generates empty numpy arrays for worker
        """

        states_array = np.zeros(shape=(self.option_interval, self.observation_space_dims), dtype=np.float32)
        next_states_array = copy.deepcopy(states_array)
        actions_array = np.zeros(shape=(self.option_interval, self.num_actions), dtype=np.float32)
        rewards_array = np.zeros(shape=(self.option_interval), dtype=np.float32)
        actor_log_probs_array = copy.deepcopy(rewards_array)
        done_array = np.zeros(shape=(self.option_interval), dtype=np.bool8)

        return states_array, actions_array, next_states_array, rewards_array, actor_log_probs_array, done_array


    def learn(self):
        """
        
        """

        if self.scheduler_memory.memory_counter < self.batch_size:
            return

        states_sample, skill_sample, next_states_sample, rewards_sample, done_samples, batch = self.scheduler_memory.sample_buffer(self.batch_size)
        states_sample = T.tensor(states_sample, dtype=T.float32).to(self.scheduler.device)
        skill_sample = T.tensor(skill_sample, dtype=T.float32).to(self.scheduler.device)
        next_states_sample = T.tensor(next_states_sample, dtype=T.float32).to(self.scheduler.device)
        rewards_sample = T.tensor(rewards_sample, dtype=T.float32).to(self.scheduler.device)
        done_samples = T.tensor(done_samples).to(self.scheduler.device)

        # Initializing the value networks
        value_network_value = self.value_network.forward(state=states_sample).view(-1)
        target_value_network_value = self.target_value_network.forward(state=next_states_sample).view(-1)
        target_value_network_value[done_samples] = 0.0

        # Update value_network
        critic_value, log_probs = self.compute_q_val(states=states_sample, reparameterize=False)
        self.value_network.optimizer.zero_grad()
        value_target = critic_value - self.alpha * log_probs
        value_loss = 0.5 * F.mse_loss(value_network_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_network.optimizer.step()

        # Updating the scheduler
        critic_value, log_probs = self.compute_q_val(states=states_sample, reparameterize=True)
        scheduler_loss = T.mean(self.alpha * log_probs - critic_value)
        self.scheduler.optimizer.zero_grad()
        scheduler_loss.backward(retain_graph=True)
        self.scheduler.optimizer.step()


        # Updating the critic networks
        self.critic_network_1.optimizer.zero_grad()
        self.critic_network_2.optimizer.zero_grad()
        q_hat = ((self.reward_scale * rewards_sample) + (self.gamma * target_value_network_value)).view(-1)
        q1_old_policy = self.critic_network_1.forward(state_array=states_sample, action_array=skill_sample).view(-1)
        q2_old_policy = self.critic_network_2.forward(state_array=states_sample, action_array=skill_sample).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_network_1.optimizer.step()
        self.critic_network_2.optimizer.step()

        # Updating networks in agent - discriminator and worker
        worker_reward = self.worker.learn(external_batch=batch)

        # Update target_value_network
        self.update_target_value_network_params()
        
        # Update alpha
        self.adjust_alpha(log_prob=log_probs)

        return worker_reward

