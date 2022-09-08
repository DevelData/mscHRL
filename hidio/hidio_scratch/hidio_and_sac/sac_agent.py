from pathlib import Path
import torch as T
import torch.nn.functional as F
import torch.optim as optim

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from value_network import ValueNetwork
from sac_buffer import ReplayBuffer



class Agent(object):
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
                 gamma,
                 alpha,
                 use_auto_entropy_adjustment,
                 learning_rate=10**-4):

        self.max_memory_size = max_memory_size
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.polyak_coeff = polyak_coeff
        self.checkpoint_dir = checkpoint_dir
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Environment variables
        self.env_name = env.spec.id
        self.num_actions = env.action_space.shape[0]
        self.observation_space_dims = env.observation_space.shape[0]
        self.max_action = env.action_space.high

        # Create the directories for saving model parameters, if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Networks
        self.memory = ReplayBuffer(memory_size=self.max_memory_size, 
                                   state_dims=self.observation_space_dims,
                                   num_actions=self.num_actions)
        self.actor_network = ActorNetwork(env_name=self.env_name, 
                                          learning_rate=self.learning_rate, 
                                          name="actor_network", 
                                          checkpoint_dir=self.checkpoint_dir, 
                                          input_dims=self.observation_space_dims, 
                                          fc1_size=256, 
                                          fc2_size=256, 
                                          output_dims=self.num_actions * 2,
                                          max_action=self.max_action)
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
        #T.zeros(1, requires_grad=True, device=self.actor_network.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Align parameters of value_network and target_value_network
        self.update_target_value_network_params(polyak_coeff=0)

    
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

    
    def remember(self, state_array, action_array, next_state_array, reward_array, done_array):
        """
        
        """

        self.memory.store_transitions(state_array, action_array, next_state_array, reward_array, done_array)

        return

    
    def choose_action(self, state, reparameterize):
        """
        state: numpy array of dims (n_elems,)
        skill: numpy array of dims (m_elems,)
        reparameterize: bool to indicate whether to sample method or rsample 
                        method.
        """

        state = T.tensor(state, dtype=T.float32).reshape(1,-1).to(self.actor_network.device)
        action, log_probs = self.actor_network.sample_distribution(states_array=state, 
                                                                   skills_array=None, 
                                                                   reparameterize=reparameterize)

        action = action.cpu().detach().numpy().squeeze(axis=0)
        log_probs = log_probs.cpu().detach().numpy().squeeze(axis=0)

        return action, log_probs


    def save_models(self):
        """
        
        """

        print("############--Saving models--############")
        self.actor_network.save_checkpoint()
        self.critic_network_1.save_checkpoint()
        self.critic_network_2.save_checkpoint()
        self.target_value_network.save_checkpoint()
        self.value_network.save_checkpoint()

        return

    
    def load_models(self):
        """
        
        """        

        print("############--Loading  models--############")
        self.actor_network.load_checkpoint()
        self.critic_network_1.load_checkpoint()
        self.critic_network_2.load_checkpoint()
        self.target_value_network.load_checkpoint()
        self.value_network.load_checkpoint()

        return

    
    def compute_q_val(self, states, reparameterize):
        """
        
        """

        sampled_actions, log_probs = self.actor_network.sample_distribution(states_array=states, 
                                                                            skills_array=None, 
                                                                            reparameterize=reparameterize)
        log_probs = log_probs.view(-1)
        q1_policy = self.critic_network_1.forward(state_array=states, action_array=sampled_actions)
        q2_policy = self.critic_network_2.forward(state_array=states, action_array=sampled_actions)
        critic_value = T.min(q1_policy, q2_policy).view(-1)

        return critic_value, log_probs


    def transfer_network_params(self, custom_state_dict=None, load_path=None):
        """
        Transfers network parameters from source_network to the target_network.
        """

        print("############--Transferring model parameters--############")
        self.actor_network.load_checkpoint(custom_state_dict=custom_state_dict, load_path=load_path)
        self.critic_network_1.load_checkpoint(custom_state_dict=custom_state_dict, load_path=load_path)
        self.critic_network_2.load_checkpoint(custom_state_dict=custom_state_dict, load_path=load_path)
        self.value_network.load_checkpoint(custom_state_dict=custom_state_dict, load_path=load_path)
        self.target_value_network.load_checkpoint(custom_state_dict=custom_state_dict, load_path=load_path)

        return


    def learn(self):
        """
        
        """

        # To allow build up of memory in replay buffer
        if self.memory.memory_counter < self.batch_size:
            return

        states_sample, actions_sample, next_states_sample, rewards_sample, done_sample = self.memory.sample_buffer(batch_size=self.batch_size)
        states_sample = T.tensor(states_sample, dtype=T.float32).to(self.actor_network.device)
        actions_sample = T.tensor(actions_sample, dtype=T.float32).to(self.actor_network.device)
        next_states_sample = T.tensor(next_states_sample, dtype=T.float32).to(self.actor_network.device)
        rewards_sample = T.tensor(rewards_sample, dtype=T.float32).to(self.actor_network.device)
        done_sample = T.tensor(done_sample).to(self.actor_network.device)

        # Initializing the value networks
        value_network_value = self.value_network.forward(states_sample).view(-1)
        target_value_network_value = self.target_value_network.forward(next_states_sample).view(-1)
        target_value_network_value[done_sample] = 0.0

        # Update value_network
        critic_value, log_probs = self.compute_q_val(states=states_sample, reparameterize=False)
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
        critic_value, log_probs = self.compute_q_val(states=states_sample, reparameterize=True)
        actor_loss = T.mean(self.alpha * log_probs - critic_value)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()

        # Critic network updates
        self.critic_network_1.optimizer.zero_grad()
        self.critic_network_2.optimizer.zero_grad()
        q_hat = ((self.reward_scale * rewards_sample) + (self.gamma * target_value_network_value)).detach().view(-1)
        q1_old_policy = self.critic_network_1.forward(state_array=states_sample, action_array=actions_sample).view(-1)
        q2_old_policy = self.critic_network_2.forward(state_array=states_sample, action_array=actions_sample).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_network_1.optimizer.step()
        self.critic_network_2.optimizer.step()

        # Update target_value_network
        self.update_target_value_network_params(polyak_coeff=self.polyak_coeff)

        # Update alpha
        self.adjust_alpha(log_prob=log_probs)
            
        return