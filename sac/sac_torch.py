import os
import numpy as np
import torch as T
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import CriticNetwork, ValueNetwork, ActorNetwork

class Agent(object):
    """

    """

    # alpha for actor_lr, beta for critic_lr and value_lr
    # reward_scale is a tunable hyperparameter, depends on number of actions
    # Modulate the parameters of the target value network - tau
    # 2 networks - value and target value networks
    def __init__(self,
                 critic_lr=0.0003,
                 value_lr=0.0003,
                 actor_lr=0.0003,
                 input_dims=[8],
                 env=None,
                 gamma=0.99,
                 tau=0.005,
                 num_actions=2,
                 max_memory_size=1000000,
                 fc1_size=256,
                 fc2_size=256,
                 batch_size=256,
                 reward_scale=2):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_memory_size=max_memory_size,
                                   input_dims=input_dims,
                                   num_actions=num_actions)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.reward_scale = reward_scale

        self.actor_network = ActorNetwork(actor_lr=actor_lr,
                                          input_dims=input_dims,
                                          num_actions=self.num_actions,
                                          max_action=env.action_space.high,
                                          name="actor_network")
        self.critic_network_1 = CriticNetwork(critic_lr=critic_lr,
                                              input_dims=input_dims,
                                              num_actions=self.num_actions,
                                              name="critic_network_1")
        self.critic_network_2 = CriticNetwork(critic_lr=critic_lr,
                                              input_dims=input_dims,
                                              num_actions=self.num_actions,
                                              name="critic_network_2")
        self.value_network = ValueNetwork(value_lr=value_lr,
                                          input_dims=input_dims,
                                          name="value_network")
        self.target_value_network = ValueNetwork(value_lr=value_lr,
                                                 input_dims=input_dims,
                                                 name="target_value_network")
        # Align parameters of value_network and target_value_network
        self.update_network_parameters(tau=1)


    def choose_action(self, observation):
        """

        """

        # np.array([observation])
        state = T.tensor([observation], dtype=T.float32).to(self.actor_network.device)
        # Why default to True
        actions, _ = self.actor_network.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy().squeeze()


    def remember(self, state, action, next_state, reward, done):
        """
        """

        self.memory.store_transitions(state, action, next_state, reward, done)

        return


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_network_parameters = self.target_value_network.named_parameters()
        value_network_parameters = self.value_network.named_parameters()

        target_value_network_state_dict = dict(target_value_network_parameters)
        value_network_state_dict = dict(value_network_parameters)

        for key in value_network_state_dict:
            value_network_state_dict[key] = (tau * value_network_state_dict[key].clone())  + ((1-tau) * target_value_network_state_dict[key].clone())

        self.target_value_network.load_state_dict(value_network_state_dict)

        return


    def save_models(self):
        """

        """

        print("############--Saving models--############")
        self.actor_network.save_checkpoint()
        self.value_network.save_checkpoint()
        self.target_value_network.save_checkpoint()
        self.critic_network_1.save_checkpoint()
        self.critic_network_2.save_checkpoint()

        return


    def load_models(self):
        """

        """

        print("************--Loading models--************")
        self.actor_network.load_checkpoint()
        self.value_network.load_checkpoint()
        self.target_value_network.load_checkpoint()
        self.critic_network_1.load_checkpoint()
        self.critic_network_2.load_checkpoint()

        return


    def learn(self):
        """

        """

        if self.memory.memory_counter < self.batch_size:
            return

        state, action, next_state, reward, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float32).to(self.actor_network.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor_network.device)
        next_state = T.tensor(next_state, dtype=T.float32).to(self.actor_network.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.actor_network.device)
        done = T.tensor(done).to(self.actor_network.device)

        # Keep an eye on this
        value_network_value = self.value_network(state).view(-1)
        target_value_network_value = self.target_value_network(next_state).view(-1)
        target_value_network_value[done] = 0.0


        actions, log_probability = self.actor_network.sample_normal(state, reparameterize=False)
        log_probability = log_probability.view(-1)
        q1_new_policy = self.critic_network_1.forward(state, actions)
        q2_new_policy = self.critic_network_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        self.value_network.optimizer.zero_grad()
        value_target = critic_value - log_probability
        value_loss = 0.5 * F.mse_loss(value_network_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_network.optimizer.step()

        # Repeats from a block of code above
        # Only difference is reparameterize is True
        actions, log_probability = self.actor_network.sample_normal(state, reparameterize=True)
        log_probability = log_probability.view(-1)
        q1_new_policy = self.critic_network_1.forward(state, actions)
        q2_new_policy = self.critic_network_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = T.mean(log_probability - critic_value)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()

        self.critic_network_1.optimizer.zero_grad()
        self.critic_network_2.optimizer.zero_grad()
        q_hat = (self.reward_scale * reward) + (self.gamma * target_value_network_value)
        q1_old_policy = self.critic_network_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_network_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_network_1.optimizer.step()
        self.critic_network_2.optimizer.step()

        self.update_network_parameters()
