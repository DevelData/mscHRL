import numpy as np
import copy


class ReplayBuffer(object):
    
    def __init__(self, max_memory_size, input_dims, num_actions):
        self.max_memory_size = max_memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros(shape=(self.max_memory_size * input_dims), 
                                     dtype=np.float32)
        self.next_state_memory = copy.deepcopy(self.state_memory)
        self.action_memory = np.zeros(shape=(self.max_memory_size, num_actions), 
                                      dtype=np.float32)
        self.terminal_memory = np.zeros(shape=self.max_memory_size, 
                                        dtype=np.bool8)
        self.reward_memory = np.zeros(shape=self.max_memory_size, 
                                      dtype=np.float32)
        
    def store_transitions(self, state, action, next_state, reward, done):
        """
        
        """
        
        idx = self.memory_counter % self.max_memory_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.next_state_memory[idx] = next_state
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        
        # Incrementing the memory counter
        self.memory_counter = self.memory_counter + 1
        
        return
    
    def sample_buffer(self, batch_size):
        """
        
        """
        
        max_current_memory = min(self.memory_counter, self.max_memory_size)
        batch = np.random.choice(max_current_memory, size=batch_size)
        
        states_sample = self.state_memory[batch]
        actions_sample = self.action_memory[batch]
        next_states_sample = self.next_state_memory[batch]
        rewards_sample = self.reward_memory[batch]
        terminal_sample = self.terminal_memory[batch]
        
        return (states_sample, 
                actions_sample, 
                next_states_sample, 
                rewards_sample, 
                terminal_sample)