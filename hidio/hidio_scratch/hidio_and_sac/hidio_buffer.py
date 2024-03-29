import numpy as np
import copy



# Option/Skill dimension (D) is different to the option interval (K).

class SchedulerBuffer(object):
    """
    
    """
    
    def __init__(self, 
                 memory_size, 
                 skill_dims, 
                 state_dims):

        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_dims = state_dims
        self.skill_dims = skill_dims
        
        self.state_memory = np.zeros(shape=(self.memory_size, self.state_dims), dtype=np.float32)
        self.next_state_memory = copy.deepcopy(self.state_memory)
        self.reward_memory = np.zeros(shape=self.memory_size, dtype=np.float32)
        self.skill_memory = np.zeros(shape=(self.memory_size, self.skill_dims), dtype=np.float32)
        self.terminal_memory = np.zeros(shape=self.memory_size, dtype=np.bool8)
    
    
    def store_transitions(self, state, skill, next_state, reward, done):
        """
        Storing the current state, skill (set of actions), next_states and the 
        reward (s_(h,0), u_h, s_(h+1, 0), R_h). R_h is the result of the rewards
        being summed 0 to K (option interval) - 1 times. The rewards could or 
        could not be discounted (question for implementation).
        """
        
        idx = self.memory_counter % self.memory_size

        # The last state from the next state for a skill is the starting state.
        self.state_memory[idx] = state
        self.skill_memory[idx] = skill
        self.next_state_memory[idx] = next_state
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.memory_counter += 1
        
        return
    

    def sample_buffer(self, batch_size):
        """

        """

        max_current_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_current_memory, size=batch_size)

        states_sample = self.state_memory[batch]
        skill_sample = self.skill_memory[batch]
        next_states_sample = self.next_state_memory[batch]
        rewards_sample = self.reward_memory[batch]
        done_sample = self.terminal_memory[batch]

        return (states_sample, skill_sample, next_states_sample, rewards_sample, done_sample, batch)
    
    
    
class WorkerReplayBuffer(object):
    """
    Stores the options, actions, states, next states and intrinsic rewards 
    (u, a, s, s', r) for the worker. The memory_size attribute is the same as 
    that of the SchedulerBuffer. Samples are picked from the option_interval
    dim for each 
    u is currently not stored. 
    """

    def __init__(self, 
                 option_interval, 
                 memory_size, 
                 state_dims, 
                 skill_dims, 
                 num_actions):
        
        self.option_interval = option_interval
        self.memory_size = memory_size
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.skill_dims = skill_dims
        self.memory_counter = 0
        
        self.state_memory = np.zeros(shape=(self.memory_size, self.option_interval, self.state_dims), dtype=np.float32)
        self.next_state_memory = copy.deepcopy(self.state_memory)
        self.action_memory = np.zeros(shape=(self.memory_size, self.option_interval, self.num_actions), dtype=np.float32)
        self.reward_memory = np.zeros(shape=(self.memory_size, self.option_interval), dtype=np.float32)
        self.skill_memory = np.zeros(shape=(self.memory_size, self.skill_dims), dtype=np.float32)
        self.terminal_memory = np.zeros(shape=(self.memory_size, self.option_interval), dtype=np.bool8)


    def store_transitions(self, 
                          state_array, 
                          action_array, 
                          next_state_array, 
                          skill, 
                          reward_array, 
                          done_array):
        """
        
        """

        idx = self.memory_counter % self.memory_size

        self.state_memory[idx] = state_array
        self.action_memory[idx] = action_array
        self.next_state_memory[idx] = next_state_array
        self.skill_memory[idx] = skill
        self.reward_memory[idx] = reward_array
        self.terminal_memory[idx] = done_array

        self.memory_counter += 1

        return

    
    def sample_buffer(self, batch_size, external_batch=None):
        """
        
        """
        
        if external_batch is not None:
            batch = external_batch
        else:
            max_current_memory = min(self.memory_counter, self.memory_size)
            batch = np.random.choice(max_current_memory, size=batch_size)

        states_sample = self.state_memory[batch]
        actions_sample = self.action_memory[batch]
        next_states_sample = self.next_state_memory[batch]
        skills_sample = self.skill_memory[batch]
        done_sample = self.terminal_memory[batch]

        return states_sample, actions_sample, next_states_sample, skills_sample, done_sample