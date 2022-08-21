import numpy as np
import copy


# Option/Skill dimension (D) is different to the option interval (K).

class SchedulerBuffer(object):
    """
    
    """
    
    def __init__(self, 
                 memory_size, 
                 skill_dims, 
                 state_dims, 
                 num_actions):

        # K = option_interval, I don't know where to use it yet
        # Don't need to use K for the SchedulerBuffer
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.skill_dims = skill_dims
        
        self.state_memory = np.zeros(shape=(self.memory_size, self.state_dims), dtype=np.float32)
        # Deciding to store only a single state for a particular option
        self.next_state_memory = copy.deepcopy(self.state_memory)
        #np.zeros(shape=(self.memory_size, self.skill_dims, self.state_dims), dtype=np.float32)
        self.reward_memory = np.zeros(shape=self.memory_size, dtype=np.float32)
        self.skill_memory = np.zeros(shape=(self.memory_size, self.skill_dims, self.num_actions), dtype=np.float32)
        self.terminal_memory = np.zeros(shape=self.memory_size, dtype=np.bool8)
    
    
    def store_transitions(self, state, skill, next_state, reward):
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
        #rewards_sample = self.reward_memory[batch]
        #terminal_sample = self.terminal_memory[batch]

        return (states_sample, skill_sample, next_states_sample)
    
    
    
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
        self.next_state_memory = np.zeros(shape=(self.memory_size, self.option_interval, self.state_dims), dtype=np.float32)
        self.action_memory = np.zeros(shape=(self.memory_size, self.option_interval, self.num_actions), dtype=np.float32)
        self.reward_memory = np.zeros(shape=(self.memory_size, self.option_interval))
        self.skill_memory = np.zeros(shape=(self.memory_size, self.skill_dims, self.num_actions), dtype=np.float32)


    def store_transitions(self, state_array, action_array, next_state_array, skill, reward_array):
        """
        
        """

        idx = self.memory_counter % self.memory_size

        self.state_memory[idx] = state_array
        self.action_memory[idx] = action_array
        self.next_state_memory[idx] = next_state_array
        self.skill_memory[idx] = skill
        self.reward_memory[idx] = reward_array

        self.memory_counter += 1

        return

    
    def sample_buffer(self, batch_size):
        """
        
        """
        # How to sample an action given a state and an option/a skill 
        # Actor network in traditional SAC only takes state as an input and 
        # outputs an action
        max_current_memory = min(self.memory_counter, self.memory_size)
        batch = np.random_choice(max_current_memory, size=batch_size)

        #states_sample = self.state_memory[batch]
        actions_sample = self.action_memory[batch]
        next_states_sample = self.next_state_memory[batch]
        skills_sample = self.skill_memory[batch]

        return actions_sample, next_states_sample, skills_sample