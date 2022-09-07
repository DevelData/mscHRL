from general_network import GeneralNetwork
import torch as T
import torch.nn.functional as F
import torch.nn as nn



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