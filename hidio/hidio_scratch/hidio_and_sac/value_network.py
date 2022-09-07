from general_network import GeneralNetwork
import torch as T
import torch.nn.functional as F
import torch.nn as nn



class ValueNetwork(GeneralNetwork):
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

        super(ValueNetwork, self).__init__(env_name, 
                                           learning_rate, 
                                           name, 
                                           checkpoint_dir, 
                                           input_dims, 
                                           fc1_size, 
                                           fc2_size, 
                                           output_dims)

        
    def forward(self, state):
        """
        """

        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        value = self.output(state_value)

        return value