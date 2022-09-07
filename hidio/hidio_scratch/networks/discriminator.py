# networks/discriminator.py
from networks.general_network import GeneralNetwork
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn




class DiscriminatorNetwork(GeneralNetwork):
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
        
        super(DiscriminatorNetwork, self).__init__(env_name,
                                                   learning_rate, 
                                                   name, 
                                                   checkpoint_dir, 
                                                   input_dims, 
                                                   fc1_size, 
                                                   fc2_size, 
                                                   output_dims)

    def forward(self, input_array, use_tanh=False):
        """
        input_array can be a combination of actions, states or next states.
        Make sure to preprocess the inputs before passing them through. Also,
        modify input_dims before using this function.
        """
        # Have to concatenate the inputs and output something that matches 
        # the dimensions of the skill.

        processed_input = F.relu(self.fc1(input_array))
        processed_input = F.relu(self.fc2(processed_input))
        predicted_skill = self.output(processed_input)

        if use_tanh:
            predicted_skill = T.tanh(predicted_skill)

        return predicted_skill


    def compute_loss(self, input_array, skill, use_tanh):
        """
        Check feature extractor table to know what to input in input_array.
        """

        predicted_skill = self.forward(input_array=input_array, use_tanh=use_tanh)

        return -1 * F.mse_loss(predicted_skill, skill)