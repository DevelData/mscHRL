# networks/general_network.py
import os
import torch as T
import torch.nn as nn
import torch.optim as optim


class GeneralNetwork(nn.Module):
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
        
        super(GeneralNetwork, self).__init__()
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.network_name = name
        self.checkpoint_dir = checkpoint_dir + self.env_name
        self.input_dims = input_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.output_dims = output_dims

        # File in which to save the parameters
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.network_name)

        # NN architecture
        self.fc1 = nn.Linear(in_features=input_dims, 
                             out_features=self.fc1_size)
        self.fc2 = nn.Linear(in_features=self.fc1_size, 
                             out_features=self.fc2_size)
        self.output = nn.Linear(in_features=self.fc2_size, 
                                out_features=self.output_dims)

        # Optimizer and device settings
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def save_checkpoint(self):
        """
        """

        T.save(self.state_dict, self.checkpoint_file)

        return

    
    def load_checkpoint(self, custom_state_dict=None, load_path=None):
        """
        """

        if custom_state_dict is not None:
            self.load_state_dict(T.load(custom_state_dict))

        elif load_path is not None:
            self.load_state_dict(T.load(load_path + self.network_name))

        elif (custom_state_dict is not None) and (load_path is not None):
            raise ValueError("custom_state_dict and load_path cannot both be not None.")
            
        else:
            self.load_state_dict(T.load(self.checkpoint_file))

        return