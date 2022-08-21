from audioop import mul
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class DiscriminatorNetwork(nn.Module):
    """
    
    """
    
    def __init__(self, env_name, learning_rate, dimensions, normal_mu, normal_std, name, checkpoint_dir, fc1_size=64, fc2_size=64):
        
        super(DiscriminatorNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.normal_mu = normal_mu
        self.normal_std = normal_std
        self.env_name = env_name
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.
        self.
    pass
