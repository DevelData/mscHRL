import os
#import functools
import gin
import numpy as np
from collections import namedtuple, OrderedDict
import torch.distributions as td
from .TensorSpec import TensorSpec



#def get_invertable(cls):
#    """"""
    
#    class NewCls(cls):
#        __init__ = functools.partialmethod(cls.__init__, cache_size=1)
#        
#    return NewCls


@gin.configurable
def calc_default_target_entropy(spec: TensorSpec, min_prob: float = 0.1):
    """"""
    
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = spec.is_continuous
    log_mp = np.log(min_prob)
    entropy = np.sum([np.log(M - m) + log_mp 
                      if cont else min_prob*(np.log(M - m) - log_mp) 
                      for M,m,_ in min_max])
    
    return entropy

