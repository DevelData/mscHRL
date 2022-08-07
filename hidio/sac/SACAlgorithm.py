import gin
from enum import Enum
from collections import namedtuple
from absl import logging
# Need dist_utils from alf.utils
from .util


ActionType = Enum("ActionType", ("Discrete", "Continuous", "Mixed"))

SacActionState = namedtuple("SacActionState", 
                            ["actor_network", "critic"], 
                            default_value=())
SacCriticState = namedtuple("SacCriticState", 
                            ["critics", "target_critics"])
SacState = namedtuple("SacState", 
                      ["action", "actor", "critic"])
SacCriticInfo = namedtuple("SacCriticInfo", 
                           ["critics", "target_critic"])
SacActorInfo = namedtuple("SacActorInfo", 
                          ["actor_loss", "neg_entropy"], 
                          default_value=())
SacInfo = namedtuple("SacInfo", 
                     ["action_distribution", "actor", "critic", "alpha"], 
                     default_value=())
SacLossInfo = namedtuple("SacLossInfo", ["actor", "critic", "alpha"])


def set_target_entropy(name, target_entropy, flat_action_spec):
    """"""
    
    if (target_entropy is None) or callable(target_entropy):
        if target_entropy is None:
            target_entropy = 