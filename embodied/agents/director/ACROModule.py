import sys

import embodied
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

from . import behaviors
from . import nets
from . import tfagent
from . import tfutils


class ACROModule(tfutils.Module):

    # ACRO Model
    # Translation from World Model to ACRO space
    # ACRO to skill VAE

    def __init__(self, act_space, config):
        self.config = config

        # ACRO Model
        self.acro_state_size = self.img_size = (self.config.frame_stack,) + tuple(self.config.env.size) if self.config.env.gray else tuple(self.config.env.size) + (3,)
        self.acro_encoder = nets.MultiEncoder({"frame_stack": self.acro_state_size}, **self.config.acro_encoder)
        self.acro_action_backbone = nets.MLP(act_space.shape, **self.config.acro_action_backbone)


        # Translation Layer
        self.wm_to_acro_backbone = nets.MLP(shape=(), **self.config.wm_to_acro_backbone)
        self.wm_to_acro_head = nets.Dense(units=self.acro_state_size[-1], **self.config.wm_to_acro_head)



    def train(self):
        pass