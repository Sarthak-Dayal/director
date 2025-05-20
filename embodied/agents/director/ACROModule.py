import sys

import numpy as np

import embodied
import ruamel.yaml as yaml
import tensorflow as tf
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

        self.acro_encoder_backbone = nets.MultiEncoder({"frame_stack": self.acro_state_size}, **self.config.acro_encoder_backbone)
        self.acro_embedding_head = nets.Dense(units=self.config.acro_embed_size, **self.config.acro_embedding_head)

        self.acro_action_backbone = nets.MLP(act_space.shape, **self.config.acro_action_backbone)

        # Translation Layer
        self.wm_to_acro_backbone = nets.MLP(shape=(self.config.acro_embed_size,), **self.config.wm_to_acro_transition)

    def embed_acro(self, frame_stack):
        hidden = self.acro_encoder_backbone({"frame_stack": frame_stack})
        return self.acro_embedding_head(hidden)

    def translate_wm(self, wm_state):
        return self.wm_to_acro_backbone(wm_state)

    def train(self, data):
        pass