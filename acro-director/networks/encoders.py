import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from configurable_mlp import ConfigurableMLP

# Input is image state
# Dreamer Config for describing model
class WorldModelEncoder(nn.Module):
    pass

# Input is image state
# ACRO Config for describing model
class ACROEncoder(nn.Module):
    pass

# Input is (Dreamer encoding, ACRO encoding)
# Output is goal latent space
# Config for describing model
class GoalAutoencoder(nn.Module):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ConfigurableMLP(config.world_latent_size + config.acro_latent_size, config.model, config.goal_latent_size)
        self.decoder = ConfigurableMLP(config.goal_latent_size, config.model, config.world_latent_size + config.acro_latent_size)

    def forward(self, world_latent: torch.Tensor, acro_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(world_latent, acro_latent))