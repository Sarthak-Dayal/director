# Input is Dreamer encoding, output is goal latent space
import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from configurable_mlp import ConfigurableMLP


class ManagerNet(nn.Module):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ConfigurableMLP(config.world_latent_size, config.model, config.goal_latent_size)

    def forward(self, world_latent: torch.Tensor) -> torch.Tensor:
        return self.net(world_latent)

# Input is ACRO encoding
# Output is env action space
class WorkerNet(nn.Module):
    def __init__(self, config: DictConfig, action_space: gym.spaces.space.Space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ConfigurableMLP(2 * config.acro_latent_size, config.model, np.prod(action_space.shape).item()) # TODO Generalize this to other configurable neural networks
        self.action_shape = action_space.shape

    def forward(self, acro_state: torch.Tensor, acro_goal: torch.Tensor) -> torch.Tensor:
        action = self.net(torch.cat([acro_state, acro_goal], dim=1))
        return action.reshape(self.action_shape)