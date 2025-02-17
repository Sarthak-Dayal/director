import torch.nn as nn
from omegaconf import DictConfig

# Activation function mapping
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    "none": nn.Identity()  # No activation
}

class ConfigurableMLP(nn.Module):
    def __init__(self, in_dim: int, config: DictConfig, output_dim: int) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, config.hidden_layers[0].size),
                  ACTIVATIONS.get(config.hidden_layers[0].activation, nn.ReLU())]

        # Lazy input layer (infers input size automatically)

        # Hidden layers
        for i in range(1, len(config.hidden_layers)):  # Start from second hidden layer
            layer_cfg = config.hidden_layers[i]
            layers.append(nn.Linear(config.hidden_layers[i - 1].size, layer_cfg.size))
            layers.append(ACTIVATIONS.get(layer_cfg.activation, nn.ReLU()))

        # Output layer (final layer without activation)
        layers.append(nn.Linear(config.hidden_layers[-1].size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
