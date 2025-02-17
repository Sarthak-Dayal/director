import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Pretty print YAML config

    # Load DreamerV3 Encoder

    # Load ACRO Encoder

    # Create ACRO-Director Network



if __name__ == '__main__':
    train()