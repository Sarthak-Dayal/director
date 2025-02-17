import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def pretrain(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Pretty print YAML config

    # Load DreamerV3 and ACRO Encoder

    # Train encoder


if __name__ == '__main__':
    pretrain()