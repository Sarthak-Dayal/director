import pathlib
import sys
import warnings

import torch

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

directory = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import sys
import copy
from types import SimpleNamespace
import torch

import embodied

# Import the world model from your agent module.
# Adjust the import below as needed if your module path is different.



# -------------------------------------------------------------------------
# Dummy configuration and observation space for testing.
# -------------------------------------------------------------------------
def create_dummy_config():
    config = SimpleNamespace()
    config.rssm = {
        'stoch_size': 30,
        'deter_size': 200,
        'hidden_size': 200,
    }
    config.encoder = {
        'hidden_sizes': [32, 32],
    }
    config.decoder = {
        'hidden_sizes': [32, 32],
    }
    config.reward_head = {
        'hidden_sizes': [32],
    }
    config.cont_head = {
        'hidden_sizes': [32],
    }
    config.model_opt = {
        'lr': 1e-3,
    }
    config.wmkl = {
        'init': 1.0,
        'tol': 0.001,
        'lr': 1e-3,
    }
    config.loss_scales = {
        'kl': 1.0,
        'decoder': 1.0,
        'reward': 1.0,
        'cont': 1.0,
    }
    # Ensure that all heads get gradients.
    config.grad_heads = ['decoder', 'reward', 'cont']
    config.priority = 'reward'
    config.tf = SimpleNamespace(debug_nans=False)
    config.imag_unroll = 1
    config.imag_discount = 0.99
    config.discount = 0.99
    return config


# A simple dummy “space” that provides a shape attribute.
class DummySpace:
    def __init__(self, shape):
        self.shape = shape


def create_dummy_obs_space():
    obs_space = {
        'image': DummySpace((3, 64, 64)),
        'reward': DummySpace((1,)),
        'is_first': DummySpace((1,)),
        'is_terminal': DummySpace((1,)),
    }
    return obs_space


def create_dummy_batch(batch_size, action_dim):
    # Create a dummy batch of data.
    # "image" is in uint8 as expected by the preprocessing.
    data = {
        "image": torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8),
        "action": torch.randn(batch_size, action_dim),
        "is_first": torch.zeros(batch_size, dtype=torch.bool),
        "is_terminal": torch.zeros(batch_size, dtype=torch.bool),
        "reward": torch.randn(batch_size, 1),
    }
    # In the agent, "cont" is computed as 1.0 - is_terminal.
    data["cont"] = 1.0 - data["is_terminal"].to(torch.float32)
    return data


# -------------------------------------------------------------------------
# Test function for the world model.
# -------------------------------------------------------------------------
def test_world_model(argv=None):
    from . import agent_torch as agnt
    torch.manual_seed(42)

    parsed, other = embodied.Flags(
        configs=['defaults'], actor_id=0, actors=0,
    ).parse_known(argv)

    config = embodied.Config(agnt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agnt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)

    config = config.update(logdir=str(embodied.Path(config.logdir)))
    print(config)

    env = embodied.envs.load_env(config.task, mode='train', **config.env)

    wm = agnt.WorldModel(env.obs_space, config)

    batch_size = 4
    action_dim = 4  # assuming a 4-dimensional action space
    import pdb; pdb.set_trace()
    data = create_dummy_batch(batch_size, env.act_space)

    # Get an initial state from the RSSM.
    state = wm.rssm.initial(batch_size)

    # Save copies of all parameters before the training step.
    modules = [wm.encoder, wm.rssm] + list(wm.heads.values())
    original_params = {}
    for mod in modules:
        for name, param in mod.named_parameters():
            original_params[f"{mod.__class__.__name__}.{name}"] = param.clone().detach()

    # Compute the loss (which includes the forward pass through the encoder, RSSM, and heads).
    loss, last_state, outputs, metrics = wm.loss(data, state=state, training=True)

    # Do a backward pass.
    loss.backward()

    # Report any parameters that did not get a gradient.
    print("\nParameters with NO gradients:")
    for mod in modules:
        for name, param in mod.named_parameters():
            if param.grad is None:
                print(f"  {mod.__class__.__name__}.{name} has no gradient!")

    # Perform an optimizer step (using the world model’s optimizer wrapper).
    wm.model_opt.step(loss, modules)

    # Compare parameters before and after the optimizer step.
    print("\nParameter update differences (mean absolute change):")
    for mod in modules:
        for name, param in mod.named_parameters():
            orig = original_params[f"{mod.__class__.__name__}.{name}"]
            diff = (param - orig).abs().mean().item()
            print(f"  {mod.__class__.__name__}.{name}: {diff:.6f}")


if __name__ == '__main__':
    test_world_model()
