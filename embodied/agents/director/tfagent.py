import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist

import embodied
# Assume that our tfutils.Module is now a thin PyTorch module (a subclass of nn.Module)
from . import tfutils

# Mimic a distributed “values” type for interface compatibility.
# In practice, PyTorch’s DDP/DataParallel handles scattering/gathering.
class PerReplica:
    def __init__(self, values):
        self.values = values

class TFAgent(tfutils.Module, embodied.Agent):
    def __new__(subcls, obs_space, act_space, step, config):
        # Create instance and initialize distribution settings.
        self = super().__new__(TFAgent)
        # IMPORTANT: Call the nn.Module initializer before assigning any submodules.
        nn.Module.__init__(self)
        # Keep the same interface: config.tf is used to hold Torch settings.
        self.config = config.tf
        self.strategy = self._setup()  # This sets self.device as well.
        # Create underlying agent instance.
        self.agent = super().__new__(subcls)
        # with self._strategy_scope():
        #     self.agent.__init__(obs_space, act_space, step, config)
        torch.set_default_device(self.device)
        self.agent.__init__(obs_space, act_space, step, config)
        # TensorFlow concrete function caching is not needed in PyTorch.
        self._cache_fns = False  # (config.tf.jit and not self.strategy)
        self._cached_fns = {}
        return self

    def dataset(self, generator):
        with self._strategy_scope():
            dataset = self.agent.dataset(generator)
        return dataset

    def policy(self, obs, state=None, mode='train'):
        # Filter out keys starting with "log_"
        obs = {k: v for k, v in obs.items() if not k.startswith('log_')}
        if state is None:
            state = self.agent.initial_policy_state(obs)
        fn = self.agent.policy
        # No caching needed in PyTorch; ignore get_concrete_function.
        act, state = fn(obs, state, mode)
        act = self._convert_outs(act)
        return act, state

    def train(self, data, state=None):
        data = self._convert_inps(data)
        if state is None:
            state = self._strategy_run(self.agent.initial_train_state, data)
        fn = self.agent.train
        # No caching.
        outs, state, metrics = self._strategy_run(fn, data, state)
        outs = self._convert_outs(outs)
        metrics = self._convert_mets(metrics)
        return outs, state, metrics

    def report(self, data):
        data = self._convert_inps(data)
        fn = self.agent.report
        # No caching.
        metrics = self._strategy_run(fn, data)
        metrics = self._convert_mets(metrics)
        return metrics

    @contextlib.contextmanager
    def _strategy_scope(self):
        # In PyTorch we do not need a special scope; if using DataParallel/DDP the model already handles it.
        yield

    def _strategy_run(self, fn, *args, **kwargs):
        # In a true distributed setting, you might need to call fn via the strategy.
        # For now, we simply call it.
        return fn(*args, **kwargs)

    def _convert_inps(self, value):
        # If no strategy, simply recursively convert inputs to torch.Tensors on our device.
        return self._recursive_convert(value, to_tensor=True)

    def _convert_outs(self, value):
        # Recursively convert outputs: if tensor, return a NumPy array.
        return self._recursive_convert(value, to_tensor=False)

    def _convert_mets(self, value):
        return self._recursive_convert(value, to_tensor=False)

    def _recursive_convert(self, value, to_tensor=True):
        """Recursively convert lists/dicts to torch.Tensors (or back to NumPy)."""
        if isinstance(value, dict):
            return {k: self._recursive_convert(v, to_tensor) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(self._recursive_convert(v, to_tensor) for v in value)
        elif isinstance(value, torch.Tensor):
            if to_tensor:
                return value.to(self.strategy_device())
            else:
                return value.detach().cpu().numpy()
        else:
            if to_tensor:
                try:
                    return torch.tensor(value, device=self.strategy_device())
                except Exception:
                    return value
            else:
                return value

    def strategy_device(self):
        # Return the device used in our strategy (or self.device).
        return self.device if hasattr(self, 'device') else torch.device("cpu")

    def _setup(self):
        # Mimic TF’s _setup but with PyTorch calls.
        # Ensure precision is either 16 or 32.
        assert self.config.precision in (16, 32), f"Unsupported precision: {self.config.precision}"
        platform = self.config.platform.lower()
        if platform == 'cpu':
            self.device = torch.device("cpu")
            return None
        elif platform == 'gpu':
            assert torch.cuda.is_available(), "GPU requested but none available"
            self.device = torch.device("cuda:0")
            if self.config.precision == 16:
                # For mixed precision, later training code can use torch.cuda.amp.autocast.
                pass
            return None
        elif platform == 'multi_gpu':
            assert torch.cuda.device_count() >= 1, "No GPUs available"
            self.device = torch.device("cuda:0")
            # Use DataParallel (or DistributedDataParallel if preferred).
            return "data_parallel"
        elif platform == 'multi_worker':
            # Setup for multi-worker training (using torch.distributed) should be done externally.
            assert torch.cuda.is_available(), "No GPUs available"
            self.device = torch.device("cuda:0")
            return "ddp"
        elif platform == 'tpu':
            # TPU support in PyTorch is experimental (via XLA). Here we raise an error.
            raise NotImplementedError("TPU support is not implemented in this PyTorch conversion.")
        else:
            raise NotImplementedError(f"Platform {self.config.platform} is not supported.")
