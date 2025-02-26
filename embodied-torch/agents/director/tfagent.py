# tfagent.py
import contextlib
import os
import logging
import inspect
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.distributed as dist

# Suppress low-level logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)

# Import helper functions from our tfutils module.
from tfutils import (
    tensor,
    shuffle,
    scan,
    symlog,
    symexp,
    action_noise,
    lambda_return,
    map_structure,
    Module
)

# ---------------------------------------------------------------------------
# Base Agent Class
# ---------------------------------------------------------------------------
class TFAgent(nn.Module):
    """
    A PyTorch re-implementation of your original TensorFlow TFAgent.
    This base class (which extends both nn.Module and your embodied.Agent interface)
    provides methods for dataset creation, converting inputs/outputs, and (optionally)
    wrapping the inner agent with multi-GPU/distributed strategies.
    """

    def __new__(subcls, obs_space, act_space, step, config):
        # Create an instance of TFAgent.
        self = super().__new__(subcls)
        # Store configuration (we assume config.tf is used to mimic the original API)
        self.config = config.tf
        # Setup device/precision and distribution strategy.
        self.strategy = self._setup()  # Returns None, "dp", or "ddp"
        # Create the inner agent instance.
        inner_agent = object.__new__(subcls)
        with self._strategy_scope():
            inner_agent.__init__(obs_space, act_space, step, config)
        self.agent = inner_agent
        # If a distribution strategy is used, wrap the agent accordingly.
        if self.strategy == "dp":
            self.agent = nn.DataParallel(self.agent)
        elif self.strategy == "ddp":
            self.agent.to(self.device)
            self.agent = nn.parallel.DistributedDataParallel(
                self.agent,
                device_ids=[self.device.index] if self.device.type == "cuda" else None
            )
        # Optionally compile the model if JIT is enabled (PyTorch 2.0+).
        if self.config.jit:
            self.agent = torch.compile(self.agent)
            self._cache_fns = False
        else:
            self._cache_fns = (self.strategy is None)
        self._cached_fns = {}
        return self

    def __init__(self, obs_space, act_space, step, config):
        """
        Initialize the base agent. (Additional initialization specific to your application
        can be added here.)
        """
        # Here you would initialize your embodied.Agent (if needed).
        # For example: embodied.Agent.__init__(self, obs_space, act_space, step, config)
        pass

    def dataset(self, generator):
        with self._strategy_scope():
            dataset = self.agent.dataset(generator)
        return dataset

    def policy(self, obs, state=None, mode='train'):
        # Remove any keys starting with 'log_'.
        obs = {k: v for k, v in obs.items() if not k.startswith('log_')}
        if state is None:
            state = self.agent.initial_policy_state(obs)
        fn = self.agent.policy
        if self._cache_fns:
            key = f'policy_{mode}'
            if key not in self._cached_fns:
                self._cached_fns[key] = fn
            fn = self._cached_fns[key]
        act, state = fn(obs, state, mode)
        act = self._convert_outs(act)
        return act, state

    def train(self, data, state=None):
        data = self._convert_inps(data)
        if state is None:
            state = self._strategy_run(self.agent.initial_train_state, data)
        fn = self.agent.train
        if self._cache_fns:
            key = 'train'
            if key not in self._cached_fns:
                self._cached_fns[key] = fn
            fn = self._cached_fns[key]
        outs, state, metrics = self._strategy_run(fn, data, state)
        outs = self._convert_outs(outs)
        metrics = self._convert_mets(metrics)
        return outs, state, metrics

    def report(self, data):
        data = self._convert_inps(data)
        fn = self.agent.report
        if self._cache_fns:
            key = 'report'
            if key not in self._cached_fns:
                self._cached_fns[key] = fn
            fn = self._cached_fns[key]
        metrics = self._strategy_run(fn, data)
        metrics = self._convert_mets(metrics)
        return metrics

    # -----------------------------------------------------------------------
    # Strategy Helpers
    # -----------------------------------------------------------------------
    @contextlib.contextmanager
    def _strategy_scope(self):
        # In PyTorch no special context is needed.
        yield

    def _strategy_run(self, fn, *args, **kwargs):
        # Simply call the function (any distribution strategy wrappers like DataParallel/DistributedDataParallel will handle replication).
        return fn(*args, **kwargs)

    # -----------------------------------------------------------------------
    # Input/Output Conversion Helpers
    # -----------------------------------------------------------------------
    def _convert_inps(self, value):
        def convert(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device) if hasattr(self, "device") else x
            return x
        return map_structure(convert, value)

    def _convert_outs(self, value):
        def convert(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x
        return map_structure(convert, value)

    def _convert_mets(self, value):
        def convert(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x
        return map_structure(convert, value)

    # -----------------------------------------------------------------------
    # Setup: device, precision, and distribution strategy.
    # -----------------------------------------------------------------------
    def _setup(self):
        assert self.config.precision in (16, 32), "Precision must be 16 or 32"
        if self.config.precision == 16:
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.float32)
        platform = self.config.platform
        if platform == 'cpu':
            self.device = torch.device("cpu")
            return None
        elif platform == 'gpu':
            assert torch.cuda.is_available(), "CUDA not available"
            self.device = torch.device("cuda:0")
            return None
        elif platform == 'multi_gpu':
            assert torch.cuda.device_count() > 1, "Multi-GPU platform requires >1 GPU"
            self.device = torch.device("cuda:0")
            return "dp"
        elif platform == 'multi_worker':
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.device = torch.device("cuda:0")
            return "ddp"
        elif platform == 'tpu':
            raise NotImplementedError("TPU support not available in PyTorch")
        else:
            raise NotImplementedError(platform)
