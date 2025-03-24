# tfutils.py
from datetime import datetime
import socket
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import re
import inspect

from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# Basic helper functions
# ---------------------------------------------------------------------------
def tensor(value):
    """Convert a value to a torch.Tensor if not already one."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)

def map_structure(fn, x):
    """Recursively apply fn to all elements in a nested structure (dict/list/tuple)."""
    if isinstance(x, dict):
        return {k: map_structure(fn, v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(map_structure(fn, v) for v in x)
    else:
        return fn(x)

def stack_nested(nested_list, dim=0):
    if isinstance(nested_list[0], dict):
        return {
            key: stack_nested([d[key] for d in nested_list], dim=dim)
            for key in nested_list[0]
        }
    elif isinstance(nested_list[0], (list, tuple)):
        return type(nested_list[0])(
            (stack_nested(sublist, dim=dim) for sublist in zip(*nested_list))
        )
    else:
        return torch.stack(nested_list, dim=dim)

def recursive_detach(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: recursive_detach(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_detach(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_detach(item) for item in obj)
    # Extend with other types (e.g., sets) if needed.
    else:
        return obj

def shuffle(x, axis):
    """Shuffle the tensor x along the specified axis."""
    perm = list(range(x.ndim))
    perm.pop(axis)
    perm.insert(0, axis)
    x_perm = x.permute(perm)
    idx = torch.randperm(x_perm.size(0))
    x_shuffled = x_perm.index_select(0, idx)
    # Invert permutation.
    inv_perm = [0] * len(perm)
    for i, j in enumerate(perm):
        inv_perm[j] = i
    return x_shuffled.permute(inv_perm)

def clone_tensors_recursive(data):
    """
    Recursively clones tensors inside a container while preserving the computation graph.

    Args:
        data: A tensor, list, tuple, or dictionary containing tensors.

    Returns:
        A structure of the same type with cloned tensors.
    """
    if isinstance(data, torch.Tensor):
        return data.clone().requires_grad_(data.requires_grad)
    elif isinstance(data, Mapping):  # Handles dict-like structures
        return {key: clone_tensors_recursive(value) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):  # Handles lists, tuples
        return type(data)(clone_tensors_recursive(item) for item in data)
    else:
        return data  # Return non-tensor values as-is

def scan(fn, inputs, start, static=True, reverse=False, axis=0):
    """
    A simple static scan function (like tf.scan) which applies fn iteratively
    along a given axis.
    """
    if axis == 1:
        # Transpose first two dimensions so that scan always runs along dim 0.
        def swap(x):
            dims = list(range(x.ndim))
            dims[0], dims[1] = dims[1], dims[0]
            return x.permute(dims)
        if isinstance(inputs, (list, tuple)):
            inputs = [swap(x) for x in inputs]
        else:
            inputs = swap(inputs)
    last = start
    collected = []
    # Assume inputs is a tensor or list/tuple of tensors with scan dimension at index 0.
    length = inputs[0].shape[0] if isinstance(inputs, (list, tuple)) else inputs.shape[0]
    indices = list(range(length))
    if reverse:
        indices = indices[::-1]
    for i in indices:
        inp = [x[i] for x in inputs] if isinstance(inputs, (list, tuple)) else inputs[i]
        last = fn(last, inp)
        collected.append(last)
    out = stack_nested(collected, dim=axis)
    if axis == 1:
        def unswap(x):
            dims = list(range(x.ndim))
            dims[0], dims[1] = dims[1], dims[0]
            return x.permute(dims)
        out = unswap(out)
    return out

def symlog(x):
    """Symmetric logarithm: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def action_noise(action, amount, act_space):
    """
    Apply noise to an action. For discrete action spaces, blend with a uniform distribution;
    for continuous actions, add Gaussian noise.
    """
    if amount == 0:
        return action
    amount = tensor(amount).to(action.dtype).to(action.device)
    if act_space.discrete:
        # Blend current action with uniform noise.
        probs = amount / action.shape[-1] + (1 - amount) * action
        return OneHotDist(probs=probs).sample()
    else:
        noise = td.Normal(action, amount).sample()
        return torch.clamp(noise, -1, 1)

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    """
    Compute the lambda-return for discounting rewards.
    """
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(reward.ndim))
    new_order = [axis] + dims[1:axis] + [0] + dims[axis+1:]
    if axis != 0:
        reward = reward.permute(new_order)
        value = value.permute(new_order)
        pcont = pcont.permute(new_order)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = scan(lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
                   (inputs, pcont), bootstrap, static=True, reverse=True, axis=0)
    if axis != 0:
        inv_order = [0] * len(new_order)
        for i, j in enumerate(new_order):
            inv_order[j] = i
        returns = returns.permute(inv_order)
    return returns

def video_grid(video):
    """Arrange video tensor (B, T, H, W, C) into a grid."""
    B, T, H, W, C = video.shape
    video = video.permute(1, 2, 0, 3, 4)
    return video.reshape(T, H, B * W, C)

def balance_stats(dist, target, thres):
    # Convert target to float32.
    target = target.to(torch.float32)
    # Identify positive and negative examples.
    pos = (target > thres).to(torch.float32)
    neg = (target <= thres).to(torch.float32)
    # Compute the prediction from the distribution's mean.
    pred = (dist.mean.to(torch.float32) > thres).to(torch.float32)
    # Compute loss as negative log probability.`
    loss = -dist.log_prob(target)
    # Calculate metrics; note that divisions by zero will yield NaN,
    # which is acceptable as they are ignored in later aggregation.
    pos_loss = (loss * pos).sum() / pos.sum()
    neg_loss = (loss * neg).sum() / neg.sum()
    pos_acc = (pred * pos).sum() / pos.sum()
    neg_acc = ((1 - pred) * neg).sum() / neg.sum()
    rate = pos.mean()
    avg = target.mean()
    pred_mean = dist.mean.to(torch.float32).mean()
    return dict(
        pos_loss=pos_loss.detach().cpu().item(),
        neg_loss=neg_loss.detach().cpu().item(),
        pos_acc=pos_acc.detach().cpu().item(),
        neg_acc=neg_acc.detach().cpu().item(),
        rate=rate.detach().cpu().item(),
        avg=avg.detach().cpu().item(),
        pred=pred_mean.detach().cpu().item(),
    )

# ---------------------------------------------------------------------------
# Core Module and Optimizer Classes
# ---------------------------------------------------------------------------
class GeneratorDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator
    def __iter__(self):
        return self.generator()

class Module(nn.Module):
    def __init__(self):
        super().__init__()
        # Define any submodules here if needed.

    def save(self):
        """Save parameters using state_dict and convert tensors to NumPy arrays."""
        state = self.state_dict()
        values = {name: param.detach().cpu().numpy() for name, param in state.items()}
        amount = len(values)
        count = int(sum(np.prod(val.shape) for val in values.values()))
        print(f"Saving module with {amount} tensors and {count} parameters.")
        return values

    def load(self, values):
        """Load parameters using load_state_dict after converting NumPy arrays to tensors."""
        # Convert numpy arrays back to torch tensors.
        state = {name: torch.tensor(param) for name, param in values.items()}
        print(f"Loading module with {len(state)} tensors.")
        self.load_state_dict(state)

    def get(self, name, ctor, *args, **kwargs):
        """
        Retrieve a submodule or parameter by name. If it does not exist, create it
        using the provided constructor. If the constructor accepts a 'name' argument,
        it is passed automatically.
        """
        # If already registered as a submodule, return it.
        if name in self._modules:
            return self._modules[name]

        if name in self._parameters:
            return self._parameters[name]

        # If the constructor takes a 'name' argument, pass the name.
        if 'name' in inspect.signature(ctor).parameters:
            kwargs['name'] = name

        # Create the module or tensor.
        mod = ctor(*args, **kwargs)

        # If it's an nn.Module, register it. Otherwise, if it's a tensor, wrap it as a parameter.
        # if isinstance(mod, nn.Module):
        #     setattr(self, name, mod)
        # else:
        #     mod = nn.Parameter(mod) if isinstance(mod, torch.Tensor) else mod
        #     setattr(self, name, mod)
        setattr(self, name, mod)

        return mod


class Optimizer(torch.nn.Module):
    def __init__(self, name, lr, opt='adam', eps=1e-5, clip=0.0, warmup=0,
                 wd=0.0, wd_pattern='kernel', scaling=True):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            super().__init__()
            assert 0 <= wd < 1, "weight decay must be in [0,1)"
            if clip:
                assert clip >= 1, "clip must be at least 1"
            self._name = name
            self._clip = clip
            self._warmup = warmup
            self._wd = wd
            self._wd_pattern = wd_pattern
            self._updates = 0
            self._base_lr = lr
            self._lr = lr
            self._opt_type = opt
            self._eps = eps
            self._scaling = scaling  # If True, do manual loss scaling.
            if self._scaling:
                self._grad_scale = 1e1
                self._fine_steps = 0
                # Retain your AMP GradScaler from your original implementation.
                self._scaler = torch.amp.GradScaler(init_scale=2.0)
            self._optimizer = None
            self._first_named_params = None
            self._once = True  # For one-time logging (parameter count & weight decay)

    @property
    def variables(self):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self._optimizer is not None:
                params = []
                for group in self._optimizer.param_groups:
                    params.extend(group['params'])
                return params
            return list(self.parameters())

    def step(self, loss, modules):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Ensure modules is a list.
            if not isinstance(modules, (list, tuple)):
                modules = [modules]
            varibs = []
            named_params = set()
            for module in modules:
                varibs.extend(list(module.parameters()))
                for name, _ in module.named_parameters():
                    named_params.add(name)
            # Log total parameter count on first call.
            if self._once:
                total_params = sum([p.numel() for p in varibs])
                print(f"Found {total_params} {self._name} parameters.")

            metrics = {}

            # Check loss numerics.
            if not torch.isfinite(loss):
                raise ValueError(f"{self._name} loss is not finite: {loss.item()}")
            metrics[f'{self._name}_loss'] = loss.item()

            # Create the optimizer on the first call.
            if self._optimizer is None:
                self._first_named_params = named_params
                if self._opt_type == 'adam':
                    self._optimizer = torch.optim.Adam(varibs, lr=self._lr, eps=self._eps)
                elif self._opt_type == 'sgd':
                    self._optimizer = torch.optim.SGD(varibs, lr=self._lr)
                elif self._opt_type == 'momentum':
                    self._optimizer = torch.optim.SGD(varibs, lr=self._lr, momentum=0.9)
                else:
                    raise NotImplementedError(self._opt_type)
            else:
                # Ensure parameter set consistency.
                assert self._first_named_params == named_params, "Module parameters changed!"

            self._optimizer.zero_grad()

            # ----- Loss Scaling and Backward Pass -----
            overflow = False  # Flag to track if any gradient is non-finite.
            if self._scaling:
                scaled_loss = self._grad_scale * loss
                scaled_loss.backward()
                # Unscale gradients manually.
                for p in varibs:
                    if p.grad is not None:
                        p.grad.data.div_(self._grad_scale)
            else:
                loss.backward()

            # ----- Overflow Checking and Grad Scale Update -----
            if self._scaling:
                for p in varibs:
                    if p.grad is not None:
                        if not torch.all(torch.isfinite(p.grad)):
                            overflow = True
                            break

                if not overflow:
                    if self._fine_steps < 1000:
                        self._fine_steps += 1
                        new_scale = self._grad_scale  # Keep same scale.
                    else:
                        self._fine_steps = 0
                        new_scale = self._grad_scale * 2  # Double scale after 1000 fine steps.
                else:
                    self._fine_steps = 0
                    new_scale = self._grad_scale / 2  # Halve scale on overflow.

                self._grad_scale = max(1e-4, min(new_scale, 1e4))
                metrics[f'{self._name}_grad_scale'] = self._grad_scale
                metrics[f'{self._name}_grad_overflow'] = float(overflow)

            # ----- Gradient Clipping -----
            total_norm = 0.0
            for p in varibs:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if self._clip:
                torch.nn.utils.clip_grad_norm_(varibs, self._clip)
            if self._scaling and not torch.isfinite(torch.tensor(total_norm)):
                total_norm = float('nan')
            metrics[f'{self._name}_grad_norm'] = total_norm

            # ----- Weight Decay (with Logging) -----
            if self._wd and not overflow:
                current_lr = self._lr
                log = (self._wd_pattern != r'.*') and self._once
                included, excluded = [], []
                for module in modules:
                    for name, param in module.named_parameters():
                        full_name = self._name + '/' + name
                        # Use __import__('re') to call regex functions.
                        if __import__('re').search(self._wd_pattern, full_name):
                            param.data.mul_(1 - self._wd * current_lr)
                            included.append(full_name)
                        else:
                            excluded.append(full_name)
                if log:
                    print(f"Optimizer applied weight decay to {self._name} variables:")
                    for nm in included:
                        print(f'[x] {nm}')
                    for nm in excluded:
                        print(f'[ ] {nm}')
                    print('')

            # ----- Apply Gradients and Update LR if No Overflow -----
            if not overflow:
                self._optimizer.step()
                self._updates += 1
                metrics[f'{self._name}_grad_steps'] = self._updates
                if self._warmup:
                    warmup_factor = min(self._updates / self._warmup, 1.0)
                    self._lr = self._base_lr * warmup_factor
                    for param_group in self._optimizer.param_groups:
                        param_group['lr'] = self._lr
            else:
                metrics[f'{self._name}_grad_steps'] = self._updates

            # Mark that one-time logging has been completed.
            self._once = False

            return metrics

# ---------------------------------------------------------------------------
# Distribution Classes (wrappers)
# ---------------------------------------------------------------------------

# Mean-squared error distribution.
class MSEDist(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = True

    def __init__(self, pred, dims, agg='sum', validate_args=None):
        self.pred = pred  # Tensor expected.
        self._dims = dims
        self._axes = tuple(-i for i in range(1, dims + 1))
        self._agg = agg
        self._validate_args = validate_args
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self):
        return self.pred.shape[:-self._dims] if self._dims > 0 else self.pred.shape

    @property
    def event_shape(self):
        return self.pred.shape[-self._dims:] if self._dims > 0 else torch.Size()

    @property
    def mean(self):
        return self.pred

    @property
    def mode(self):
        return self.pred

    def rsample(self, sample_shape=torch.Size()):
        return self.pred.expand(sample_shape + self.pred.shape)

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self.pred.shape != value.shape:
            raise ValueError(f"Shape mismatch: {self.pred.shape} vs {value.shape}")
        distance = (self.pred - value) ** 2
        if self._agg == 'mean':
            loss = distance.mean(dim=self._axes)
        elif self._agg == 'sum':
            loss = distance.sum(dim=self._axes)
        else:
            raise NotImplementedError(self._agg)
        return -loss

    def expand(self, batch_shape, _instance=None):
        new_pred = self.pred.expand(batch_shape + self.event_shape)
        return type(self)(new_pred, self._dims, self._agg, validate_args=self._validate_args)


# Cosine similarity based distribution.
class CosineDist(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = True

    def __init__(self, pred, validate_args=None):
        # Normalize along the last dimension.
        self.pred = F.normalize(pred, p=2, dim=-1)
        self._validate_args = validate_args
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self):
        return self.pred.shape[:-1]

    @property
    def event_shape(self):
        return self.pred.shape[-1:]

    @property
    def mean(self):
        return self.pred

    @property
    def mode(self):
        return self.pred

    def rsample(self, sample_shape=torch.Size()):
        return self.pred.expand(sample_shape + self.pred.shape)

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self.pred.shape != value.shape:
            raise ValueError(f"Shape mismatch: {self.pred.shape} vs {value.shape}")
        return (self.pred * value).sum(dim=-1)

    def expand(self, batch_shape, _instance=None):
        new_pred = self.pred.expand(batch_shape + self.event_shape)
        return type(self)(new_pred, validate_args=self._validate_args)


# Directional distribution built on a Normal and reparameterized via rsample.
class DirDist(td.Independent):
    has_rsample = True

    def __init__(self, mean, std, validate_args=None):
        # Normalize and broadcast parameters.
        mean = mean.float()
        std = std.float()
        norm_mean = F.normalize(mean, p=2, dim=-1)
        norm_mean, std = torch.broadcast_tensors(norm_mean, std)
        self._mean = norm_mean
        self.std_tensor = std
        self._validate_args = validate_args
        base = td.Normal(norm_mean, std, validate_args=validate_args)
        # Wrap base distribution with one event dimension.
        super().__init__(base, 1, validate_args=validate_args)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mean

    def rsample(self, sample_shape=torch.Size()):
        sample = super().rsample(sample_shape)
        return F.normalize(sample, p=2, dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        norm_value = F.normalize(value.float(), p=2, dim=-1)
        return super().log_prob(norm_value)

    def expand(self, batch_shape, _instance=None):
        new_mean = self._mean.expand(batch_shape + self._mean.shape[-1:])
        new_std = self.std_tensor.expand(batch_shape + self.std_tensor.shape[-1:])
        return type(self)(new_mean, new_std, validate_args=self._validate_args)


# Symlog distribution.
class SymlogDist(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = True

    def __init__(self, mode, dims, agg='sum', validate_args=None):
        self._mode = mode
        self._dims = dims  # number of event dimensions.
        self._agg = agg
        self._axes = tuple(-i for i in range(1, dims + 1))
        self._validate_args = validate_args
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self):
        return self._mode.shape[:-self._dims] if self._dims > 0 else self._mode.shape

    @property
    def event_shape(self):
        return self._mode.shape[-self._dims:] if self._dims > 0 else torch.Size()

    @property
    def mean(self):
        return symexp(self._mode)

    @property
    def mode(self):
        return symexp(self._mode)

    def rsample(self, sample_shape=torch.Size()):
        return symexp(self._mode).expand(sample_shape + self._mode.shape)

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self._mode.shape != value.shape:
            raise ValueError(f"Shape mismatch: {self._mode.shape} vs {value.shape}")
        distance = (self._mode - symlog(value)) ** 2
        if self._agg == 'mean':
            loss = distance.mean(dim=self._axes)
        elif self._agg == 'sum':
            loss = distance.sum(dim=self._axes)
        else:
            raise NotImplementedError(self._agg)
        return -loss

    def expand(self, batch_shape, _instance=None):
        new_mode = self._mode.expand(batch_shape + self.event_shape)
        return type(self)(new_mode, self._dims, self._agg, validate_args=self._validate_args)



class OneHotDist(td.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class NormalDist(td.Normal):
    arg_constraints = {'loc': td.constraints.real, 'scale': td.constraints.positive}
    support = td.constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        # Broadcast loc and scale so that they have a common shape.
        loc, scale = torch.broadcast_tensors(loc, scale)
        self._validate_args = validate_args
        super().__init__(loc, scale, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2

    @property
    def stddev(self):
        return self.scale

    def sample(self, sample_shape=torch.Size()):
        # Ensure that even sample() returns a reparameterized sample.
        return self.rsample(sample_shape)

    def expand(self, batch_shape, _instance=None):
        new_loc = self.loc.expand(batch_shape)
        new_scale = self.scale.expand(batch_shape)
        return type(self)(new_loc, new_scale, validate_args=self._validate_args)

# Truncated Normal distribution wrapper.
class TruncatedNormalDist(td.Distribution):
    arg_constraints = {'loc': td.constraints.real, 'scale': td.constraints.positive}
    support = td.constraints.interval(-1, 1)
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.broadcast_tensors(loc, scale)
        self._base = td.Normal(self.loc, self.scale, validate_args=validate_args)
        self._validate_args = validate_args
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self):
        return self.loc.shape

    @property
    def event_shape(self):
        return torch.Size([])

    @property
    def mean(self):
        return self.loc.clamp(-1, 1)

    @property
    def mode(self):
        return self.loc.clamp(-1, 1)

    def rsample(self, sample_shape=torch.Size()):
        sample = self._base.rsample(sample_shape)
        return sample.clamp(-1, 1)

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        return self._base.log_prob(value)

    def expand(self, batch_shape, _instance=None):
        new_loc = self.loc.expand(batch_shape)
        new_scale = self.scale.expand(batch_shape)
        return type(self)(new_loc, new_scale, validate_args=self._validate_args)


# Differentiable Bernoulli distribution.
class BernoulliDist(td.Bernoulli):
    has_rsample = True

    def __init__(self, logits=None, probs=None, validate_args=None):
        self._validate_args = validate_args
        super().__init__(logits=logits, probs=probs, validate_args=validate_args)

    @property
    def batch_shape(self):
        if self.logits is not None:
            return self.logits.shape
        else:
            return self.probs.shape

    @property
    def event_shape(self):
        return torch.Size([])

    @property
    def mean(self):
        return self.probs

    @property
    def mode(self):
        return (self.probs >= 0.5).float()

    def rsample(self, sample_shape=torch.Size()):
        if self.probs is None:
            raise ValueError("Bernoulli distribution requires logits or probabilities.")
        gumbel_noise = -torch.log(
            -torch.log(torch.rand(sample_shape + self.probs.shape, device=self.probs.device) + 1e-10) + 1e-10
        )
        sample = torch.sigmoid((self.logits + gumbel_noise))
        return sample

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def expand(self, batch_shape, _instance=None):
        if self.logits is not None:
            new_logits = self.logits.expand(batch_shape)
            return type(self)(logits=new_logits, validate_args=self._validate_args)
        else:
            new_probs = self.probs.expand(batch_shape)
            return type(self)(probs=new_probs, validate_args=self._validate_args)

# ---------------------------------------------------------------------------
# AutoAdapt and Normalize
# ---------------------------------------------------------------------------
class AutoAdapt(Module):
    def __init__(self, shape, impl, scale, target, min, max, vel=0.1, thres=0.1, inverse=False):
        super().__init__()
        self._shape = shape
        self._impl = impl
        self._target = target
        self._min = min
        self._max = max
        self._vel = vel
        self._inverse = inverse
        self._thres = thres
        if self._impl == 'fixed':
            self._scale = tensor(scale)
        elif self._impl in ['mult', 'prop']:
            self._scale = nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False)
        else:
            raise NotImplementedError(self._impl)

    def forward(self, reg, update=True):
        if update:
            self.update(reg)
        scale = self.scale()
        loss = scale * (-reg if self._inverse else reg)
        metrics = {
            'mean': reg.mean().item(),
            'std': reg.std().item(),
            'scale_mean': scale.mean().item(),
            'scale_std': scale.std().item(),
        }
        return loss, metrics

    def scale(self):
        return self._scale.detach()

    def update(self, reg):
        dims = len(reg.shape) - len(self._shape)
        avg = reg.mean(dim=tuple(range(dims))) if dims > 0 else reg
        if self._impl == 'fixed':
            return
        elif self._impl == 'mult':
            below = (avg < (1 / (1 + self._thres)) * self._target).float()
            above = (avg > (1 + self._thres) * self._target).float()
            if self._inverse:
                below, above = above, below
            inside = 1.0 - (below + above)
            adjusted = (above * self._scale * (1 + self._vel) +
                        below * self._scale / (1 + self._vel) +
                        inside * self._scale)
            self._scale.data.copy_(torch.clamp(adjusted, self._min, self._max))
        elif self._impl == 'prop':
            direction = avg - self._target
            if self._inverse:
                direction = -direction
            self._scale.data.copy_(torch.clamp(self._scale + self._vel * direction, self._min, self._max))
        else:
            raise NotImplementedError(self._impl)

class Normalize:
    def __init__(self, impl='mean_std', decay=0.99, max=1e8, vareps=0.0, stdeps=0.0):
        self._impl = impl
        self._decay = decay
        self._max = max
        self._stdeps = stdeps
        self._vareps = vareps
        self._mean = torch.tensor(0.0, dtype=torch.float64)
        self._sqrs = torch.tensor(0.0, dtype=torch.float64)
        self._step = 0

    def __call__(self, values, update=True):
        if update:
            self.update(values)
        return self.transform(values)

    def update(self, values):
        with torch.no_grad():
            x = values.to(torch.float64)
            self._step += 1
            self._mean = self._decay * self._mean + (1 - self._decay) * x.mean().double()
            self._sqrs = self._decay * self._sqrs + (1 - self._decay) * (x ** 2).mean().double()

    def transform(self, values):
        correction = 1 - self._decay ** self._step
        mean = self._mean / correction
        var = (self._sqrs / correction) - mean ** 2
        if self._max > 0.0:
            scale = torch.rsqrt(torch.clamp(var, min=1 / (self._max ** 2) + self._vareps) + self._stdeps)
        else:
            scale = torch.rsqrt(var + self._vareps) + self._stdeps
        if self._impl == 'off':
            return values
        elif self._impl == 'mean_std':
            return (values - mean.to(values.dtype)) * scale.to(values.dtype)
        elif self._impl == 'std':
            return values * scale.to(values.dtype)
        else:
            raise NotImplementedError(self._impl)

# ---------------------------------------------------------------------------
# Input and get_act
# ---------------------------------------------------------------------------
class Input(Module):
    def __init__(self, keys=['tensor'], dims=None):
        super().__init__()
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)
        self._dims = dims or self._keys[0]

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {'tensor': inputs}
        if not all(k in inputs for k in self._keys):
            needs = f'{{{", ".join(self._keys)}}}'
            found = f'{{{", ".join(inputs.keys())}}}'
            raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
        values = [inputs[k] for k in self._keys]
        dims = len(inputs[self._dims].shape)
        new_vals = []
        for value in values:
            if value.dim() > dims:
                new_shape = list(value.shape[:dims - 1]) + [int(np.prod(value.shape[dims - 1:]))]
                value = value.view(*new_shape)
            new_vals.append(value.to(inputs[self._dims].dtype))
        return torch.cat(new_vals, dim=-1)

# Dataset wrapper for generator.
class GeneratorDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator
    def __iter__(self):
        return self.generator()

def get_act(name):
    if callable(name):
        return name
    elif name == 'none':
        return lambda x: x
    elif name == 'mish':
        return lambda x: x * torch.tanh(F.softplus(x))
    elif name == 'gelu':
        return F.gelu
    elif hasattr(F, name):
        return getattr(F, name)
    elif hasattr(torch, name):
        return getattr(torch, name)
    else:
        raise NotImplementedError(name)

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   # prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
