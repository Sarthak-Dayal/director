import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Independent
# Assume that torch.distributions has the methods you need.
# You should have PyTorch versions of your helper modules.
from . import nets_torch
from . import tfutils

# ---------------------------------------------------------------------------
# Disag: Disagreement Module
# ---------------------------------------------------------------------------
class Disag(tfutils.Module):  # tfutils.Module is assumed to be a subclass of nn.Module.
    def __init__(self, wm, act_space, config):
        super().__init__()
        # Make a copy of config and update it.
        config = dict(config)
        config['disag_head.inputs'] = ['tensor']
        self.config = config
        self.opt = tfutils.Optimizer('disag', **config['expl_opt'])
        self.inputs = nets_torch.Input(config['disag_head.inputs'], dims='deter')
        self.target = nets_torch.Input(config['disag_target'], dims='deter')
        self.nets_torch = None  # Will be built on first call

    def _build(self, data):
        if self.nets_torch is None:
            # Call self.target(data) to get output shape
            target_tensor = self.target(data)
            out_dim = target_tensor.shape[-1]
            # Build a list of MLPs (wrapped in a ModuleList so parameters are registered)
            self.nets_torch = nn.ModuleList([
                nets_torch.MLP(out_dim, **self.config['disag_head'])
                for _ in range(self.config['disag_models'])
            ])

    def forward(self, traj):
        self._build(traj)
        inp = self.inputs(traj)
        # For each head, assume calling head(inp) returns a distribution with a mode() method.
        preds = [head(inp).mode for head in self.nets_torch]
        # Stack predictions along a new dimension and compute std across models then average over the last dim.
        stacked = torch.stack(preds, dim=0)
        disag = torch.std(stacked, dim=0).mean(dim=-1)
        if 'action' in self.config['disag_head.inputs']:
            return disag[:-1]
        else:
            return disag[1:]

    def train_step(self, data):
        # NOTE: This method is analogous to the TensorFlow train() method.
        # Adjust the action alignment: shift 'action' by one timestep.
        data = dict(data)
        # Assuming data['action'] is a torch.Tensor of shape [B, T, ...]
        data['action'] = torch.cat([data['action'][:, 1:], torch.zeros_like(data['action'][:, :1])], dim=1)
        self._build(data)
        inp = self.inputs(data)[:, :-1]
        target = self.target(data)[:, 1:].float()  # detach/stop gradients via .detach() if needed

        self.opt.optimizer.zero_grad()
        # Compute loss as the negative log-probability (averaged over heads)
        loss = 0.0
        for head in self.nets_torch:
            pred_dist = head(inp)
            loss = loss - pred_dist.log_prob(target).mean()
        self.opt.step(loss, self.nets_torch)
        # Optionally, you can return loss.item() or a dictionary of metrics.
        return {'disag_loss': loss.item()}

# ---------------------------------------------------------------------------
# LatentVAE: A latent variational autoencoder module.
# ---------------------------------------------------------------------------
class LatentVAE(tfutils.Module):
    def __init__(self, wm, act_space, config):
        super().__init__()
        self.config = config
        # Build encoder and decoder using your nets_torch.MLP.
        self.enc = nets_torch.MLP(**self.config['expl_enc'])
        self.dec = nets_torch.MLP(self.config['rssm']['deter'], **self.config['expl_dec'])
        shape = self.config['expl_enc']['shape']
        if self.config['expl_enc']['dist'] == 'onehot':
            # Assume tfutils.OneHotDist exists in your PyTorch utilities.
            self.prior = tfutils.OneHotDist(torch.zeros(shape))
            self.prior = Independent(self.prior, len(shape) - 1)
        else:
            base = Normal(torch.zeros(shape), torch.ones(shape))
            self.prior = Independent(base, len(shape))
        self.kl = tfutils.AutoAdapt(**self.config['expl_kl'])
        self.opt = tfutils.Optimizer('disag', **self.config['expl_opt'])
        # A helper lambda to flatten the tail dimensions.
        self.flatten = lambda x: x.view(x.shape[:-len(shape)] + (-1,))

    def forward(self, traj):
        # Obtain a latent distribution from the encoder.
        dist = self.enc(traj)
        # Stop gradients on the target.
        target = traj['deter'].float().detach()
        # Sample from the encoder (using reparameterization if available).
        sample = dist.rsample() if hasattr(dist, 'rsample') else dist.sample()
        ll = self.dec(self.flatten(sample)).log_prob(target)
        if self.config['expl_vae_elbo']:
            kl = kl_divergence(dist, self.prior)
            # Assume self.kl.scale() is callable and returns a scaling factor.
            return kl - ll / self.kl.scale()
        else:
            reward = -ll
            # Return reward from time step 1 onward.
            return reward[1:]

    def train_step(self, data):
        metrics = {}
        target = data['deter'].float().detach()
        self.opt.optimizer.zero_grad()
        dist = self.enc(data)
        kl = kl_divergence(dist, self.prior)
        kl, kl_mets = self.kl(kl)
        for k, v in kl_mets.items():
            metrics[f'kl_{k}'] = v
        sample = dist.rsample() if hasattr(dist, 'rsample') else dist.sample()
        ll = self.dec(self.flatten(sample)).log_prob(target)
        assert kl.shape == ll.shape, "Shape mismatch between KL and log-likelihood"
        loss = (kl - ll).mean()
        self.opt.step(loss, [self.enc, self.dec])
        metrics['vae_kl'] = kl.mean().item()
        metrics['vae_ll'] = ll.mean().item()
        metrics['vae_loss'] = loss.item()
        return metrics

# ---------------------------------------------------------------------------
# CtrlDisag: A disagreement module for control
# ---------------------------------------------------------------------------
class CtrlDisag(tfutils.Module):
    def __init__(self, wm, act_space, config):
        super().__init__()
        # Create a copy of config and set disag_target to ['ctrl'].
        new_config = dict(config)
        new_config['disag_target'] = ['ctrl']
        self.disag = Disag(wm, act_space, new_config)
        self.embed = nets_torch.MLP((config['ctrl_size'],), **config['ctrl_embed'])
        self.head = nets_torch.MLP(act_space.shape, **config['ctrl_head'])
        self.opt = tfutils.Optimizer('ctrl', **config['ctrl_opt'])

    def forward(self, traj):
        return self.disag(traj)

    def train_step(self, data):
        metrics = {}
        self.opt.optimizer.zero_grad()
        # Embed the data and extract the mode of the distribution.
        ctrl = self.embed(data).mode
        # Build a distribution over actions using the embedded "current" and "next" features.
        # Here we assume that self.head accepts a dict input.
        dist = self.head({'current': ctrl[:, :-1], 'next': ctrl[:, 1:]})
        loss = -dist.log_prob(data['action'][:, 1:]).mean()
        self.opt.step(loss, [self.embed, self.head])
        metrics['ctrl_loss'] = loss.item()
        # Also train the underlying disag module using the embedded ctrl features.
        new_data = dict(data)
        new_data['ctrl'] = ctrl
        disag_metrics = self.disag.train_step(new_data)
        metrics.update(disag_metrics)
        return metrics

# ---------------------------------------------------------------------------
# PBE: A Predictive Behavior Embedding module.
# ---------------------------------------------------------------------------
class PBE(tfutils.Module):
    def __init__(self, wm, act_space, config):
        super().__init__()
        self.config = config
        self.inputs = nets_torch.Input(config['pbe_inputs'], dims='deter')

    def forward(self, traj):
        feat = self.inputs(traj)
        # Flatten the feature tensor to shape [N, D]
        flat = feat.view(-1, feat.shape[-1])
        # Compute pairwise Euclidean distances.
        # flat[:, None, :] has shape [N,1,D] and flat[None, :, :] has shape [1,N,D]
        dists = torch.norm(flat[:, None, :] - flat[None, :, :], dim=-1)
        # Get the top-k (nearest neighbors) distances along the last dim.
        topk_vals, _ = torch.topk(-dists, k=self.config['pbe_knn'], dim=-1, largest=True, sorted=True)
        # Compute the average of these distances and take the negative.
        rew = -topk_vals.mean(dim=-1)
        # Reshape to the original feature spatial dimensions (all dims except the last one)
        return rew.view(*feat.shape[:-1]).float()

    def train_step(self, data):
        # This module does not require training.
        return {}
