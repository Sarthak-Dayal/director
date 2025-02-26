# file: agent_torch.py
import sys
import functools
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data as data
import ruamel.yaml as yaml

import embodied  # Assumed to be your RL framework with a similar PyTorch API

# These are your converted tfutils components.
from tfutils import (
    Optimizer,
    AutoAdapt,
    Normalize,
    action_noise,
    scan,
    Module,
    map_structure  # A helper function similar to tf.nest.map_structure
)

# Network modules converted from TensorFlow (e.g. in nets_torch.py)
from nets import RSSM, MultiEncoder, MultiDecoder, MLP
import nets  # For additional network components if needed

# Converted behavior modules and the base agent wrapper.
from behaviors import *  # Your behavior modules converted to PyTorch
from tfagent import TFAgent  # The PyTorch version of your TFAgent base class

# =============================================================================
# Agent
# =============================================================================
class Agent(TFAgent):
    # Load configuration from YAML
    configs = yaml.YAML(typ='safe').load(
        (embodied.Path(sys.argv[0]).parent / 'configs.yaml').read()
    )

    def __init__(self, obs_space, act_space, step, config):
        self.config = config  # Retain config as given
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        # Create the world model (which includes the RSSM, encoder, decoder heads, etc.)
        self.wm = WorldModel(obs_space, config)
        # Instantiate task behavior and exploration behavior (converted to PyTorch)
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config)
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config)
        # initial_policy_state returns a tuple: (wm initial state, task behavior initial state, expl behavior initial state, zeros)
        self.initial_policy_state = lambda obs: (
            self.wm.rssm.initial(len(obs['is_first'])),
            self.task_behavior.initial(len(obs['is_first'])),
            self.expl_behavior.initial(len(obs['is_first'])),
            torch.zeros(len(obs['is_first']), *self.act_space.shape, device=torch.device("cpu"))
        )
        # initial_train_state returns just the world model’s initial state
        self.initial_train_state = lambda obs: self.wm.rssm.initial(len(obs['is_first']))

    def policy(self, obs, state=None, mode='train'):
        if self.config.tf.jit:
            print('Tracing policy function.')
        if state is None:
            state = self.initial_policy_state(obs)
        obs = self.preprocess(obs)
        latent, task_state, expl_state, action = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'])
        noise = self.config.expl_noise
        if mode == 'eval':
            noise = self.config.eval_noise
            outs, task_state = self.task_behavior.policy(latent, task_state)
            # Use deterministic mode for evaluation
            outs = {**outs, 'action': outs['action'].mode()}
        elif mode == 'explore':
            outs, expl_state = self.expl_behavior.policy(latent, expl_state)
            outs = {**outs, 'action': outs['action'].sample()}
        elif mode == 'train':
            outs, task_state = self.task_behavior.policy(latent, task_state)
            outs = {**outs, 'action': outs['action'].sample()}
        outs = {**outs, 'action': action_noise(outs['action'], noise, self.act_space)}
        state = (latent, task_state, expl_state, outs['action'])
        return outs, state

    def train(self, data, state=None):
        if self.config.tf.jit:
            print('Tracing train function.')
        metrics = {}
        if state is None:
            state = self.initial_train_state(data)
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        # Combine the world model outputs with data to form the planning context.
        context = {**data, **wm_outs['post']}
        # Flatten the context along the time dimension.
        start = map_structure(lambda x: x.view(-1, *x.shape[2:]), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != 'None':
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({f'expl_{k}': v for k, v in mets.items()})
        outs = {}
        if 'key' in data:
            criteria = {**data, **wm_outs}
            outs.update(key=data['key'], priority=criteria[self.config.priority])
        return outs, state, metrics

    def report(self, data):
        if self.config.tf.jit:
            print('Tracing report function.')
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f'task_{k}': v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f'expl_{k}': v for k, v in mets.items()})
        return report

    def dataset(self, generator):
        if self.config.data_loader == 'tfdata':
            # Create a PyTorch IterableDataset from the generator.
            class GenDataset(data.IterableDataset):
                def __iter__(self):
                    return generator()
            return data.DataLoader(GenDataset(), batch_size=self.config.batch_size,
                                     num_workers=8, prefetch_factor=4)
        elif self.config.data_loader == 'embodied':
            return embodied.Prefetch(sources=[generator] * self.config.batch_size,
                                      workers=8, prefetch=4)
        else:
            raise NotImplementedError(self.config.data_loader)

    def preprocess(self, obs):
        # Convert inputs to torch tensors and cast to appropriate dtype.
        dtype = torch.get_default_dtype()
        obs = {k: torch.as_tensor(v) for k, v in obs.items()}
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if value.dim() > 3 and value.dtype == torch.uint8:
                value = value.to(dtype) / 255.0
            else:
                value = value.to(torch.float32)
            obs[key] = value
        transform = {
            'off': lambda r: r,
            'sign': torch.sign,
            'tanh': torch.tanh,
            'symlog': lambda r: torch.sign(r) * torch.log1p(torch.abs(r))
        }[self.config.transform_rewards]
        obs['reward'] = transform(obs['reward'])
        obs['cont'] = 1.0 - obs['is_terminal'].to(torch.float32)
        return obs

# =============================================================================
# WorldModel
# =============================================================================
class WorldModel(Module):
    def __init__(self, obs_space, config):
        # Compute shapes from the observation space
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
        self.config = config
        self.rssm = nets.RSSM(**config.rssm)
        self.encoder = nets.MultiEncoder(shapes, **config.encoder)
        self.heads = {}
        self.heads['decoder'] = nets.MultiDecoder(shapes, **config.decoder)
        self.heads['reward'] = nets.MLP((), **config.reward_head)
        self.heads['cont'] = nets.MLP((), **config.cont_head)
        self.model_opt = Optimizer('model', **config.model_opt)
        self.wmkl = AutoAdapt((), **self.config.wmkl, inverse=False)

    def train(self, data, state=None):
        model_loss, state, outputs, metrics = self.loss(data, state, training=True)
        modules = [self.encoder, self.rssm] + list(self.heads.values())
        metrics.update(self.model_opt.step(model_loss, modules))
        return state, outputs, metrics

    def loss(self, data, state=None, training=False):
        metrics = {}
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data['action'], data['is_first'], state)
        dists = {}
        post_const = map_structure(lambda x: x.detach(), post)
        for name, head in self.heads.items():
            # If this head should receive gradients, use post; otherwise use detached post.
            out = head(post if name in self.config.grad_heads else post_const)
            if not isinstance(out, dict):
                out = {name: out}
            dists.update(out)
        losses = {}
        kl = self.rssm.kl_loss(post, prior, self.config.wmkl_balance)
        kl, mets = self.wmkl(kl, update=training)
        losses['kl'] = kl
        metrics.update({f'wmkl_{k}': v for k, v in mets.items()})
        for key, dist in dists.items():
            losses[key] = -dist.log_prob(data[key].to(torch.float32))
        metrics.update({f'{k}_loss_mean': v.mean().item() for k, v in losses.items()})
        metrics.update({f'{k}_loss_std': v.std().item() for k, v in losses.items()})
        scaled = {key: loss * self.config.loss_scales.get(key, 1.0)
                  for key, loss in losses.items()}
        model_loss = sum(scaled.values())
        if 'prob' in data and self.config.priority_correct:
            weights = (1.0 / data['prob']) ** self.config.priority_correct
            weights /= weights.max()
            model_loss = model_loss * weights
        out = {'embed': embed, 'post': post, 'prior': prior}
        for k, v in losses.items():
            out[f'{k}_loss'] = v
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean().item()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean().item()
        metrics['model_loss_mean'] = model_loss.mean().item()
        metrics['model_loss_std'] = model_loss.std().item()
        if not self.config.tf.debug_nans:
            if 'reward' in dists:
                stats = Optimizer.balance_stats(dists['reward'], data['reward'], 0.1)
                metrics.update({f'reward_{k}': v for k, v in stats.items()})
            if 'cont' in dists:
                stats = Optimizer.balance_stats(dists['cont'], data['cont'], 0.5)
                metrics.update({f'cont_{k}': v for k, v in stats.items()})
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss.mean(), last_state, out, metrics

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start['is_terminal']).to(torch.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start['action'] = policy(start)
        def step(prev, _):
            prev = prev.copy()
            action = prev.pop('action')
            state = self.rssm.img_step(prev, action)
            action = policy(state)
            return {**state, 'action': action}
        traj = scan(step, torch.arange(horizon), start, static=self.config.imag_unroll)
        traj = {k: torch.cat([start[k].unsqueeze(0), v], dim=0) for k, v in traj.items()}
        traj['cont'] = torch.cat([first_cont.unsqueeze(0), self.heads['cont'](traj).mean()[1:]], dim=0)
        traj['weight'] = torch.cumprod(self.config.discount * traj['cont'], dim=0) / self.config.discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start['is_terminal']).to(torch.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        keys = keys + list(carry.keys()) + ['action']
        states = [start]
        outs, carry = policy(start, carry)
        action = outs['action']
        if hasattr(action, 'sample'):
            action = action.sample()
        actions = [action]
        carries = [carry]
        for _ in range(horizon):
            states.append(self.rssm.img_step(states[-1], actions[-1]))
            outs, carry = policy(states[-1], carry)
            action = outs['action']
            if hasattr(action, 'sample'):
                action = action.sample()
            actions.append(action)
            carries.append(carry)
        def transp(x):
            return {k: [x[t][k] for t in range(len(x))] for k in x[0]}
        traj = {**transp(states), **transp(carries), 'action': actions}
        traj = {k: torch.stack(v, dim=0) for k, v in traj.items()}
        cont = self.heads['cont'](traj).mean()
        cont = torch.cat([first_cont.unsqueeze(0), cont[1:]], dim=0)
        traj['cont'] = cont
        traj['weight'] = torch.cumprod(self.config.imag_discount * cont, dim=0) / self.config.imag_discount
        return traj

    def report(self, data):
        report = {}
        # Use the loss function to compute report metrics (here we ignore state outputs)
        _, _, out, _ = self.loss(data)
        report.update(out)
        context, _ = self.rssm.observe(
            self.encoder(data)[:6, :5],
            data['action'][:6, :5],
            data['is_first'][:6, :5]
        )
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads['decoder'](context)
        openl = self.heads['decoder'](self.rssm.imagine(data['action'][:6, 5:], start))
        for key in self.heads['decoder'].cnn_shapes.keys():
            truth = data[key][:6].to(torch.float32)
            model = torch.cat([recon[key].mode()[:, :5], openl[key].mode()], dim=1)
            error = (model - truth + 1) / 2
            video = torch.cat([truth, model, error], dim=2)
            report[f'openl_{key}'] = Optimizer.video_grid(video)
        return report

# =============================================================================
# ImagActorCritic
# =============================================================================
class ImagActorCritic(Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        self.actor = nets.MLP(act_space.shape, **self.config.actor,
                              dist=(config.actor_dist_disc if act_space.discrete
                                    else config.actor_dist_cont))
        self.grad = config.actor_grad_disc if act_space.discrete else config.actor_grad_cont
        self.advnorm = Normalize(**self.config.advnorm)
        self.retnorms = {k: Normalize(**self.config.retnorm) for k in self.critics}
        self.scorenorms = {k: Normalize(**self.config.scorenorm) for k in self.critics}
        if self.config.actent_perdim:
            shape = act_space.shape[:-1] if act_space.discrete else act_space.shape
            self.actent = AutoAdapt(shape, **self.config.actent, inverse=True)
        else:
            self.actent = AutoAdapt((), **self.config.actent, inverse=True)
        self.opt = Optimizer('actor', **self.config.actor_opt)

    def initial(self, batch_size):
        return None

    def policy(self, state, carry):
        return {'action': self.actor(state)}, carry

    def train(self, imagine, start, context):
        policy = lambda s: self.actor(map_structure(lambda x: x.detach(), s)).sample()
        with torch.enable_grad():
            traj = imagine(policy, start, self.config.imag_horizon)
        metrics = self.update(traj)
        return traj, metrics

    def update(self, traj, tape=None):
        # Ensure gradients are enabled
        with torch.enable_grad():
            metrics = {}
            for key, critic in self.critics.items():
                mets = critic.train(traj, self.actor)
                metrics.update({f'{key}_{k}': v for k, v in mets.items()})
            scores = []
            for key, critic in self.critics.items():
                ret, baseline = critic.score(traj, self.actor)
                ret = self.retnorms[key](ret)
                baseline = self.retnorms[key](baseline, update=False)
                score = self.scorenorms[key](ret - baseline)
                metrics[f'{key}_score_mean'] = score.mean().item()
                metrics[f'{key}_score_std'] = score.std().item()
                metrics[f'{key}_score_mag'] = torch.abs(score).mean().item()
                metrics[f'{key}_score_max'] = torch.abs(score).max().item()
                scores.append(score * self.scales[key])
            score = self.advnorm(sum(scores))
            loss, mets = self.loss(traj, score)
            metrics.update(mets)
            loss = loss.mean()
        metrics.update(self.opt.step(loss, [self.actor]))
        return metrics

    def loss(self, traj, score):
        metrics = {}
        policy = self.actor(map_structure(lambda x: x.detach(), traj))
        action = traj['action'].detach()
        if self.grad == 'backprop':
            loss = -score
        elif self.grad == 'reinforce':
            loss = -policy.log_prob(action)[:-1] * score.detach()
        else:
            raise NotImplementedError(self.grad)
        shape = self.act_space.shape[:-1] if self.act_space.discrete else self.act_space.shape
        if self.config.actent_perdim and len(shape) > 0:
            assert isinstance(policy, torch.distributions.Independent)
            ent = policy.base_dist.entropy()[:-1]
            if self.config.actent_norm:
                lo = policy.minent / ent.shape[-1]
                hi = policy.maxent / ent.shape[-1]
                ent = (ent - lo) / (hi - lo)
            ent_loss, mets = self.actent(ent)
            ent_loss = ent_loss.sum(-1)
        else:
            ent = policy.entropy()[:-1]
            if self.config.actent_norm:
                lo, hi = policy.minent, policy.maxent
                ent = (ent - lo) / (hi - lo)
            ent_loss, mets = self.actent(ent)
        metrics.update({f'actent_{k}': v for k, v in mets.items()})
        loss += ent_loss
        loss *= traj['weight'][:-1].detach()
        return loss, metrics
