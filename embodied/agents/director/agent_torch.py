import sys
from ruamel import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

import embodied

from tfutils import Optimizer, balance_stats, Module, scan, AutoAdapt, symlog, Normalize, action_noise, GeneratorDataset
from .tfutils import recursive_detach, video_grid

# =============================================================================
# Agent and World Model (PyTorch version)
# =============================================================================

# Assume that you have equivalent PyTorch implementations in your "nets_torch" and "behaviors" modules.
from . import behaviors, nets_torch, tfagent  # These modules must be reimplemented for PyTorch.

class Agent(tfagent.TFAgent):

    configs = yaml.YAML(typ='safe').load((
        embodied.Path(sys.argv[0]).parent / 'configs.yaml').read())

    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = WorldModel(obs_space, config)
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config)
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config)

    def initial_policy_state(self, obs):
        batch_size = len(obs['is_first'])
        latent = self.wm.rssm.initial(batch_size)
        task_state = self.task_behavior.initial(batch_size)
        expl_state = self.expl_behavior.initial(batch_size)
        # Make sure the zeros tensor is created on the proper device.
        device = latent[next(iter(latent))].device if isinstance(latent, dict) else torch.device("cpu")
        action = torch.zeros((batch_size,) + self.act_space.shape, device=device)
        return (latent, task_state, expl_state, action)

    def initial_train_state(self, obs):
        batch_size = len(obs['is_first'])
        return self.wm.rssm.initial(batch_size)

    def policy(self, obs, state=None, mode='train'):
        if state is None:
            state = self.initial_policy_state(obs)
        obs = self.preprocess(obs)
        latent, task_state, expl_state, last_action = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(latent, last_action, embed, obs['is_first'])
        noise = self.config.expl_noise
        if mode == 'eval':
            noise = self.config.eval_noise
            outs, task_state = self.task_behavior.policy(latent, task_state)
            outs['action'] = outs['action'].mode  # Assume distribution API
        elif mode == 'explore':
            outs, expl_state = self.expl_behavior.policy(latent, expl_state)
            outs['action'] = outs['action'].sample()
        elif mode == 'train':
            outs, task_state = self.task_behavior.policy(latent, task_state)
            outs['action'] = outs['action'].sample()
        outs['action'] = action_noise(outs['action'], noise, self.act_space)
        state = (latent, task_state, expl_state, outs['action'])
        return outs, state

    def train(self, data, state=None):
        metrics = {}
        if state is None:
            state = self.initial_train_state(data)
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs['post']}
        # Reshape each tensor: merge the first two dimensions.
        start = {k: v.reshape(-1, *v.shape[2:]) for k, v in context.items()}
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != 'None':
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({f'expl_{k}': v for k, v in mets.items()})
        outs = {}
        if 'key' in data:
            criteria = {**data, **wm_outs}
            outs.update({'key': data['key'], 'priority': criteria[self.config.priority]})
        return outs, state, metrics

    def report(self, data):
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
            return DataLoader(GeneratorDataset(generator), batch_size=self.config.batch_size)
        elif self.config.data_loader == 'embodied':
            return embodied.Prefetch(sources=[generator] * self.config.batch_size, workers=8, prefetch=4)
        else:
            raise NotImplementedError(self.config.data_loader)

    def preprocess(self, obs):
        # Use torch.get_default_dtype() (typically torch.float32)
        dtype = torch.get_default_dtype()
        obs = {k: torch.as_tensor(v) for k, v in obs.items()}
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if value.ndim > 3 and value.dtype == torch.uint8:
                value = value.to(dtype) / 255.0
            else:
                value = value.to(torch.float32)
            obs[key] = value
        reward_transform = {
            'off': lambda r: r,
            'sign': torch.sign,
            'tanh': torch.tanh,
            'symlog': symlog,
        }
        obs['reward'] = reward_transform[self.config.transform_rewards](obs['reward'])
        obs['cont'] = 1.0 - obs['is_terminal'].to(torch.float32)
        return obs

# =============================================================================
# WorldModel and related modules
# =============================================================================

class WorldModel(Module):
    def __init__(self, obs_space, config):
        super().__init__()
        # Extract shapes from the observation space.
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
        self.config = config
        self.rssm = nets_torch.RSSM(**config.rssm)
        self.encoder = nets_torch.MultiEncoder(shapes, **config.encoder)
        self.heads = {}
        self.heads['decoder'] = nets_torch.MultiDecoder(shapes, **config.decoder)
        self.heads['reward'] = nets_torch.MLP((), **config.reward_head)
        self.heads['cont'] = nets_torch.MLP((), **config.cont_head)

        self.model_opt = Optimizer('model', **config.model_opt)
        self.wmkl = AutoAdapt((), **self.config.wmkl, inverse=False)

    def train(self, data, state=None):
        super(WorldModel, self).train()  # set module to training mode
        loss, state, outputs, metrics = self.loss(data, state, training=True)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        self.model_opt.step(loss, modules)
        state = recursive_detach(state)
        outputs = recursive_detach(outputs)
        return state, outputs, metrics

    def loss(self, data, state=None, training=False):
        metrics = {}
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data['action'], data['is_first'], state)
        dists = {}
        # Create a detached copy of post.
        post_const = {k: v.detach() for k, v in post.items()}
        for name, head in self.heads.items():
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
        scaled = {}
        for key, loss_val in losses.items():
            scaled[key] = loss_val * self.config.loss_scales.get(key, 1.0)
        model_loss = sum(scaled.values())
        if 'prob' in data and self.config.priority_correct:
            weights = (1.0 / data['prob'].to(torch.float32)) ** self.config.priority_correct
            weights = weights / weights.max()
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
                stats = balance_stats(dists['reward'], data['reward'], 0.1)
                metrics.update({f'reward_{k}': v for k, v in stats.items()})
            if 'cont' in dists:
                stats = balance_stats(dists['cont'], data['cont'], 0.5)
                metrics.update({f'cont_{k}': v for k, v in stats.items()})
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss.mean(), last_state, out, metrics

    def imagine(self, policy, start, horizon):
        # Compute continuation mask as float tensor.
        first_cont = (1.0 - start['is_terminal']).float()
        # Only keep keys present in the initial state.
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start['action'] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            # Pop the previous action.
            action = prev.pop('action')
            # Perform one imaginated step using the RSSM.
            state = self.rssm.img_step(prev, action)
            # Compute next action from policy.
            action = policy(state)
            return {**state, 'action': action}

        # Use the pytorch version of tfutils.scan.
        traj = scan(
            step, torch.arange(horizon), start, self.config.imag_unroll
        )
        # Concatenate the initial state to the beginning of each trajectory.
        traj = {k: torch.cat([start[k].unsqueeze(0), v], dim=0) for k, v in traj.items()}
        # Compute the continuation values: first element is first_cont, then the mean from heads.
        traj['cont'] = torch.cat([
            first_cont.unsqueeze(0),
            self.heads['cont'](traj).mean[1:]
        ], dim=0)
        # Compute trajectory weights using cumulative product of discounted continuation values.
        traj['weight'] = torch.cumprod(self.config.discount * traj['cont'], dim=0) / self.config.discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start['is_terminal'].to(torch.float32))
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        keys += list(carry.keys()) + ['action']
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
        # Transpose list-of-dicts into dict-of-lists.
        def transp(x):
            keys = x[0].keys()
            return {k: [d[k] for d in x] for k in keys}
        traj = {**transp(states), **transp(carries), 'action': actions}
        for k, v in traj.items():
            traj[k] = torch.stack(v, dim=0)
        cont = self.heads['cont'](traj).mean
        cont = torch.cat([first_cont.unsqueeze(0), cont[1:]], dim=0)
        traj['cont'] = cont
        traj['weight'] = torch.cumprod(self.config.imag_discount * cont, dim=0) / self.config.imag_discount
        return traj

    def report(self, data):
        report = {}
        # We ignore the loss value here and just report the outputs.
        loss_out = self.loss(data)[-1]
        report.update(loss_out)
        context, _ = self.rssm.observe(self.encoder(data)[:6, :5],
                                        data['action'][:6, :5],
                                        data['is_first'][:6, :5])
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads['decoder'](context)
        openl = self.heads['decoder'](self.rssm.imagine(data['action'][:6, 5:], start))
        for key in self.heads['decoder'].cnn_shapes.keys():
            truth = data[key][:6].to(torch.float32)
            model = torch.cat([recon[key].mode[:, :5], openl[key].mode], dim=1)
            error = (model - truth + 1) / 2
            video = torch.cat([truth, model, error], dim=2)
            report[f'openl_{key}'] = video_grid(video)
        return report

# =============================================================================
# Actor-Critic and Critic functions (PyTorch version)
# =============================================================================

class ImagActorCritic(Module):
    def __init__(self, critics, scales, act_space, config):
        super().__init__()
        # Filter critics based on nonzero scales.
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert (not scale) or (key in critics), key
        self.scales = scales
        self.act_space = act_space
        self.config = config
        self.actor = nets_torch.MLP(act_space.shape, **config.actor,
                              dist=(config.actor_dist_disc if act_space.discrete else config.actor_dist_cont))
        self.grad = config.actor_grad_disc if act_space.discrete else config.actor_grad_cont
        self.advnorm = Normalize(**config.advnorm)
        self.retnorms = {k: Normalize(**config.retnorm) for k in self.critics}
        self.scorenorms = {k: Normalize(**config.scorenorm) for k in self.critics}
        if config.actent_perdim:
            shape = act_space.shape[:-1] if act_space.discrete else act_space.shape
            self.actent = AutoAdapt(shape, **config.actent, inverse=True)
        else:
            self.actent = AutoAdapt((), **config.actent, inverse=True)
        self.opt = Optimizer('actor', **self.config.actor_opt)

    def initial(self, batch_size):
        return None

    def policy(self, state, carry=None):
        return {'action': self.actor(state)}, carry

    def train_actor(self, imagine, start, context):
        # Define a policy function that does not backpropagate.
        def policy_fn(s):
            with torch.no_grad():
                return self.actor(s).sample()
        traj = imagine(policy_fn, start, self.config.imag_horizon)
        metrics = self.update(traj)
        return traj, metrics

    def update(self, traj):
        metrics = {}
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            for k, v in mets.items():
                metrics[f'{key}_{k}'] = v
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
        score_sum = self.advnorm(torch.sum(torch.stack(scores, dim=0), dim=0))
        loss, mets = self.loss(traj, score_sum)
        metrics.update(mets)
        loss = loss.mean()
        self.opt.step(loss, [self.actor])
        return metrics

    def loss(self, traj, score):
        metrics = {}
        policy_out = self.actor({k: v.detach() for k, v in traj.items()})
        action = traj['action'].detach()
        if self.grad == 'backprop':
            loss = -score
        elif self.grad == 'reinforce':
            loss = -policy_out.log_prob(action)[:-1] * score.detach()
        else:
            raise NotImplementedError(self.grad)
        shape = self.act_space.shape[:-1] if self.act_space.discrete else self.act_space.shape
        if self.config.actent_perdim and len(shape) > 0:
            ent = policy_out.base_dist.entropy()[:-1]
            if self.config.actent_norm:
                lo = policy_out.minent / ent.shape[-1]
                hi = policy_out.maxent / ent.shape[-1]
                ent = (ent - lo) / (hi - lo)
            ent_loss, mets = self.actent(ent)
            ent_loss = ent_loss.sum(-1)
        else:
            ent = policy_out.entropy()[:-1]
            if self.config.actent_norm:
                lo, hi = policy_out.minent, policy_out.maxent
                ent = (ent - lo) / (hi - lo)
            ent_loss, mets = self.actent(ent)
        metrics.update({f'actent_{k}': v for k, v in mets.items()})
        loss = loss + ent_loss
        loss = loss * traj['weight'][:-1].detach()
        return loss, metrics

class VFunction(Module):
    def __init__(self, rewfn, config):
        super().__init__()
        assert 'action' not in config.critic.inputs, config.critic.inputs
        self.rewfn = rewfn
        self.config = config
        self.net = nets_torch.MLP((), **config.critic)
        if self.config.slow_target:
            self.target_net = nets_torch.MLP((), **config.critic)
            self.updates = -1
        else:
            self.target_net = self.net
        self.opt = Optimizer('critic', **self.config.critic_opt)

    def train(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target, _ = self.target(traj, reward, self.config.critic_return)
        dist = self.net({k: v[:-1] for k, v in traj.items()})
        loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
        self.opt.step(loss, [self.net])
        metrics.update({
            'critic_loss': loss.item(),
            'imag_reward_mean': reward.mean().item(),
            'imag_reward_std': reward.std().item(),
            'imag_critic_mean': dist.mean.mean().item(),
            'imag_critic_std': dist.mean.std().item(),
            'imag_return_mean': target.mean().item(),
            'imag_return_std': target.std().item(),
        })
        self.update_slow()
        return metrics

    def score(self, traj, actor):
        return self.target(traj, self.rewfn(traj), self.config.actor_return)

    def target(self, traj, reward, impl):
        if len(reward) != len(traj['action']) - 1:
            raise AssertionError('Should provide rewards for all but last action.')
        disc = traj['cont'][1:] * self.config.discount
        value = self.target_net(traj).mean
        if impl == 'gae':
            advs = [torch.zeros_like(value[0])]
            deltas = reward + disc * value[1:] - value[:-1]
            for t in reversed(range(len(disc))):
                advs.append(deltas[t] + disc[t] * self.config.return_lambda * advs[-1])
            adv = torch.stack(list(reversed(advs))[:-1])
            return adv + value[:-1], value[:-1]
        elif impl == 'gve':
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            ret = torch.stack(list(reversed(vals))[:-1])
            return ret, value[:-1]
        else:
            raise NotImplementedError(impl)

    def update_slow(self):
        if not self.config.slow_target:
            return
        initialize = self.updates == -1
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates = 0
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for s, d in zip(self.net.parameters(), self.target_net.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self.updates += 1

class QFunction(Module):
    def __init__(self, rewfn, config):
        super().__init__()
        assert config.actor_grad_disc == 'backprop'
        assert config.actor_grad_cont == 'backprop'
        assert 'action' in config.actor.inputs
        self.rewfn = rewfn
        self.config = config
        self.net = nets_torch.MLP((), **config.critic)
        if self.config.slow_target:
            self.target_net = nets_torch.MLP((), **config.critic)
            self.updates = -1
        else:
            self.target_net = self.net
        self.opt = Optimizer('critic', **self.config.critic_opt)

    def score(self, traj, actor):
        traj_detached = {k: v.detach() for k, v in traj.items()}
        inps = {**traj_detached, 'action': actor(traj_detached).sample()}
        ret = self.net({**traj, 'action': actor(traj).sample()}).mode[:-1]
        baseline = torch.zeros_like(ret)
        return ret, baseline

    def train_q(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target = self.target(traj, actor, reward).detach()
        inps = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(inps)
        loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
        self.opt.step(loss,[self.net])
        metrics.update({
            'imag_reward_mean': reward.mean().item(),
            'imag_reward_std': reward.std().item(),
            'imag_critic_mean': dist.mean().mean().item(),
            'imag_critic_std': dist.mean().std().item(),
            'imag_target_mean': target.mean().item(),
            'imag_target_std': target.std().item(),
        })
        self.update_slow()
        return metrics

    def target(self, traj, actor, reward):
        if len(reward) != len(traj['action']) - 1:
            raise AssertionError('Should provide rewards for all but last action.')
        cont = traj['cont'][1:]
        disc = cont * self.config.discount
        action_sample = actor(traj).sample()
        value = self.target_net({**traj, 'action': action_sample}).mean()
        if self.config.pengs_qlambda:
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            tar = torch.stack(list(reversed(vals))[:-1])
            return tar
        else:
            return reward + disc * value[1:]

    def update_slow(self):
        if not self.config.slow_target:
            return
        initialize = self.updates == -1
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates = 0
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for s, d in zip(self.net.parameters(), self.target_net.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self.updates += 1

class TwinQFunction(Module):
    def __init__(self, rewfn, config):
        super().__init__()
        assert config.actor_grad_disc == 'backprop'
        assert config.actor_grad_cont == 'backprop'
        assert 'action' in config.actor.inputs
        self.rewfn = rewfn
        self.config = config
        self.net1 = nets_torch.MLP((), **config.critic)
        self.net2 = nets_torch.MLP((), **config.critic)
        if self.config.slow_target:
            self.target_net1 = nets_torch.MLP((), **config.critic)
            self.target_net2 = nets_torch.MLP((), **config.critic)
            self.updates = -1
        else:
            self.target_net1 = self.net1
            self.target_net2 = self.net2
        self.opt = Optimizer('critic', **self.config.critic_opt)

    def score(self, traj, actor):
        traj_detached = {k: v.detach() for k, v in traj.items()}
        inps = {**traj_detached, 'action': actor(traj_detached).sample()}
        ret1 = self.net1(inps).mode
        ret2 = self.net2(inps).mode
        ret = torch.min(ret1, ret2)[:-1]
        baseline = torch.zeros_like(ret)
        return ret, baseline

    def train(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target = self.target(traj, actor, reward).detach()
        inps = {k: v[:-1] for k, v in traj.items()}
        self.opt.zero_grad()
        dist1 = self.net1(inps)
        dist2 = self.net2(inps)
        loss1 = -(dist1.log_prob(target) * traj['weight'][:-1]).mean()
        loss2 = -(dist2.log_prob(target) * traj['weight'][:-1]).mean()
        loss = loss1 + loss2
        modules = [self.net1, self.net2]
        self.opt.step(loss, modules)
        metrics.update({
            'imag_reward_mean': reward.mean().item(),
            'imag_reward_std': reward.std().item(),
            'imag_critic_mean': dist1.mean().mean().item(),
            'imag_critic_std': dist2.mean().std().item(),
            'imag_target_mean': target.mean().item(),
            'imag_target_std': target.std().item(),
        })
        self.update_slow()
        return metrics

    def target(self, traj, actor, reward):
        if len(reward) != len(traj['action']) - 1:
            raise AssertionError('Should provide rewards for all but last action.')
        cont = traj['cont'][1:]
        disc = cont * self.config.discount
        action_sample = actor(traj).sample()
        value1 = self.target_net1({**traj, 'action': action_sample}).mean()
        value2 = self.target_net2({**traj, 'action': actor(traj).sample()}).mean()
        value = torch.min(value1, value2)
        if self.config.pengs_qlambda:
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            tar = torch.stack(list(reversed(vals))[:-1])
            return tar
        else:
            return reward + disc * value[1:]

    def update_slow(self):
        if not self.config.slow_target:
            return
        initialize = self.updates == -1
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates = 0
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for s, d in zip(self.net1.parameters(), self.target_net1.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.net2.parameters(), self.target_net2.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self.updates += 1
