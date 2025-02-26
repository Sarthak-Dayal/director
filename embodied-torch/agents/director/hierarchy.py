# hierarchy_torch.py
import functools
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td

import embodied  # Assuming embodied.Space exists in PyTorch
from . import agent_torch
from . import expl
from . import nets_torch
from . import tfutils
from tfutils import map_structure, tensor, scan  # our helper functions


# ----------------------------------------------------------------------------
# Hierarchy module in PyTorch
# ----------------------------------------------------------------------------
class Hierarchy(tfutils.Module):
    def __init__(self, wm, act_space, config):
        super().__init__()
        self.wm = wm
        self.config = config
        # extr_reward ignores the first timestep.
        self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]
        # Create a skill space. (Assuming embodied.Space works similarly in PyTorch.)
        dtype = np.int32 if config.goal_encoder.dist == 'onehot' else np.float32
        self.skill_space = embodied.Space(dtype, config.skill_shape)

        # Update the configuration for the worker.
        wconfig = config.update({
            'actor.inputs': config.worker_inputs,
            'critic.inputs': config.worker_inputs,
        })
        # Instantiate the worker actor–critic.
        self.worker = agent_torch.ImagActorCritic({
            'extr': agent_torch.VFunction(lambda s: s['reward_extr'], config)
        }, config.worker_rews, act_space, wconfig)

        # Manager configuration: update some keys.
        mconfig = config.update({
            'actor_grad_cont': 'reinforce',
            'actent.target': config.manager_actent,
        })
        self.manager = agent_torch.ImagActorCritic({
            'extr': agent_torch.VFunction(lambda s: s['reward_extr'], mconfig),
            'expl': agent_torch.VFunction(lambda s: s['reward_expl'], mconfig),
            'goal': agent_torch.VFunction(lambda s: s['reward_goal'], mconfig),
        }, config.manager_rews, self.skill_space, mconfig)

        # Set up exploration reward.
        if config.expl_rew == 'disag':
            self.expl_reward = expl.Disag(wm, act_space, config)
        elif config.expl_rew == 'adver':
            self.expl_reward = self.elbo_reward
        else:
            raise NotImplementedError(config.expl_rew)

        if config.explorer:
            self.explorer = agent_torch.ImagActorCritic({
                'expl': agent_torch.VFunction(self.expl_reward, config)
            }, {'expl': 1.0}, act_space, config)

        # Prior over skills.
        shape = self.skill_space.shape
        if self.skill_space.discrete:
            prior_logits = torch.zeros(shape)
            # OneHotDist is defined in tfutils.
            prior = tfutils.OneHotDist(prior_logits)
            # Wrap with Independent: note that if shape has more than one dim, use len(shape)-1 event dims.
            self.prior = td.Independent(prior, len(shape) - 1)
        else:
            prior_mean = torch.zeros(shape)
            prior_std = torch.ones(shape)
            self.prior = td.Independent(td.Normal(prior_mean, prior_std), len(shape))

        # Additional network components.
        self.feat = nets_torch.Input(['deter'])
        self.goal_shape = (config.rssm.deter,)
        self.enc = nets_torch.MLP(config.skill_shape, dims='context', **config.goal_encoder)
        self.dec = nets_torch.MLP(self.goal_shape, dims='context', **config.goal_decoder)
        self.kl = tfutils.AutoAdapt((), **config.encdec_kl)
        self.opt = tfutils.Optimizer('goal', **config.encdec_opt)

    def initial(self, batch_size):
        return {
            'step': torch.zeros(batch_size, dtype=torch.int64),
            'skill': torch.zeros((batch_size,) + self.config.skill_shape, dtype=torch.float32),
            'goal': torch.zeros((batch_size,) + self.goal_shape, dtype=torch.float32),
        }

    def policy(self, latent, carry, imag=False):
        # Determine duration based on whether we are imagining rollout.
        duration = self.config.train_skill_duration if imag else self.config.env_skill_duration
        # Use map_structure to detach.
        sg = lambda x: map_structure(lambda y: y.detach(), x)
        update = (carry['step'] % duration == 0)  # Boolean tensor of shape [batch]
        # switch(x, y): for each sample, select x if update==False, y if update==True.
        switch = lambda x, y: (
                torch.einsum('i,i...->i...', (1 - update.float()), x) +
                torch.einsum('i,i...->i...', update.float(), y)
        )
        skill = sg(switch(carry['skill'], self.manager.actor(sg(latent)).sample()))
        new_goal = self.dec({'skill': skill, 'context': self.feat(latent)}).mode()
        new_goal = (self.feat(latent).to(torch.float32) + new_goal) if self.config.manager_delta else new_goal
        goal = sg(switch(carry['goal'], new_goal))
        delta = goal - self.feat(latent).to(torch.float32)
        # Build input for the worker.
        worker_input = sg({**latent, 'goal': goal, 'delta': delta})
        dist_obj = self.worker.actor(worker_input)
        outs = {'action': dist_obj}
        if 'image' in self.wm.heads['decoder'].shapes:
            dec_out = self.wm.heads['decoder']({
                'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)
            })
            outs['log_goal'] = dec_out['image'].mode()
        new_step = carry['step'] + 1
        new_carry = {'step': new_step, 'skill': skill, 'goal': goal}
        return outs, new_carry

    def train(self, imagine, start, data):
        success = lambda rew: (rew[-1] > 0.7).float().mean()
        metrics = {}
        if self.config.expl_rew == 'disag':
            metrics.update(self.expl_reward.train(data))
        if self.config.vae_replay:
            metrics.update(self.train_vae_replay(data))
        if self.config.explorer:
            traj, mets = self.explorer.train(imagine, start, data)
            metrics.update({f'explorer_{k}': v for k, v in mets.items()})
            metrics.update(self.train_vae_imag(traj))
            if self.config.explorer_repeat:
                goal = self.feat(traj)[-1]
                metrics.update(self.train_worker(imagine, start, goal)[1])
        if self.config.jointly == 'new':
            traj, mets = self.train_jointly(imagine, start)
            metrics.update(mets)
            metrics['success_manager'] = success(traj['reward_goal'])
            if self.config.vae_imag:
                metrics.update(self.train_vae_imag(traj))
        elif self.config.jointly == 'old':
            traj, mets = self.train_jointly_old(imagine, start)
            metrics.update(mets)
            metrics['success_manager'] = success(traj['reward_goal'])
            if self.config.vae_imag:
                metrics.update(self.train_vae_imag(traj))
        elif self.config.jointly == 'off':
            for impl in self.config.worker_goals:
                goal = self.propose_goal(start, impl)
                traj, mets = self.train_worker(imagine, start, goal)
                metrics.update(mets)
                metrics[f'success_{impl}'] = success(traj['reward_goal'])
                if self.config.vae_imag:
                    metrics.update(self.train_vae_imag(traj))
            traj, mets = self.train_manager(imagine, start)
            metrics.update(mets)
            metrics['success_manager'] = success(traj['reward_goal'])
        else:
            raise NotImplementedError(self.config.jointly)
        return None, metrics

    def train_jointly(self, imagine, start):
        start = start.copy()
        metrics = {}
        with torch.enable_grad():
            policy = functools.partial(self.policy, imag=True)
            traj = self.wm.imagine_carry(
                policy, start, self.config.imag_horizon,
                self.initial(len(start['is_first']))
            )
            traj['reward_extr'] = self.extr_reward(traj)
            traj['reward_expl'] = self.expl_reward(traj)
            traj['reward_goal'] = self.goal_reward(traj)
            traj['delta'] = traj['goal'] - self.feat(traj).to(torch.float32)
            wtraj = self.split_traj(traj)
            mtraj = self.abstract_traj(traj)
        worker_mets = self.worker.update(wtraj, tape=None)
        metrics.update({f'worker_{k}': v for k, v in worker_mets.items()})
        manager_mets = self.manager.update(mtraj, tape=None)
        metrics.update({f'manager_{k}': v for k, v in manager_mets.items()})
        return traj, metrics

    def train_jointly_old(self, imagine, start):
        start = start.copy()
        metrics = {}
        sg = lambda x: map_structure(lambda y: y.detach(), x)
        context = self.feat(start)
        with torch.enable_grad():
            skill = self.manager.actor(sg(start)).sample()
            goal = self.dec({'skill': skill, 'context': context}).mode()
            goal = (self.feat(start).to(torch.float32) + goal) if self.config.manager_delta else goal
            worker = lambda s: self.worker.actor(
                sg({**s, 'goal': goal, 'delta': goal - self.feat(s).to(torch.float32)})).sample()
            traj = imagine(worker, start, self.config.imag_horizon)
            traj['goal'] = goal.repeat(1 + self.config.imag_horizon, 1)
            traj['skill'] = skill.repeat(1 + self.config.imag_horizon, 1)
            traj['reward_extr'] = self.extr_reward(traj)
            traj['reward_expl'] = self.expl_reward(traj)
            traj['reward_goal'] = self.goal_reward(traj)
            traj['delta'] = traj['goal'] - self.feat(traj).to(torch.float32)
            wtraj = traj.copy()
            mtraj = self.abstract_traj_old(traj)
        worker_mets = self.worker.update(wtraj, tape=None)
        metrics.update({f'worker_{k}': v for k, v in worker_mets.items()})
        manager_mets = self.manager.update(mtraj, tape=None)
        metrics.update({f'manager_{k}': v for k, v in manager_mets.items()})
        return traj, metrics

    def train_manager(self, imagine, start):
        start = start.copy()
        with torch.enable_grad():
            policy = functools.partial(self.policy, imag=True)
            traj = self.wm.imagine_carry(
                policy, start, self.config.imag_horizon,
                self.initial(len(start['is_first']))
            )
            traj['reward_extr'] = self.extr_reward(traj)
            traj['reward_expl'] = self.expl_reward(traj)
            traj['reward_goal'] = self.goal_reward(traj)
            traj['delta'] = traj['goal'] - self.feat(traj).to(torch.float32)
            mtraj = self.abstract_traj(traj)
        metrics = self.manager.update(mtraj, tape=None)
        metrics = {f'manager_{k}': v for k, v in metrics.items()}
        return traj, metrics

    def train_worker(self, imagine, start, goal):
        start = start.copy()
        metrics = {}
        sg = lambda x: map_structure(lambda y: y.detach(), x)
        with torch.enable_grad():
            worker = lambda s: self.worker.actor(
                sg({**s, 'goal': goal, 'delta': goal - self.feat(s).to(torch.float32)})).sample()
            traj = imagine(worker, start, self.config.imag_horizon)
            traj['goal'] = goal.repeat(1 + self.config.imag_horizon, 1)
            traj['reward_extr'] = self.extr_reward(traj)
            traj['reward_expl'] = self.expl_reward(traj)
            traj['reward_goal'] = self.goal_reward(traj)
            traj['delta'] = traj['goal'] - self.feat(traj).to(torch.float32)
        mets = self.worker.update(traj, tape=None)
        metrics.update({f'worker_{k}': v for k, v in mets.items()})
        return traj, metrics

    def train_vae_replay(self, data):
        metrics = {}
        feat = self.feat(data).to(torch.float32)
        if 'context' in self.config.goal_decoder.inputs:
            if self.config.vae_span:
                context = feat[:, 0]
                goal = feat[:, -1]
            else:
                assert feat.shape[1] > self.config.train_skill_duration
                context = feat[:, :-self.config.train_skill_duration]
                goal = feat[:, self.config.train_skill_duration:]
        else:
            goal = context = feat
        # Run encoding and decoding
        enc = self.enc({'goal': goal, 'context': context})
        dec = self.dec({'skill': enc.sample(), 'context': context})
        rec = -dec.log_prob(goal.detach())
        if self.config.goal_kl:
            kl = td.kl_divergence(enc, self.prior)
            kl, mets = self.kl(kl)
            metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
        else:
            kl = 0.0
        loss = (rec + kl).mean()
        metrics.update(self.opt.step(loss, [self.enc, self.dec]))
        metrics['goalrec_mean'] = rec.mean().item()
        metrics['goalrec_std'] = rec.std().item()
        return metrics

    def train_vae_imag(self, traj):
        metrics = {}
        feat = self.feat(traj).to(torch.float32)
        if 'context' in self.config.goal_decoder.inputs:
            if self.config.vae_span:
                context = feat[0]
                goal = feat[-1]
            else:
                assert feat.shape[0] > self.config.train_skill_duration
                context = feat[:-self.config.train_skill_duration]
                goal = feat[self.config.train_skill_duration:]
        else:
            goal = context = feat
        with torch.enable_grad():
            enc = self.enc({'goal': goal, 'context': context})
            dec = self.dec({'skill': enc.sample(), 'context': context})
            rec = -dec.log_prob(goal.detach())
            if self.config.goal_kl:
                kl = td.kl_divergence(enc, self.prior)
                kl, mets = self.kl(kl)
                metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
            else:
                kl = 0.0
            loss = (rec + kl).mean()
        metrics.update(self.opt.step(loss, [self.enc, self.dec]))
        metrics['goalrec_mean'] = rec.mean().item()
        metrics['goalrec_std'] = rec.std().item()
        return metrics

    def propose_goal(self, start, impl):
        feat = self.feat(start).to(torch.float32)
        if impl == 'replay':
            target = feat[torch.randperm(feat.shape[0])]
            skill = self.enc({'goal': target, 'context': feat}).sample()
            return self.dec({'skill': skill, 'context': feat}).mode()
        elif impl == 'replay_direct':
            return feat[torch.randperm(feat.shape[0])].to(torch.float32)
        elif impl == 'manager':
            skill = self.manager.actor(start).sample()
            goal = self.dec({'skill': skill, 'context': feat}).mode()
            goal = feat + goal if self.config.manager_delta else goal
            return goal
        elif impl == 'prior':
            skill = self.prior.sample(len(start['is_terminal']))
            return self.dec({'skill': skill, 'context': feat}).mode()
        else:
            raise NotImplementedError(impl)

    def goal_reward(self, traj):
        feat = self.feat(traj).to(torch.float32)
        goal = traj['goal'].to(torch.float32).detach()
        skill = traj['skill'].to(torch.float32).detach()
        context = feat[0].unsqueeze(0).repeat(1 + self.config.imag_horizon, 1)
        if self.config.goal_reward == 'dot':
            return torch.einsum('...i,...i->...', goal, feat)[1:]
        elif self.config.goal_reward == 'dir':
            return torch.einsum('...i,...i->...', F.normalize(goal, dim=-1), feat)[1:]
        elif self.config.goal_reward == 'normed_inner':
            norm = feat.norm(dim=-1, keepdim=True)
            return torch.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
        elif self.config.goal_reward == 'normed_squared':
            norm = feat.norm(dim=-1, keepdim=True)
            return -((goal / norm - feat / norm) ** 2).mean(dim=-1)[1:]
        elif self.config.goal_reward == 'cosine_lower':
            gnorm = goal.norm(dim=-1, keepdim=True) + 1e-12
            fnorm = feat.norm(dim=-1, keepdim=True) + 1e-12
            fnorm = torch.max(gnorm, fnorm)
            return torch.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
        elif self.config.goal_reward == 'cosine_lower_pos':
            gnorm = goal.norm(dim=-1, keepdim=True) + 1e-12
            fnorm = feat.norm(dim=-1, keepdim=True) + 1e-12
            fnorm = torch.max(gnorm, fnorm)
            cos = torch.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
            return F.relu(cos)
        elif self.config.goal_reward == 'cosine_frac':
            gnorm = goal.norm(dim=-1) + 1e-12
            fnorm = feat.norm(dim=-1) + 1e-12
            goal = goal / gnorm.unsqueeze(-1)
            feat = feat / fnorm.unsqueeze(-1)
            cos = torch.einsum('...i,...i->...', goal, feat)
            mag = torch.min(gnorm, fnorm) / torch.max(gnorm, fnorm)
            return (cos * mag)[1:]
        elif self.config.goal_reward == 'cosine_frac_pos':
            gnorm = goal.norm(dim=-1) + 1e-12
            fnorm = feat.norm(dim=-1) + 1e-12
            goal = goal / gnorm.unsqueeze(-1)
            feat = feat / fnorm.unsqueeze(-1)
            cos = torch.einsum('...i,...i->...', goal, feat)
            mag = torch.min(gnorm, fnorm) / torch.max(gnorm, fnorm)
            return F.relu(cos * mag)[1:]
        elif self.config.goal_reward == 'cosine_max':
            gnorm = goal.norm(dim=-1, keepdim=True) + 1e-12
            fnorm = feat.norm(dim=-1, keepdim=True) + 1e-12
            norm_val = torch.max(gnorm, fnorm)
            return torch.einsum('...i,...i->...', goal / norm_val, feat / norm_val)[1:]
        elif self.config.goal_reward == 'cosine_max_pos':
            gnorm = goal.norm(dim=-1, keepdim=True) + 1e-12
            fnorm = feat.norm(dim=-1, keepdim=True) + 1e-12
            norm_val = torch.max(gnorm, fnorm)
            cos = torch.einsum('...i,...i->...', goal / norm_val, feat / norm_val)[1:]
            return F.relu(cos)
        elif self.config.goal_reward == 'normed_inner_clip':
            norm = goal.norm(dim=-1, keepdim=True)
            cosine = torch.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
            return cosine.clamp(-1.0, 1.0)
        elif self.config.goal_reward == 'normed_inner_clip_pos':
            norm = goal.norm(dim=-1, keepdim=True)
            cosine = torch.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
            return cosine.clamp(0.0, 1.0)
        elif self.config.goal_reward == 'diff':
            goal_norm = F.normalize(goal[:-1], dim=-1)
            diff = feat[1:] - feat[:-1]
            return torch.einsum('...i,...i->...', goal_norm, diff)
        elif self.config.goal_reward == 'norm':
            return - (goal - feat).norm(dim=-1)[1:]
        elif self.config.goal_reward == 'squared':
            return - ((goal - feat) ** 2).sum(dim=-1)[1:]
        elif self.config.goal_reward == 'epsilon':
            return ((goal - feat).mean(dim=-1) < 1e-3).float()[1:]
        elif self.config.goal_reward == 'enclogprob':
            return self.enc({'goal': goal, 'context': context}).log_prob(skill)[1:]
        elif self.config.goal_reward == 'encprob':
            return self.enc({'goal': goal, 'context': context}).prob(skill)[1:]
        elif self.config.goal_reward == 'enc_normed_cos':
            dist_obj = self.enc({'goal': goal, 'context': context})
            probs = dist_obj.distribution.probs_parameter()
            norm_val = probs.norm(dim=[-2, -1], keepdim=True)
            return torch.einsum('...ij,...ij->...', probs / norm_val, skill / norm_val)[1:]
        elif self.config.goal_reward == 'enc_normed_squared':
            dist_obj = self.enc({'goal': goal, 'context': context})
            probs = dist_obj.distribution.probs_parameter()
            norm_val = probs.norm(dim=[-2, -1], keepdim=True)
            return - ((probs / norm_val - skill / norm_val) ** 2).mean(dim=[-2, -1])[1:]
        else:
            raise NotImplementedError(self.config.goal_reward)

    def elbo_reward(self, traj):
        feat = self.feat(traj).to(torch.float32)
        context = feat[0].unsqueeze(0).repeat(1 + self.config.imag_horizon, 1)
        enc = self.enc({'goal': feat, 'context': context})
        dec = self.dec({'skill': enc.sample(), 'context': context})
        ll = dec.log_prob(feat)
        kl = td.kl_divergence(enc, self.prior)
        if self.config.adver_impl == 'abs':
            return (dec.mode() - feat).abs().mean(dim=-1)[1:]
        elif self.config.adver_impl == 'squared':
            return ((dec.mode() - feat) ** 2).mean(dim=-1)[1:]
        elif self.config.adver_impl == 'elbo_scaled':
            return (kl - ll / self.kl.scale())[1:]
        elif self.config.adver_impl == 'elbo_unscaled':
            return (kl - ll)[1:]
        else:
            raise NotImplementedError(self.config.adver_impl)

    def split_traj(self, traj):
        traj = traj.copy()
        k = self.config.train_skill_duration
        # Ensure length is compatible.
        assert (traj['action'].shape[0] - 1) % k == 0, f"Length mismatch: {traj['action'].shape[0]}, k={k}"
        reshape_fn = lambda x: x[:-1].view(x.shape[0] // k, k, *x.shape[1:])
        for key, val in list(traj.items()):
            if 'reward' in key:
                val = torch.cat([torch.zeros_like(val[:1]), val], dim=0)
            val = torch.cat([reshape_fn(val), val[k::k].unsqueeze(1)], dim=1)
            # Transpose so that the new time axis is first.
            dims = list(range(val.dim()))
            dims[0], dims[1] = dims[1], dims[0]
            val = val.permute(dims)
            # Merge first two dims.
            new_shape = (val.shape[0], -1) + val.shape[2:]
            val = val.reshape(new_shape)
            if 'reward' in key:
                val = val[1:]
            traj[key] = val
        traj['goal'] = torch.cat([traj['goal'][:-1], traj['goal'][:1]], dim=0)
        traj['weight'] = torch.cumprod(self.config.discount * traj['cont'], dim=0) / self.config.discount
        return traj

    def abstract_traj(self, traj):
        traj = traj.copy()
        traj['action'] = traj.pop('skill')
        k = self.config.train_skill_duration
        reshaped = lambda x: x[:-1].view(x.shape[0] // k, k, *x.shape[1:])
        weights = torch.cumprod(reshaped(traj['cont'][:-1]), dim=1)
        for key, value in list(traj.items()):
            if 'reward' in key:
                traj[key] = reshaped(value) * weights
                traj[key] = traj[key].mean(dim=1)
            elif key == 'cont':
                traj[key] = torch.cat([value[:1], reshaped(value[1:]).prod(dim=1)], dim=0)
            else:
                traj[key] = torch.cat([reshaped(value)[:, 0], value[-1:]], dim=0)
        traj['weight'] = torch.cumprod(self.config.discount * traj['cont'], dim=0) / self.config.discount
        return traj

    def abstract_traj_old(self, traj):
        traj = traj.copy()
        traj['action'] = traj.pop('skill')
        mult = torch.cumprod(traj['cont'][1:], dim=0)
        for key, value in list(traj.items()):
            if 'reward' in key:
                traj[key] = (mult * value).mean(dim=0).unsqueeze(0)
            elif key == 'cont':
                traj[key] = torch.stack([value[0], value[1:].prod(dim=0)], dim=0)
            else:
                traj[key] = torch.stack([value[0], value[-1]], dim=0)
        return traj

    def report(self, data):
        report = {}
        for impl in ('manager', 'prior', 'replay'):
            for key, video in self.report_worker(data, impl).items():
                report[f'impl_{impl}_{key}'] = video
        return report

    def report_worker(self, data, impl):
        # Prepare initial state.
        decoder = self.wm.heads['decoder']
        states, _ = self.wm.rssm.observe(self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
        start = {k: v[:, 4] for k, v in states.items()}
        start['is_terminal'] = data['is_terminal'][:6, 4]
        goal = self.propose_goal(start, impl)
        worker = lambda s: self.worker.actor(map_structure(lambda x: x.detach(), {**s, 'goal': goal,
                                                                                  'delta': goal - self.feat(s).to(
                                                                                      torch.float32)})).sample()
        traj = self.wm.imagine(worker, start, self.config.worker_report_horizon)
        initial = decoder(start)
        target = decoder({'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})
        rollout = decoder(traj)
        videos = {}
        for k in rollout.keys():
            if k not in decoder.cnn_shapes:
                continue
            length = 1 + self.config.worker_report_horizon
            rows = []
            rows.append(initial[k].mode().unsqueeze(1).repeat(1, length, 1, 1, 1))
            if target is not None:
                rows.append(target[k].mode().unsqueeze(1).repeat(1, length, 1, 1, 1))
            # For rollout, transpose to have time first.
            r = rollout[k].mode().permute(1, 0, 2, 3, 4)
            rows.append(r)
            videos[k] = tfutils.video_grid(torch.cat(rows, dim=2))
        return videos

