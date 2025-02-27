# behaviors.py
import torch
import torch.distributions as td

from . import agent_torch
from . import expl
from . import tfutils

from .hierarchy import Hierarchy  # if needed

class Greedy(tfutils.Module):
    def __init__(self, wm, act_space, config):
        # Define a reward function that uses the world model's reward head.
        # Note: [1:] is used to ignore the first timestep.
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        if config.critic_type == 'vfunction':
            critics = {'extr': agent_torch.VFunction(rewfn, config)}
        elif config.critic_type == 'qfunction':
            critics = {'extr': agent_torch.QFunction(rewfn, config)}
        else:
            raise NotImplementedError(f"Unknown critic_type: {config.critic_type}")
        self.ac = agent_torch.ImagActorCritic(critics, {'extr': 1.0}, act_space, config)

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}

class Random(tfutils.Module):
    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return torch.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist_obj = tfutils.OneHotDist(torch.zeros(shape))
        else:
            dist_obj = td.Uniform(-torch.ones(shape), torch.ones(shape))
            dist_obj = td.Independent(dist_obj, 1)
        return {'action': dist_obj}, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}

class Explore(tfutils.Module):
    # Map keys to corresponding exploration reward classes from expl.
    REWARDS = {
        'disag': expl.Disag,
        'vae': expl.LatentVAE,
        'ctrl': expl.CtrlDisag,
        'pbe': expl.PBE,
    }

    def __init__(self, wm, act_space, config):
        self.config = config
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            if key == 'extr':
                # For extrinsic reward, use the reward head from the world model.
                reward = lambda traj: wm.heads['reward'](traj).mean()[1:]
                critics[key] = agent_torch.VFunction(reward, config)
            else:
                reward = self.REWARDS[key](wm, act_space, config)
                # Use config.update to change discount and retnorm if needed.
                critics[key] = agent_torch.VFunction(
                    reward, config.update(discount=config.expl_discount, retnorm=config.expl_retnorm)
                )
                self.rewards[key] = reward
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = agent_torch.ImagActorCritic(critics, scales, act_space, config)

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        metrics = {}
        for key, reward in self.rewards.items():
            # We assume that each reward module has a train method returning a dict.
            metrics.update(reward.train(data))
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}
