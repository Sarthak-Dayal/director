import numpy as np

from embodied.replay import FixedLength
from embodied.replay.hrl_goal_sampler import GoalSampler


class CombinedFixedLength(FixedLength):
    def __init__(self, store, goal_store, chunk, **kwargs):
        super().__init__(store, chunk, **kwargs)
        self.goal_buffer = GoalSampler(goal_store)

    def _sample(self):
        chunk = super()._sample()
        chunk['shitware'] = np.array([0.0])

