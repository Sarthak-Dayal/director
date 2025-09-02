import time

import numpy as np
import embodied


class GoalSampler(embodied.Replay):
    def __init__(
            self, store):
        self.store = store
        self.random = np.random.RandomState(seed=42)

    def __len__(self):
        return self.store.steps

    @property
    def stats(self):
        return {f'replay_{k}': v for k, v in self.store.stats().items()}

    def dataset(self):
        while True:
            traj = self._sample()
            if traj is None:
                print('Waiting for episodes.')
                time.sleep(1)
                continue
            yield traj

    def _sample(self):
        keys = self.store.keys()
        if not keys:
            return None
        traj = self.store[keys[self.random.randint(0, len(keys))]]
        upper = len(next(iter(traj.values())))
        lower = 0
        index = self.random.randint(lower, upper)
        goal = {k: traj[k][index] for k in traj.keys()}
        return goal