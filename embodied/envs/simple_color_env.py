import embodied
import numpy as np

class SimpleColorEnv(embodied.Env):
    def __init__(self, size=(64, 64), seed=None, max_steps=244):
        self._size = size
        self._rng = np.random.RandomState(seed)
        self._state = None
        self._done = True
        self._t = 0
        self._max_steps = max_steps

    @property
    def obs_space(self):
        return {
            'image':      embodied.Space(np.uint8, self._size + (3,)),
            'reward':     embodied.Space(np.float32),
            'is_first':   embodied.Space(bool),
            'is_last':    embodied.Space(bool),
            'is_terminal':embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.int32, (), 0, 3),
            'reset':  embodied.Space(bool),
        }

    def step(self, action):

        if action.get('reset', False) or self._done:
            self._done = False
            self._t = 0
            self._state = 'red'
            return self._make_obs(is_first=True)

        self._t += 1

        a = int(action['action'])
        if   a == 0: self._state = 'green'
        elif a == 1: self._state = 'blue'
        elif a == 2: self._state = 'red'

        if self._t >= self._max_steps:
            self._done = True
            is_last = True
            is_terminal = True
        else:
            is_last = False
            is_terminal = False

        return self._make_obs(is_last=is_last, is_terminal=is_terminal)

    def _make_obs(self, is_first=False, is_last=False, is_terminal=False):
        img = np.zeros(self._size + (3,), np.uint8)
        if   self._state == 'red':   img[..., 0] = 255
        elif self._state == 'green': img[..., 1] = 255
        elif self._state == 'blue':  img[..., 2] = 255

        return {
            'image':       img,
            'reward':      np.float32(1.0),  # always 1
            'is_first':    is_first,
            'is_last':     is_last,
            'is_terminal': is_terminal,
        }

    def render(self):
        return self._make_obs()['image']
