from typing import List
import pathlib
import sys

directory = pathlib.Path(__file__)
try:
    import google3  # noqa
except ImportError:
    directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied.envs.n_room_maps import *
import numpy as np
import cv2

# wall structure in all arrays is exactly the same
registered_layouts = {
    "empty_room": [EMPTY_ROOM],
    "unreachable": [UNREACHABLE_TEST],
    "4_rooms": [FOUR_ROOMS_A, FOUR_ROOMS_B, FOUR_ROOMS_C, FOUR_ROOMS_D],
    "test_room": [TEST_ROOM],
    "4_rooms_allways": [FOUR_ROOMS_ALLWAYS],
    "4_rooms_1_hallway": [FOUR_ROOMS_1_HALLWAY_A, FOUR_ROOMS_1_HALLWAY_B, FOUR_ROOMS_1_HALLWAY_C]
}

colors = {
    "#": (0, 0, 0), # wall, black
    ".": (255, 255, 255), # floor, white
    "A": (255, 0, 0), # agent, red
    "G": (0, 255, 0) # goal, green
}

class NRooms(embodied.Env):
    def __init__(self, layout_str, time_limit=100) -> None:
        self._layout = registered_layouts[layout_str]
        self._possible_maps = None
        self._possible_spawn_locations = None
        self._time_limit = time_limit
        self._collision_map = None

        self._t = 0
        self._pos = None
        self._goal = None
        self._done = False
        self._prev_pos = None

        self.parse_maps(self._layout)
        self.reset()

    def parse_maps(self, raw_maps: List[str]) -> None:
        """
        Parse goal location, agent spawn locations possible (uniform random)
        """
        self._possible_maps = np.array([None] * len(raw_maps))
        for i, map_str in enumerate(raw_maps):
            map_np = np.array([list(line) for line in map_str.split('\n')])
            self._possible_maps[i] = map_np
        self._possible_spawn_locations = [np.argwhere(map_np == '.') for map_np in self._possible_maps]

    def reset(self):
        """
        Reset Environment
        """
        self._t = 0
        self._done = False

        # Pick a random map
        map_idx = np.random.choice(len(self._possible_spawn_locations))
        spawn_locations = self._possible_spawn_locations[map_idx]
        self._collision_map = self._possible_maps[map_idx]
        
        # Pick two distinct locations
        chosen = np.random.choice(len(spawn_locations), 2, replace=False)
        self._pos = spawn_locations[chosen[0]]
        self._goal = spawn_locations[chosen[1]]
        self._prev_pos = self._pos.copy()

    @property
    def obs_space(self):
        return {
            'image':      embodied.Space(np.uint8, (64, 64, 3,)),
            'reward':     embodied.Space(np.float32),
            'is_first':   embodied.Space(bool),
            'is_last':    embodied.Space(bool),
            'is_terminal':embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.int32, (), 0, 4),
            'reset':  embodied.Space(bool),
        }

    def step(self, action):

        if action.get('reset', False) or self._done:
            self.reset()
            return self._make_obs(is_first=True)

        self._t += 1

        a = int(action['action'])
        self._take_action(a)

        if np.all(self._pos == self._goal):
            self._done = True
            is_last = True
            is_terminal = True
        elif self._t >= self._time_limit:
            self._done = True
            is_last = True
            is_terminal = False
        else:
            is_last = False
            is_terminal = False

        return self._make_obs(is_last=is_last, is_terminal=is_terminal)

    def _take_action(self, action: int):
        new_pos = self._pos.copy()
        if action == 0:
            new_pos[0] -= 1
        elif action == 1:
            new_pos[0] += 1
        elif action == 2:
            new_pos[1] -= 1
        elif action == 3:
            new_pos[1] += 1

        # check if new position is valid
        if self._collision_map[new_pos[0], new_pos[1]] != '#':
            self._prev_pos = self._pos.copy()
            self._pos = new_pos

    def _make_obs(self, is_first=False, is_last=False, is_terminal=False):
        # Create an RGB image and assign wall and floor colors
        h, w = self._collision_map.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        wall_mask = self._collision_map == "#"
        floor_mask = (self._collision_map == ".") | (self._collision_map == '-')
        img[wall_mask] = colors["#"]
        img[floor_mask] = colors["."]
        # Draw agent and goal
        img[self._pos[0], self._pos[1]] = colors["A"]
        img[self._goal[0], self._goal[1]] = colors["G"]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)

        return {
            'image':       img,
            'reward':      self.get_reward(),
            'is_first':    is_first,
            'is_last':     is_last,
            'is_terminal': is_terminal,
        }

    def get_reward(self):
        # negative delta L1 distance to goal
        prev_dist = np.sum(np.abs(self._prev_pos - self._goal))
        curr_dist = np.sum(np.abs(self._pos - self._goal))
        return prev_dist - curr_dist

    def render(self):
        return self._make_obs()['image']

    @staticmethod
    def render_pure(pos, goal, layout_str):

        collision_map = registered_layouts[layout_str][0]
        map_np = np.array([list(line) for line in collision_map.split('\n')])

        h, w = len(map_np), len(map_np[0])
        img = np.zeros((h, w, 3), dtype=np.uint8)

        wall_mask = map_np == "#"
        floor_mask = (map_np == ".") | (map_np == '-')

        img[wall_mask] = colors["#"]
        img[floor_mask] = colors["."]
        
        img[pos[0], pos[1]] = colors["A"]
        img[goal[0], goal[1]] = colors["G"]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)

        return img