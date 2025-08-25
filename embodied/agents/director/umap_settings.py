from typing import List
from dataclasses import dataclass
from typing import Tuple
import random

@dataclass
class UMAPSettings:
    layout_str: str
    structured_data: List
    category_labels: List[str]
    color_setting: str = "random"

four_room_boxes = [
    (1, 10, 1, 10),
    (1, 10, 12, 21),
    (12, 21, 1, 10),
    (12, 21, 12, 21)

]

FIXED_GOAL_CHANGING_POS = UMAPSettings(
    layout_str="4_rooms",
    structured_data=[
        # Center Points
        [
            [(5, 5), (10, 10)],
            [(5, 16), (10, 21)],
            [(16, 5), (21, 10)],
            [(16, 16), (21, 21)],
        ],
        # Top Points
        [
            [(1, 5), (10, 10)],
            [(1, 16), (10, 21)],
            [(12, 5), (21, 10)],
            [(12, 16), (21, 21)]
        ],
        # Left Points
        [
            [(5, 1), (10, 10)],
            [(5, 12), (10, 21)], 
            [(16, 1), (21, 10)],
            [(16, 12), (21, 21)],
        ],
        # Bottom Points
        [
            [(10, 5), (10, 10)],
            [(10, 16), (10, 21)],
            [(21, 5), (21, 10)],
            [(21, 16), (21, 21)]
        ],
        # Right Points
        [
            [(5, 10), (10, 10)],
            [(5, 21), (10, 21)],
            [(16, 10), (21, 10)],
            [(16, 21), (21, 21)],
        ],
    ],
    category_labels = ["Center Points", "Top Points", "Left Points", "Bottom Points", "Right Points"]
)

# Takes in existing point, xy box, generates a random goal point within the box (inclusive) that isn't the point
def generate_random_goal(point, box: Tuple[int, int, int, int]):
    x_min, x_max, y_min, y_max = box
    while True:
        goal = (random.randint(x_min, x_max), random.randint(y_min, y_max))
        if goal != point:
            return goal

RANDOM_GOAL_FIXED_POS = UMAPSettings(
    layout_str="4_rooms",
    structured_data=[
        # Center Points
        [
            [(5, 5), generate_random_goal((5, 5), four_room_boxes[0])],
            [(5, 16), generate_random_goal((5, 16), four_room_boxes[1])],
            [(16, 5), generate_random_goal((16, 5), four_room_boxes[2])],
            [(16, 16), generate_random_goal((16, 16), four_room_boxes[3])],
        ],
        # Top Points
        [
            [(1, 5), generate_random_goal((1, 5), four_room_boxes[0])],
            [(1, 16), generate_random_goal((1, 16), four_room_boxes[1])],
            [(12, 5), generate_random_goal((12, 5), four_room_boxes[2])],
            [(12, 16), generate_random_goal((12, 16), four_room_boxes[3])]
        ],
        # Left Points
        [
            [(5, 1), generate_random_goal((5, 1), four_room_boxes[0])],
            [(5, 12), generate_random_goal((5, 12), four_room_boxes[1])],
            [(16, 1), generate_random_goal((16, 1), four_room_boxes[2])],
            [(16, 12), generate_random_goal((16, 12), four_room_boxes[3])],
        ],
        # Bottom Points
        [
            [(10, 5), generate_random_goal((10, 5), four_room_boxes[0])],
            [(10, 16), generate_random_goal((10, 16), four_room_boxes[1])],
            [(21, 5), generate_random_goal((21, 5), four_room_boxes[2])],
            [(21, 16), generate_random_goal((21, 16), four_room_boxes[3])]
        ],
        # Right Points
        [
            [(5, 10), generate_random_goal((5, 10), four_room_boxes[0])],
            [(5, 21), generate_random_goal((5, 21), four_room_boxes[1])],
            [(16, 10), generate_random_goal((16, 10), four_room_boxes[2])],
            [(16, 21), generate_random_goal((16, 21), four_room_boxes[3])],
        ],
    ],
    category_labels = ["Center Points", "Top Points", "Left Points", "Bottom Points", "Right Points"]
)

def generate_single_pos_changing_goal():
    corner_points = [
        (1, 1),
        (1, 10),
        (10, 1),
        (10, 10)
    ]
    
    structured_data = []

    for corner in corner_points:
        data = []
        for i in range(1, 10):
            for j in range(1, 10):
                if not (i, j) == corner:
                    data.append([corner, (i, j)])
        
        structured_data.append(data)
    
    return structured_data


SINGLE_POS_CHANGING_GOAL = UMAPSettings(
    layout_str="4_rooms",
    structured_data=generate_single_pos_changing_goal(),
    category_labels = ["Top Left Corner", "Top Right Corner", "Bottom Left Corner", "Bottom Right Corner"]
)

def generate_tiling_points():
    centers = [
        (5, 5),
        (5, 16),
        (16, 5),
        (16, 16)
    ]

    structured_data = [[] for _ in range(11)]
    for center in centers:
        for i in range(-4, 6):
            for j in range(-4, 6):
                agent_pos = (center[0] + i, center[1] + j)
                goal_pos = generate_random_goal(agent_pos, (center[0] - 4, center[0] + 5, center[1] - 4, center[1] + 5))

                dist = abs(i) + abs(j)
                structured_data[dist].append([agent_pos, goal_pos])
    
    return structured_data

def generate_corresponding_labels():
    labels = []
    for i in range(1, 11):
        for j in range(1, 11):
            labels.append(f"XY_{i}{j}")

    return labels

def generate_corresponding_points():
    starts = [(1, 1), (1, 12), (12, 1), (12, 12)]
    structured_data = [[] for _ in range(100)]
    for i in range(0, 9):
        for j in range(0, 9):
            for s in starts:
                agent_pos = (s[0] + i, s[1] + j)
                goal_pos = generate_random_goal(agent_pos, (s[0], s[0] + 9, s[1], s[1] + 9))
                structured_data[i * 10 + j].append([agent_pos, goal_pos])

    return structured_data

TILING_POS = UMAPSettings(
    layout_str="4_rooms",
    structured_data=generate_tiling_points(),
    category_labels=["DIST_0", "DIST_1", "DIST_2", "DIST_3", "DIST_4", "DIST_5", "DIST_6", "DIST_7", "DIST_8", "DIST_9", "DIST_10"],
    color_setting="gradient"
)

CORRESPONDING_POS = UMAPSettings(
    layout_str="4_rooms",
    structured_data=generate_corresponding_points(),
    category_labels=generate_corresponding_labels()
)