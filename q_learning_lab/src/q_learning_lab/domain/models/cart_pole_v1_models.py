from __future__ import annotations
from enum import Enum
from typing import Any, NamedTuple
from pathlib import Path


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    n_max_steps: int  # Max steps per episode
    learning_rate: float  # Learning rate (alpha)
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    savefig_folder: Path  # Root folder where plots are saved
    start_epsilon: float = 1.0  # Starting exploration probability
    min_epsilon: float = 0.05  # Minimum exploration probability
    decay_rate: float = 0.001  # Exponential decay rate for exploration prob
    min_replay_size: int = 1000  # minimum replay size to train the model


class Action_Space(int, Enum):
    LEFT = 0
    RIGHT = 1
