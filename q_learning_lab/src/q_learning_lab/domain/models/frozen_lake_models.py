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
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved
    start_epsilon: float = 1.0  # Starting exploration probability
    min_epsilon: float = 0.05  # Minimum exploration probability
    decay_rate: float = 0.001  # Exponential decay rate for exploration prob


class Action_Space(int, Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


# class State:
#     def __init__(self) -> None:
#         pass


# class Policy:
#     def __init__(self) -> None:
#         pass

#     def act(self, observation: Any) -> Action_Space:
#         raise NotImplementedError("Policy not implemented")


# class Reward:
#     def __init__(self) -> None:
#         pass

#     def get_reward(self, state: State) -> float:
#         raise NotImplementedError("Reward not implemented")
