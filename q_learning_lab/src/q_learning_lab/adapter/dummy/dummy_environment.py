from ...adapter import Interface_Environment
import random
import numpy as np
from typing import Any
from enum import Enum


class Action_Space(int, Enum):
    NO_ACTION = 0
    LEFT = 1
    RIGHT = 2


class Dummy_Environment(Interface_Environment):
    def __init__(self, params: dict) -> None:
        self._observation_space_dim = 4

    def render(self):
        """Render and display current state of the environment"""
        pass

    def reset(self) -> tuple[np.ndarray, Any]:
        # Return numpy array of (dim,1) in shape
        return np.random.rand(self._observation_space_dim), None

    def step(self, action: Action_Space) -> tuple[np.ndarray, float, bool, bool, dict]:
        """step function
            Return random values for testing
            - observation: np.ndarray
            - reward: float
            - done: bool
            - truncated: bool
            - info: probability

        Args:
            action (Action_Space): action to be taken

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: _description_
        """
        return (
            np.random.rand(*self.observation_space_dim),
            random.random(),
            random.choice([True, False]),
            random.choice([True, False]),
            {},
        )

    def close(self):
        """Close the environment"""
        pass

    def get_description(self) -> Any:
        """Get description of the environment

        Returns:
            Any: description of the environment
        """
        return "dummy environment"

    def sample_action_space(self) -> Any:
        """Sample action space

        Returns:
            Any: action space of the specifc environment
        """
        return random.choice(list(Action_Space))

    @property
    def action_space_dim(self) -> int:
        """action space dimension

        Returns:
            int: action space dimension
        """
        return len(list(Action_Space))

    @property
    def observation_space_dim(self) -> tuple[int]:
        """Return action space dimension
            for gym
            if hasattr(self.env.observation_space, "n"):
                return self.env.observation_space.n
            elif hasattr(self.env.observation_space, "shape"):
                return self.env.observation_space.shape

        Returns:
            tuple[int]: dimension of the observation space
        """
        return (self._observation_space_dim,)
