import numpy as np
from typing import Any


class Interface_Environment:
    def render(self):
        raise NotImplementedError("render() not implemented.")

    def reset(self) -> tuple[np.ndarray, Any]:
        return None

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        return None

    def close(self):
        raise NotImplementedError("close() not implemented.")

    def get_description(self) -> Any:
        return None

    def sample_action_space(self) -> Any:
        """Sample action space

        Returns:
            Any: action space of the specifc environment
        """
        return None

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
        raise NotImplementedError("No dimension found for observation space.")

    @property
    def action_space_dim(self) -> int:
        """action space dimension

        Returns:
            int: action space dimension
        """
        raise NotImplementedError("No dimension found for action space.")
