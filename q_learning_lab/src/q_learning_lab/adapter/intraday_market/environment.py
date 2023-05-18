from ...adapter import Interface_Environment
import random
import numpy as np
from typing import Any
from enum import Enum

#numeric action which is compatible with gym
from ...domain.models.intraday_market_models import Intraday_Trade_Action_Space
from  tradesignal_mtm_runner.models import Buy_Sell_Action_Enum
from .data_input import Data_Source

#Import crypto_feature_precess package here
from crypto_feature_preprocess.port.interfaces import (
    Feature_Definition,
    RSI_Feature_Interface,
    SMA_Feature_Interface,
    Log_Price_Feature_Interface,
    Feature_Enum
)


class Intraday_Market_Environment(Interface_Environment):
    def __init__(self, params: dict, data_market_source:Data_Source) -> None:
        #The data source can be realtime or random historical data

        self._observation_space_dim = 4
        

    def render(self):
        """Render and display current state of the environment"""
        pass

    def reset(self) -> tuple[np.ndarray, Any]:
        # Return numpy array of (dim,1) in shape
        return np.random.rand(self._observation_space_dim), None

    def step(self, action: Intraday_Trade_Action_Space) -> tuple[np.ndarray, float, bool, bool, dict]:
        """step function
            Return random values for testing
            - observation: np.ndarray
            - reward: float
            - done: bool
            - truncated: bool
            - info: probability

        Args:
            action (Intraday_Trade_Action_Space): action to be taken

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: states, reward, done, truncated, info
        """

        buy_sell_action_space:Buy_Sell_Action_Enum = Intraday_Trade_Action_Space.convert_to_buy_sell_action_enum(action)



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
