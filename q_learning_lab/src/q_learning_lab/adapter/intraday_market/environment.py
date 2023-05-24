from ...adapter import Interface_Environment
import random
import numpy as np
from typing import Any
from enum import Enum

#numeric action which is compatible with gym
from ...domain.models.intraday_market_models import Intraday_Trade_Action_Space
from  tradesignal_mtm_runner.models import Buy_Sell_Action_Enum
from .data_input import Data_Source

import pandas as pd
#Import crypto_feature_precess package here
from .features.feature_port import Feature_Generator_Factory, Feature_Generator_Enum
from .features.feature_interface import Feature_Generator_Interface


class FeatureRunner():
    def __init__(self, data_source:Data_Source, feature_generator_type:str, feature_plan:dict[str, list[dict]] ) -> None:
        self._data_source = data_source
        self.feature_generator_type:Feature_Generator_Enum = Feature_Generator_Enum(feature_generator_type)
        self.feature_schema:dict[str, Feature_Generator_Interface] = {}
        for col, _feature_struct in feature_plan.items():
            self.feature_schema[col] = Feature_Generator_Factory.create_generator(self.feature_generator_type, _feature_struct)
        self.read_pointer:int = 0
        pass

    
    def calculate_features(self) -> np.ndarray:
        #Get data from data source
        df:pd.DataFrame = self._data_source.get_market_data_candles()
        features_list = []
        for col, _feature_generator in self.feature_schema.items():
            g:Feature_Generator_Interface = _feature_generator
            _data:np.ndarray = g.generate_feature(price_vector=df[col])
            features_list.append(_data)
        
        return np.concatenate(features_list, axis=1)

    def reset(self, **kwargs):
        """reset data source and feature generator
        """
        self._data_source.reset(kwargs=kwargs)
        pass




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
