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
from crypto_feature_preprocess.port.features import Feature_Output
from q_learning_lab.utility.logging import get_logger
from cachetools import LRUCache                 

logger = get_logger(__name__)

class FeatureRunner():
    def __init__(self, data_source:Data_Source, feature_generator_type:str, feature_plan:dict[str, list[dict]] ) -> None:
        self._data_source = data_source
        self.feature_generator_type:Feature_Generator_Enum = Feature_Generator_Enum(feature_generator_type)
        self.feature_schema:dict[str, Feature_Generator_Interface] = {}
        for col, _feature_struct in feature_plan.items():
            self.feature_schema[col] = Feature_Generator_Factory.create_generator(self.feature_generator_type, _feature_struct)
        self.read_pointer:int = 0
        self.feature_plan = feature_plan
        pass
    
    @property
    def feature_observation_space_dim(self) -> int:
        """ calcuatie the feature observation dimension
            Return:
                int: _description_
        """
        feature_dim:int = 0
        for col, _feature_struct_lst in self.feature_plan.items():
            for _feature_struct in _feature_struct_lst:
                feature_dim += _feature_struct["feature_params"]["dimension"]
        return feature_dim
    
    def calculate_features(self) -> Feature_Output:
        """ calculate features
            it should have cached feature data to avoid re-calculate for each call
            the cache can be reset by calling reset()

        Returns:
            Feature_Output: _description_
        """
        #Get data from data source
        df:pd.DataFrame = self._data_source.get_market_data_candles()
        features_output_list:list[Feature_Output] = []
        for col, _feature_generator in self.feature_schema.items():
            logger.info("Calculating feature for column: %s", col)
            g:Feature_Generator_Interface = _feature_generator
            feature_output:Feature_Output = g.generate_feature(data_vector=df[col])
            features_output_list.append(feature_output)
            logger.info("Feature for column: %s, shape: %s", col, feature_output.feature_data.shape)
        
        new_feature_output:Feature_Output = Feature_Output.merge_feature_output_list(feature_output_list=features_output_list)
        logger.info("Final feature shape: %s", new_feature_output.feature_data)
        return new_feature_output
        

    def reset(self, **kwargs):
        """reset data source and feature generator
        """
        self._data_source.reset(kwargs=kwargs)
        pass




class Intraday_Market_Environment(Interface_Environment):
    def __init__(self, params: dict, feature_runner:FeatureRunner) -> None:
        #The data source can be realtime or random historical data
        self.feature_runner = feature_runner
        self._observation_space_dim = feature_runner.feature_observation_space_dim
        
    def _fresh_data_sources(self):
        """fresh data sources
        """
        
        pass
    def render(self):
        """Render and display current state of the environment"""
        raise NotImplementedError("Render function not implemented yet")
        pass

    def reset(self, **kwargs) -> tuple[np.ndarray, Any]:
        # Return numpy array of (dim,1) in shape
        self.feature_runner.reset(**kwargs)
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
