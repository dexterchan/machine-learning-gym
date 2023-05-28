from ...adapter import Interface_Environment
import random
import numpy as np
from typing import Any, Union, Optional
from enum import Enum

#numeric action which is compatible with gym
from ...domain.models.intraday_market_models import Intraday_Trade_Action_Space 

from tradesignal_mtm_runner.models import Buy_Sell_Action_Enum, Mtm_Result
from tradesignal_mtm_runner.runner_mtm import Trade_Mtm_Runner
from tradesignal_mtm_runner.config import PnlCalcConfig


from .data_input import Data_Source

import pandas as pd
#Import crypto_feature_precess package here
from .features.feature_port import Feature_Generator_Factory, Feature_Generator_Enum
from .features.feature_interface import Feature_Generator_Interface
from crypto_feature_preprocess.port.features import Feature_Output
from q_learning_lab.utility.logging import get_logger
from cachetools import LRUCache                 

from datetime import datetime

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
        self._feature_cache:LRUCache = LRUCache(maxsize=100)
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
        #Check cache
        #Return cached feature if exists
        if self._data_source.data_id in self._feature_cache:
            return self._feature_cache[self._data_source.data_id]
        
        #If not in cache, calculate feature
        #Get data from data source
        df:pd.DataFrame = self._data_source.get_market_data_candles()
        features_output_list:list[Feature_Output] = []
        for col, _feature_generator in self.feature_schema.items():
            logger.debug("Calculating feature for column: %s", col)
            g:Feature_Generator_Interface = _feature_generator
            feature_output:Feature_Output = g.generate_feature(data_vector=df[col])
            features_output_list.append(feature_output)
            logger.debug("Feature for column: %s, shape: %s", col, feature_output.feature_data.shape)
        
        new_feature_output:Feature_Output = Feature_Output.merge_feature_output_list(feature_output_list=features_output_list)
        logger.info("Final feature shape: %s", new_feature_output.feature_data.shape)
        #Cache feature
        self._feature_cache[self._data_source.data_id] = new_feature_output
        return new_feature_output
        

    def reset(self, **kwargs) -> pd.DataFrame:
        """Reset the data source and return the market data candles

        Returns:
            pd.DataFrame: _description_
        """
        self._data_source.reset(kwargs=kwargs)
        self.read_pointer = 0
        self._feature_cache.clear()
        return self._data_source.get_market_data_candles()
        

    def stateful_step(self, increment_step:bool=True) -> tuple[np.ndarray, datetime, bool]:
        """_summary_

        Args:
            increment_step (bool, optional): increment the step. Defaults to True.

        Returns:
            tuple[np.ndarray, datetime, bool]: observation, time_index, end_of_episode(bool)
        """
        end_of_episode:bool = False
        features_output:Feature_Output = self.calculate_features()
        num_of_feature, _ = features_output.feature_data.shape
        
        observation:np.ndarray = features_output.feature_data[self.read_pointer]
        time_inx = features_output.time_index[self.read_pointer]
        

        #check the boundary if it is the end:
        if self.read_pointer < num_of_feature-1 and increment_step:
            self.read_pointer+=1
        if self.read_pointer >= num_of_feature:
            end_of_episode = True
        return observation, time_inx, end_of_episode
    
    def get_current_market_data(self) -> pd.DataFrame:
        return self._data_source.get_market_data_candles()




class Intraday_Market_Environment(Interface_Environment):

    def __init__(self, params: Optional[dict]={}) -> None:
        #The data source can be realtime or random historical data
        self.symbol:str = params.get("symbol", "unknown")
        self._observation_space_dim = 0
        self._feature_runner:FeatureRunner = None
        self.reward_generator:Trade_Mtm_Runner = None
        self._current_data:pd.DataFrame = None
        

    def register_feature_runner(self, feature_runner:FeatureRunner):
        self._feature_runner = feature_runner    
        self._observation_space_dim += feature_runner.feature_observation_space_dim
    
    
    def register_reward_generator(self, reward_generator:Union[Trade_Mtm_Runner, dict]):
        """_summary_

        Args:
            reward_generator (Union): either Trade_Mtm_Runner or dict of PnlCalcConfig
        """
        if isinstance(reward_generator, dict):
            #convert dict to PnlCalcConfig
            pnl_calc_config:PnlCalcConfig = PnlCalcConfig(**reward_generator)
            self.reward_generator = Trade_Mtm_Runner(pnl_config=pnl_calc_config)
        else:
            self.reward_generator = reward_generator
        pass

    def render(self):
        """Render and display current state of the environment"""
        raise NotImplementedError("Render function not implemented yet")
        pass

    def reset(self, **kwargs) -> tuple[np.ndarray, Any]:
        # Return numpy array of (dim,1) in shape
        self._feature_runner.reset(**kwargs)
        observation, time_inx, _ = self._feature_runner.stateful_step(increment_step=False)
        logger.info("Reset environment, current time index: %s", time_inx)
        self._current_data = self._feature_runner.get_current_market_data().copy()
        self._current_data["buy"] = 0
        self._current_data["sell"] = 0
        self._current_data["inx"] = np.arange(len(self._current_data))
        return observation, time_inx

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
        #Check if feature runner is registered
        if self._feature_runner is None:
            raise ValueError("Feature runner is not registered")
        #Check if reward generator is registered
        if self.reward_generator is None:
            raise ValueError("Reward generator is not registered")
        #Check if action is valid
        if not isinstance(action, Intraday_Trade_Action_Space):
            raise ValueError("Action is not valid")

        #Convert Intraday_Trade_Action_Space to Buy_Sell_Action_Enum of tradesignal runner
        buy_sell_action_space:Buy_Sell_Action_Enum = Intraday_Trade_Action_Space.convert_to_buy_sell_action_enum(action)
        #Get current observation
        observation, time_inx, end_of_episode = self._feature_runner.stateful_step(increment_step=True)
        if buy_sell_action_space == Buy_Sell_Action_Enum.BUY:
            self._current_data.loc[self._current_data.index == time_inx, ["buy"]] = 1
        elif buy_sell_action_space == Buy_Sell_Action_Enum.SELL:
            self._current_data.loc[self._current_data.index == time_inx, ["sell"]] = 1
        
        #update reward history
        mtm_result:Mtm_Result = self.reward_generator.calculate(
            symbol=self.symbol,
            buy_signal_dataframe=self._current_data[:time_inx],
            sell_signal_dataframe=self._current_data[:time_inx],
        )
        # read pnl_timeline dict and extract "mtm_ratio" by the key: datetime ms timestamp
        
        inx = self._current_data[self._current_data.index == time_inx]["inx"]
        
        return (
            observation,
            random.random(),
            end_of_episode,
            random.choice([True, False]),
            {"mtm_ratio":(mtm_result.pnl_timeline["mtm_ratio"]),
             "time_inx":time_inx,
             "inx":inx,
             "long_trade_outstanding":mtm_result.long_trades_outstanding,
             "long_trade_archive":mtm_result.long_trades_archive,},
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
        return random.choice(list(Intraday_Trade_Action_Space))

    @property
    def action_space_dim(self) -> int:
        """action space dimension

        Returns:
            int: action space dimension
        """
        return len(list(Intraday_Trade_Action_Space))

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
