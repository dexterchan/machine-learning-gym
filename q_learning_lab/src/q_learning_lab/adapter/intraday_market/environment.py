from __future__ import annotations
from ...adapter import Interface_Environment
from .training_interface import TrainingDataBundleParameter
import random
import string
import re
import os
import numpy as np
from typing import Any, Union, Optional
from enum import Enum

#numeric action which is compatible with gym
from ...domain.models.intraday_market_models import Intraday_Trade_Action_Space 

from tradesignal_mtm_runner.models import Buy_Sell_Action_Enum, Mtm_Result
from tradesignal_mtm_runner.trade_reward import TradeBookKeeperAgent
from tradesignal_mtm_runner.config import PnlCalcConfig


from .data_input import (
    Data_Source,
    File_Data_Source_Factory
)

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
        self._read_pointer:int = 0
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
    
    def _init_runner_state(self) -> None:
        self._read_pointer = 0
        self._feature_cache.clear()

    def reset(self, **kwargs) -> pd.DataFrame:
        """Reset the data source and return the market data candles

        Returns:
            pd.DataFrame: _description_
        """
        self._data_source.reset(kwargs=kwargs)
        self._init_runner_state()
        df:pd.DataFrame = self._data_source.get_market_data_candles()
        logger.debug(df)
        if len(df) == 0:
            raise Exception("Data not found from this datasouce")
        return df
    
    def __iter__(self) -> FeatureRunner:
        for _ in (self._data_source):
            self._init_runner_state()
            yield self

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
        
        observation:np.ndarray = features_output.feature_data[self._read_pointer]
        time_inx = features_output.time_index[self._read_pointer]
        

        #check the boundary if it is the end:
        if self._read_pointer < num_of_feature and increment_step:
            self._read_pointer+=1
        if self._read_pointer >= num_of_feature:
            end_of_episode = True
        return observation, time_inx, end_of_episode
    
    def get_current_market_data(self) -> pd.DataFrame:
        return self._data_source.get_market_data_candles()
    
    @property
    def episode_id(self) -> str:
        return self._data_source.data_id
    
    @property
    def read_pointer(self) -> int:
        return self._read_pointer
    
    @property
    def max_step(self) -> int:
        max_step, _ = self.calculate_features().feature_data.shape
        return max_step

    @property
    def data_dimension(self)->tuple[int,int]:
        return self._data_source.get_data_dimension()




class Intraday_Market_Environment(Interface_Environment):

    def __init__(self, params: Optional[dict]={}) -> None:
        #The data source can be realtime or random historical data
        self.symbol:str = params.get("symbol", "unknown")
        self.price_movement_col:str = params.get("price_movement_col", "close")
        self._observation_space_dim = 0
        self._feature_runner:FeatureRunner = None
        self._pnl_calc_config:PnlCalcConfig = None
        self._current_data:pd.DataFrame = None
        self._trade_order_agent:TradeBookKeeperAgent = None
        self._step_counter:int = 0
        self._iterate_feature_mode_on:bool = False
        
    @classmethod
    def parse_command(cls, command: str) -> str:
        """_summary_

        Args:
            command (str): _description_

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            str: _description_
        """
        if command.upper() == "RANDOM":
            return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
        else:
            return ""

    @classmethod
    def _regex_replace_with_env_variables(cls, raw_data_config:dict) -> dict:
        """_summary_

        Args:
            raw_data_config (dict): configuration dictionary

        Returns:
            dict: _description_
        """
        env_var_regex = re.compile(r"\$\{(\w+)\}")
        #in the raw_data_config dictionary, we may have some env variables
        #we need to replace them with the actual value from the environment variable
        for k, v in raw_data_config.items():
            if isinstance(v, str):
                match = env_var_regex.match(v)
                if match:
                    env_var_name = match.group(1)
                    env_var_value = os.getenv(env_var_name)
                    if env_var_value:
                        raw_data_config[k] = env_var_value
                    else:
                        raise ValueError(f"Environment variable {env_var_name} is not defined")
        return raw_data_config
    
    @classmethod
    def _regex_sub_with_command(cls, raw_data_config:dict) -> dict:
        """regex substitution with command

        Args:
            raw_data_config (dict): configuration dictionary

        Returns:
            dict: _description_
        """
        command_var_regex = re.compile(r"\$\((\w+)\)")
        for k, v in raw_data_config.items():
            if isinstance(v, str):
                match = command_var_regex.search(v)
                if match:
                    command = match.group(1)
                    command_value = cls.parse_command(command)
                    if command_value:
                        raw_data_config[k] = command_var_regex.sub(command_value, v)
                    else:
                        raise ValueError(f"Command {command} is not defined")
        return raw_data_config

    
    @classmethod
    def create_from_config(cls, 
                           raw_data_config:dict, 
                           feature_schema_config:dict,
                           pnl_config:dict,
                           ) -> tuple[Intraday_Market_Environment, Intraday_Market_Environment]:
        """
        Create the environment from TrainingDataBundleParameter config dictionary

        Args:
            raw_data_config (dict): TrainingDataBundleParameter config dictionary
            feature_schema_config (dict): feature schema config dictionary e.g. RSI_Feature_Interface
            pnl_config (dict): pnl config dictionary to convert to PnlCalcConfig

        Returns:
            tuple[Intraday_Market_Environment, Intraday_Market_Environment]: training and evaluation environment

        """
        
        #in the raw_data_config dictionary, we may have some env variables
        #we need to replace them with the actual value from the environment variable
        cls._regex_replace_with_env_variables(raw_data_config)

        
        #in the raw_data_config dictionary, we may have some command variables
        #we need to replace them with the actual value from the command
        cls._regex_sub_with_command(raw_data_config)


        bundle_params = TrainingDataBundleParameter(**raw_data_config)
        logger.info(f"Note: Execute training with {bundle_params._asdict()}")
        
        train_data_source, eval_data_source = File_Data_Source_Factory.prepare_training_eval_data_source(
            bundle_para=bundle_params
        )

        def __prepare_env(my_data_source:Data_Source) -> Intraday_Market_Environment:
            _data_source = my_data_source
            _feature_runner_map:dict[str, FeatureRunner] = {
                g_type: FeatureRunner(
                    data_source=_data_source,
                    feature_generator_type=g_type,
                    feature_plan=g_params)
                for g_type, g_params in feature_schema_config.items()
            }
            _intraday_market_train_env: Intraday_Market_Environment = Intraday_Market_Environment()
            for _, item in _feature_runner_map.items():
                _intraday_market_train_env.register_feature_runner(item)
            _intraday_market_train_env.register_pnl_calc_config(pnl_config)
            return _intraday_market_train_env
        return __prepare_env(train_data_source), __prepare_env(eval_data_source)

    def register_feature_runner(self, feature_runner:FeatureRunner):
        self._feature_runner = feature_runner    
        self._observation_space_dim += feature_runner.feature_observation_space_dim
    
    
    def register_pnl_calc_config(self, pnl_calc_config:Union[PnlCalcConfig, dict]):
        """_summary_

        Args:
            pnl_calc_config (Union): either PnlCalcConfig or dict of PnlCalcConfig
        """
        if isinstance(pnl_calc_config, dict):
            #convert dict to PnlCalcConfig
            self._pnl_calc_config = PnlCalcConfig(**pnl_calc_config)
        else:
            self._pnl_calc_config = pnl_calc_config
        pass

    def render(self):
        """Render and display current state of the environment"""
        raise NotImplementedError("Render function not implemented yet")
        pass

    def reset(self, **kwargs) -> tuple[np.ndarray, Any]:
        # Return numpy array of (dim,1) in shape
        if not self._iterate_feature_mode_on:
            self._feature_runner.reset(**kwargs)
        observation, time_inx, _ = self._feature_runner.stateful_step(increment_step=False)
        logger.info("Reset environment, current time index: %s", time_inx)
        #Reset current data
        self._current_data = self._feature_runner.get_current_market_data().copy()
        self._current_data["buy"] = 0
        self._current_data["sell"] = 0
        self._current_data["price_movement"] = self._current_data[self.price_movement_col].diff(1)
        self._current_data["inx"] = np.arange(len(self._current_data))

        #Reset trade order agent if we reset the environment
        self._trade_order_agent = TradeBookKeeperAgent(
            symbol=self.symbol, pnl_config=self._pnl_calc_config, fixed_unit=True
        )
        self._step_counter = 0
        return observation, time_inx
    
    # iterate all feature runners
    def __iter__(self) -> Intraday_Market_Environment:
        self._iterate_feature_mode_on = True
        _feature_runner_itr = iter(self._feature_runner)
        for _runner in _feature_runner_itr:
            self.register_feature_runner(_runner)
            yield self


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
        #Check if trade order generator is registered
        if self._trade_order_agent is None:
            raise ValueError("trade order generator is not registered, please register pnl_calc_config and reset the environment")
        #Check if action is valid
        if not isinstance(action, Intraday_Trade_Action_Space):
            action = Intraday_Trade_Action_Space(action)
            #raise ValueError(f"Action: {action} is not valid")

        #Convert Intraday_Trade_Action_Space to Buy_Sell_Action_Enum of tradesignal runner
        buy_sell_action_space:Buy_Sell_Action_Enum = Intraday_Trade_Action_Space.convert_to_buy_sell_action_enum(action)
        logger.debug("Step %s, action: %s", self._step_counter, buy_sell_action_space)
        #Get current observation
        observation, time_inx, end_of_episode = self._feature_runner.stateful_step(increment_step=True)
        if buy_sell_action_space == Buy_Sell_Action_Enum.BUY:
            self._current_data.loc[self._current_data.index == time_inx, ["buy"]] = 1
        elif buy_sell_action_space == Buy_Sell_Action_Enum.SELL:
            self._current_data.loc[self._current_data.index == time_inx, ["sell"]] = 1
        
        #Run the trade order agent at time stamp
        #inx:int = self._current_data[self._current_data.index == time_inx]["inx"][0]
        _price_value, _price_movement , _inx = self._current_data.loc[self._current_data.index == time_inx, 
                                                               [self.price_movement_col,
                                                                "price_movement", 
                                                                "inx"] ] \
                                                                .values[0].tolist()
        
        self._trade_order_agent.run_at_timestamp(
                dt=time_inx.to_pydatetime(),
                price=_price_value,
                price_diff=_price_movement,
                buy_sell_action=buy_sell_action_space,
            )
        logger.debug("MTM history: %s", self._trade_order_agent.mtm_history_value)
        logger.debug("MTM history length: %s", len(self._trade_order_agent.mtm_history_value))
        interm_reward = self._trade_order_agent.mtm_history_value[self.step_counter]
        
        logger.debug("Interm reward: %s", interm_reward)

        #increment the step counter
        self.step_counter += 1

        return (
            observation,
            interm_reward,
            end_of_episode,
            end_of_episode,
            {}
            
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
    
    @property
    def step_counter(self) -> int:
        """Return step counter

        Returns:
            int: step counter
        """
        return self._step_counter
    
    @step_counter.setter
    def step_counter(self, value:int):
        """Set step counter

        Args:
            value (int): value to be set
        """
        self._step_counter = value

    def get_data_dimension(self) -> tuple[int,int]:
        """Get data dimension

        Returns:
            tuple[int,int]: data dimension
        """
        return self._feature_runner.data_dimension
    
    def get_current_market_data(self) -> pd.DataFrame:
        """get current market data

        Returns:
            pd.DataFrame: Data frame
        """
        return self._feature_runner.get_current_market_data()
    
    @property
    def measure_result(self) -> float:
        sharpe_ratio = self._trade_order_agent.calculate_sharpe_ratio()
        return sharpe_ratio