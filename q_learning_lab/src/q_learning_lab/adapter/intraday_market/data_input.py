from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple, Tuple, List, Dict, Any, Union, Optional
import pandas as pd
from datetime import timedelta, datetime
from enum import Enum
from ...utility.logging import get_logger
import os
import pyarrow.dataset as pqds
import pyarrow as pa
import random
from functools import cached_property
from datetime import datetime, timedelta
from crypto_feature_preprocess.port.interfaces import Training_Eval_Enum
from cryptomarketdata.port.db_client import get_data_db_client, Database_Type
from cachetools import LRUCache

logger = get_logger(__name__)
class Data_Source(ABC):
    @abstractmethod
    def get_market_data_candles(self) -> pd.DataFrame:
        """ Get market data candles
        Returns:
            Any: OHLCV data in panda form
        """
        pass
    
    @abstractmethod
    def reset(self, **kwargs) -> None:
        """ reset data source
        """
        pass
    
    @abstractproperty
    def data_id(self) -> str:
        """ get data id
            useful when u want to cache feature calculation from the data source
        """
        pass

    

class Data_Source_Enum(str, ):
    RANDOM = "RANDOM"
    HISTORICAL = "HISTORICAL"
    REALTIME = "REALTIME"



class Historical_File_Access_Data_Source(Data_Source):
    def __init__(self, parquet_file_path:str, exchange:str, symbol:str) -> None:
        """ Historical file access data source

        Args:
            parquet_file_path (str): Parquet File path
            exchange (str): Exchange name
            symbol (str): Symbol name
        """
        self.symbol = symbol
        self.db_client = get_data_db_client(
            exchange=exchange,
            database_type=Database_Type.PARQUET,
            data_directory=parquet_file_path,
        )
        self._start_date = None
        self._end_date = None
        self._cache = LRUCache(maxsize=100)
        pass
        

    def get_market_data_candles(self) -> pd.DataFrame:
        """  get market data candles from the pool
        Returns:
            Any: OHLCV data in panda form
        """
        
        if self.data_id in self._cache:
            return self._cache[self.data_id]
        
        #Read parquet file and filter the index by start_date and end_date
        from_time:int = int(self.start_date.timestamp()*1000)
        to_time:int = int(self.end_date.timestamp()*1000)

        df = self.db_client.get_candles(
            symbol=self.symbol,
            from_time=from_time,
            to_time=to_time,
        )
        self._cache[self.data_id] = df
        
        return df
    
    @property
    def data_id(self) -> str:
        if self.start_date is None or self.end_date is None:
            raise ValueError(f"start_date or end_date is not set, Please call reset of {__name__}")
        from_time:int = int(self.start_date.timestamp()*1000)
        to_time:int = int(self.end_date.timestamp()*1000)
        return f"{self.symbol}_{from_time}_{to_time}"
    
    def reset(self, **kwargs) -> None:
        """ reset historical market data

        Args:   
            **kwargs: expect {start_date:datetime, end_date:datetime}
        """
        
        self.start_date = kwargs["start_date"]
        self.end_date = kwargs["end_date"]
        pass
    
    @property
    def start_date(self) -> datetime:
        return self._start_date
    
    @property
    def end_date(self) -> datetime:
        return self._end_date
    
    @start_date.setter
    def start_date(self, value:datetime) -> None:
        if isinstance(  value, datetime) == False:
            raise ValueError("start_date is not datetime")
        self._start_date = value

    @end_date.setter
    def end_date(self, value:datetime) -> None:
        if isinstance(  value, datetime) == False:
            raise ValueError("start_date is not datetime")
        self._end_date = value



class Random_File_Access_Data_Source(Data_Source):
    def __init__(self, parquet_file_path:str) -> None:
        dataset: pqds.dataset = pqds.dataset(
            parquet_file_path,
            format="parquet"
        )
        dt: pa.table = dataset.to_table()
        self.df: pd.DataFrame = dt.to_pandas()
        self.pick_episode = -1
        self.reset()
        self._cache = LRUCache(maxsize=100)
        pass
    
    def get_market_data_candles(self) -> pd.DataFrame:
        """ Randomly get market data candles from the pool
        Returns:
            Any: OHLCV data in panda form
        """
        if self.pick_episode < 0:
            self.reset()
        
        if self.data_id in self._cache:
            return self._cache[self.data_id]
        _df = self.get_episode_data(self.pick_episode)
        
        self._cache[self.data_id ] = _df
        return _df
    
    @property
    def data_id(self) -> str:
        return str(self.pick_episode)
    
    def reset(self, **kwargs) -> None:
        """ reset historical market data
            generate a random number as episode id
            for market data extraction
        """
        episodes = self.all_episode_numbers
        #Randomly pick an episode
        self.pick_episode = random.choice(episodes)
        pass
    
    def get_data_dimension(self) -> tuple[int, int]:
        """ Get data dimension
        Returns:
            tuple[int, int]: data dimension
        """
        row, col = self.df.shape
        #Exclude scenario column
        return row, col-1
    
    @cached_property
    def all_episode_numbers(self) -> list[int]:
        """ Get all episode number
        Returns:
            list[int]: all episode number
        """
        return self.df["scenario"].unique().tolist()
    
    def get_episode_data(self, episode_id:int) -> pd.DataFrame:
        """ Get episode data
        Args:
            episode_id (int): episode id
        Returns:
            pd.DataFrame: episode data
        """
        df = self.df[self.df["scenario"]==episode_id]
        #Exclude column "scenario"
        df = df.drop(columns=["scenario"])
        return df
        

    
    


from crypto_feature_preprocess.port.training_data_parquet import (
    prepare_training_data_and_eval_from_parquet,
    derive_min_candle_population_in_episode
)
from .training_interface import TrainingDataBundleParameter
class File_Data_Source_Factory():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def prepare_training_eval_data_source(
                                        bundle_para:TrainingDataBundleParameter
                                        ) -> tuple[Data_Source, Data_Source]:
        """ prepare training data and eval data from file path

        Args:
            bundle_para (TrainingDataBundleParameter): training data bundle parameter

        Returns:
            tuple[data_source, data_source]: training data source and eval data source
        """

        min_candle_population: int = derive_min_candle_population_in_episode(
            candle_size_minutes=bundle_para.candle_size_minutes,
            data_length_days=bundle_para.data_length_days,
            data_presence_ratio=bundle_para.data_presence_ratio
        )
        
        #Parse start_date_ymd string to start_date datetime
        start_date = datetime.strptime(bundle_para.start_date_ymd, "%Y%m%d")
        end_date = datetime.strptime(bundle_para.end_date_ymd, "%Y%m%d")
        (
            num_training_data_row,
            num_eval_data_row,
        ) = prepare_training_data_and_eval_from_parquet(
            exchange=bundle_para.exchange,
            symbol=bundle_para.symbol,
            data_type="parquet",
            data_directory=bundle_para.input_data_dir,
            start_date=start_date,
            end_date=end_date,
            data_length=timedelta(days=bundle_para.data_length_days),
            data_step=timedelta(days=bundle_para.data_step),
            split_ratio=bundle_para.split_ratio,
            output_folder=bundle_para.output_data_dir,
            candle_size=f"{bundle_para.candle_size_minutes}Min",
            min_candle_population=min_candle_population,
        )
        logger.info(f"num_training_data_row: {num_training_data_row}")
        training_data_source = Random_File_Access_Data_Source(
            os.path.join(bundle_para.output_data_dir, str(Training_Eval_Enum.TRAINING.value))
        )
        logger.info(f"num_eval_data_row: {num_eval_data_row}")
        eval_data_source = Random_File_Access_Data_Source(
            os.path.join(bundle_para.output_data_dir, str(Training_Eval_Enum.EVAL.value))
            )
        return training_data_source, eval_data_source

    @staticmethod
    def prepare_historical_data_source(parquet_file_path:str, exchange:str, symbol:str) -> Data_Source:
        """ prepare historical data source

        Args:
            parquet_file_path (str): parquet file path
            exchange (str): exchange name
            symbol (str): symbol name

        Returns:
            Data_Source: historical data source
        """
        data_source = Historical_File_Access_Data_Source(
            parquet_file_path=parquet_file_path,
            exchange=exchange,
            symbol=symbol
        )
        return data_source




