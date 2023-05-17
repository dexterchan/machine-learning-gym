from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple, List, Dict, Any, Union, Optional
import pandas as pd

from enum import Enum

class data_source(ABC):
    @abstractmethod
    def get_market_data_candles(self) -> pd.DataFrame:
        """ Get market data candles
        Returns:
            Any: OHLCV data in panda form
        """
        pass

class Data_Source_Enum(str, ):
    RANDOM = "RANDOM"
    HISTORICAL = "HISTORICAL"
    REALTIME = "REALTIME"
    



class Data_Source_Factory():
    def __init__(self) -> None:
        pass
    

    def prepare_training_eval_data_source(self, file_path:str) -> tuple[data_source, data_source]:
        """ prepare training data and eval data from file path

        Args:
            file_path (str): file path of the data

        Returns:
            tuple[data_source, data_source]: training data source and eval data source
        """
        pass

class Random_Data_Source(data_source):
    
    def get_training_data_candles(self) -> pd.DataFrame:
        """ Randomly get market data candles from the pool
        Returns:
            Any: OHLCV data in panda form
        """
        pass
