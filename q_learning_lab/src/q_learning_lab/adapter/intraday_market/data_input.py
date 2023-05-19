from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple, List, Dict, Any, Union, Optional
import pandas as pd
from datetime import timedelta, datetime
from enum import Enum
from ...utility.logging import get_logger
import os
import pyarrow.dataset as pqds
import pyarrow as pa
import random

from crypto_feature_preprocess.port.interfaces import Training_Eval_Enum

logger = get_logger(__name__)
class Data_Source(ABC):
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
    
class Random_Data_Source(Data_Source):
    def __init__(self, parquet_file_path:str) -> Any:
        dataset: pqds.dataset = pqds.dataset(
            parquet_file_path,
            format="parquet"
        )
        dt: pa.table = dataset.to_table()
        self.df: pd.DataFrame = dt.to_pandas()
        pass
    
    def get_market_data_candles(self) -> pd.DataFrame:
        """ Randomly get market data candles from the pool
        Returns:
            Any: OHLCV data in panda form
        """
        pick_scenario = random.randint(0, self.df["scenario"].max())
        _df = self.df[self.df["scenario"]==pick_scenario]
        return _df

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
            data_presence_ratio=0.8
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
        training_data_source = Random_Data_Source(
            os.path.join(bundle_para.output_data_dir, str(Training_Eval_Enum.TRAINING.value))
        )
        logger.info(f"num_eval_data_row: {num_eval_data_row}")
        eval_data_source = Random_Data_Source(
            os.path.join(bundle_para.output_data_dir, str(Training_Eval_Enum.EVAL.value))
            )
        return training_data_source, eval_data_source




