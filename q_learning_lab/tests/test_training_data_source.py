#testing Random_Data_Source
import os

from q_learning_lab.adapter.intraday_market.training_interface import TrainingDataBundleParameter
from q_learning_lab.adapter.intraday_market.data_input import (
    Data_Source,
    File_Data_Source_Factory,
    Random_File_Access_Data_Source,
    Historical_File_Access_Data_Source
)
import pandas as pd
import pytest
from datetime import datetime, timedelta
from q_learning_lab.utility.logging import get_logger
import math
logger = get_logger(__name__)

from q_learning_lab.utility.tools import timeit


input_data_dir = os.environ.get("DATA_DIR")
def test_training_data(get_TrainingDataBundleParameter) -> None:
    """ test the file data source factory
    """
    bundle_param = get_TrainingDataBundleParameter
    train_data_source, eval_data_source = File_Data_Source_Factory.prepare_training_eval_data_source(
        bundle_para=bundle_param
    )

    assert isinstance(train_data_source, Random_File_Access_Data_Source)
    assert isinstance(eval_data_source, Random_File_Access_Data_Source)
    
    train_rows, train_feature_length = train_data_source.get_data_dimension()
    logger.info(f"train_rows: {train_rows}, train_feature_length: {train_feature_length}")
    assert train_feature_length == 5

    eval_rows, eval_feature_length = eval_data_source.get_data_dimension()
    logger.info(f"eval_rows: {eval_rows}, eval_feature_length: {eval_feature_length}")
    assert eval_feature_length == 5

    
    assert train_rows > eval_rows > 0
    assert abs((train_rows / eval_rows) - (bundle_param.split_ratio / (1-bundle_param.split_ratio))) < 0.6

    #randomly pick a scenario
    df = train_data_source.get_market_data_candles()
    assert df is not None
    assert len(df) > 0

def test_historical_data(get_TrainingDataBundleParameter) -> None:
    """_summary_

    Args:
        get_TrainingDataBundleParameter (_type_): _description_
    """
    bundle_param = get_TrainingDataBundleParameter
    data_source:Data_Source = File_Data_Source_Factory.prepare_historical_data_source(
        parquet_file_path=bundle_param.input_data_dir,
        exchange=bundle_param.exchange,
        symbol=bundle_param.symbol,
    )   
    assert isinstance(data_source, Historical_File_Access_Data_Source)
    try:
        data_source.data_id
        assert False
    except ValueError as e:
        assert True

    start_date:datetime = datetime.strptime(bundle_param.start_date_ymd, "%Y%m%d")
    end_date:datetime = start_date + timedelta(days=1)
    data_source.reset(start_date=start_date, end_date=end_date)

    @timeit()
    def __get_data()->pd.DataFrame:
        return  data_source.get_market_data_candles()
    candles = __get_data()
    load_data_time_1st:float = __get_data.execution_time[0]
    assert candles is not None
    assert len(candles) > 0
    outofrange = candles[((candles.index<start_date) | (candles.index>end_date))]
    assert outofrange.empty

    candles_2nd = __get_data()
    load_data_time_2nd:float = __get_data.execution_time[0]
    assert candles_2nd is not None
    assert load_data_time_2nd < load_data_time_1st
    
    #logger.info(candles)
    pass

def test_random_data(get_TrainingDataBundleParameter) -> None:
    bundle_param = get_TrainingDataBundleParameter

    training_data_source, eval_data_source = File_Data_Source_Factory.prepare_training_eval_data_source(
        bundle_para=bundle_param
    )
    @timeit()
    def __get_data()->pd.DataFrame:
        return  training_data_source.get_market_data_candles()
    df = __get_data()
    load_data_time_1st:float = __get_data.execution_time[0]
    logger.debug(f"load_data_time_1st: {load_data_time_1st}")
    assert df is not None
    start_date = df.index[0]
    end_date = df.index[-1]
    logger.info(f"start_date: {start_date}, end_date: {end_date}")
    logger.info(f"time length: {end_date-start_date}")
    assert abs((end_date-start_date).days - bundle_param.data_length_days)<=1

    df2:pd.DataFrame = __get_data()
    load_data_time_2nd:float = __get_data.execution_time[0]
    logger.debug(f"load_data_time_2nd: {load_data_time_2nd}")

    assert load_data_time_2nd < load_data_time_1st

    assert start_date == df2.index[0]
    assert end_date == df2.index[-1]

    training_data_source.reset()
    df3:pd.DataFrame = __get_data()
    load_data_time_3rd:float = __get_data.execution_time[0]
    logger.debug(f"load_data_time_3rd: {load_data_time_3rd}")
    assert load_data_time_3rd > load_data_time_2nd

    assert start_date != df3.index[0]
    assert end_date != df3.index[-1]
    assert abs((df3.index[-1]-df3.index[0]).days - bundle_param.data_length_days) <= 1

    
    assert len(training_data_source.all_episode_numbers) > len(eval_data_source.all_episode_numbers) > 0

