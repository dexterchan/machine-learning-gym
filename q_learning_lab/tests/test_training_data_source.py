#testing Random_Data_Source
import os

from q_learning_lab.adapter.intraday_market.training_interface import TrainingDataBundleParameter
from q_learning_lab.adapter.intraday_market.data_input import (
    Data_Source,
    File_Data_Source_Factory,
    Random_File_Access_Data_Source,
    Historical_File_Access_Data_Source
)
import pytest
from datetime import datetime, timedelta
from q_learning_lab.utility.logging import get_logger
import math
logger = get_logger(__name__)


@pytest.fixture()
def get_TrainingDataBundleParameter() -> TrainingDataBundleParameter:
    bundle_param: TrainingDataBundleParameter = TrainingDataBundleParameter(
        input_data_dir=os.environ.get("DATA_DIR"),
        exchange="kraken",
        symbol="ETHUSD",
        start_date_ymd="20220401",
        end_date_ymd="20230401",
        data_length_days=3,
        data_step=1,
        split_ratio=0.9,
        output_data_dir="/tmp/output",
        candle_size_minutes=15,
    )
    return bundle_param

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
    start_date:datetime = datetime.strptime(bundle_param.start_date_ymd, "%Y%m%d")
    end_date:datetime = start_date + timedelta(days=1)
    candles = data_source.get_market_data_candles(start_date=start_date, end_date=end_date)
    assert candles is not None
    assert len(candles) > 0
    outofrange = candles[((candles.index<start_date) | (candles.index>end_date))]
    assert outofrange.empty
    
    #logger.info(candles)
    pass

