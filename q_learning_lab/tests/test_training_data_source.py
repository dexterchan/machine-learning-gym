#testing Random_Data_Source
import os

from q_learning_lab.adapter.intraday_market.training_interface import TrainingDataBundleParameter
from q_learning_lab.adapter.intraday_market.data_input import (
    Data_Source,
    File_Data_Source_Factory,
    Random_Data_Source
)
import pytest

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
        split_ratio=0.8,
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

    assert isinstance(train_data_source, Random_Data_Source)
    assert isinstance(eval_data_source, Random_Data_Source)
    
    train_rows, train_feature_length = train_data_source.get_data_dimension()
    logger.info(f"train_rows: {train_rows}, train_feature_length: {train_feature_length}")
    assert train_feature_length == 5

    eval_rows, eval_feature_length = eval_data_source.get_data_dimension()
    logger.info(f"eval_rows: {eval_rows}, eval_feature_length: {eval_feature_length}")
    assert eval_feature_length == 5

    
    assert train_rows > eval_rows > 0
    assert abs((train_rows / eval_rows) - (bundle_param.split_ratio / (1-bundle_param.split_ratio))) < 0.1

    #randomly pick a scenario
    df = train_data_source.get_market_data_candles()
    assert df is not None
    assert len(df) > 0


