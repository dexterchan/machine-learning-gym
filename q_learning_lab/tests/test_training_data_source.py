#testing Random_Data_Source
import os

from q_learning_lab.adapter.intraday_market.training_interface import TrainingDataBundleParameter
from q_learning_lab.adapter.intraday_market.data_input import (
    Data_Source,
    File_Data_Source_Factory,
    Random_Data_Source
)
import pytest


@pytest.fixture()
def get_TrainingDataBundleParameter() -> TrainingDataBundleParameter:
    bundle_param: TrainingDataBundleParameter = TrainingDataBundleParameter(
        input_data_dir=os.environ.get("DATA_DIR"),
        exchange="kraken",
        symbol="ETHUSD",
        start_date_ymd="20200101",
        end_date_ymd="20210101",
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

    File_Data_Source_Factory.prepare_training_eval_data_source(
        bundle_para=get_TrainingDataBundleParameter
    )

