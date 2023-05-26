import pytest
import os
from q_learning_lab.adapter.intraday_market.training_interface import TrainingDataBundleParameter
import json
import random
import string

#random character in 10 length  
def random_choice(dim:int=10):
    return ''.join(random.choice(string.ascii_letters) for i in range(dim))

@pytest.fixture()
def get_TrainingDataBundleParameter() -> TrainingDataBundleParameter:
    bundle_param: TrainingDataBundleParameter = TrainingDataBundleParameter(
        input_data_dir=os.environ.get("DATA_DIR"),
        exchange="kraken",
        symbol="ETHUSD",
        start_date_ymd="20220401",
        end_date_ymd="20230401",
        data_length_days=10,
        data_step=1,
        split_ratio=0.9,
        output_data_dir=f"/tmp/output/{random_choice()}",
        candle_size_minutes=15,
    )
    return bundle_param

@pytest.fixture()
def get_feature_schema() -> dict[str, list]:
    with open("scripts/models/ohlcv_feature_schema.json", "r") as f:
        return json.load(f)
    

from q_learning_lab.adapter.intraday_market.data_input import (
    Data_Source,
    File_Data_Source_Factory
)

@pytest.fixture()
def get_training_eval_test_data_source(get_TrainingDataBundleParameter) -> tuple[Data_Source, Data_Source]:
    train_data_source, eval_data_source = File_Data_Source_Factory.prepare_training_eval_data_source(
        bundle_para=get_TrainingDataBundleParameter
    )
    return train_data_source, eval_data_source