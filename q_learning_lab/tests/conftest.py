import pytest
import os
from q_learning_lab.adapter.intraday_market.training_interface import TrainingDataBundleParameter

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
        output_data_dir="/tmp/output",
        candle_size_minutes=15,
    )
    return bundle_param