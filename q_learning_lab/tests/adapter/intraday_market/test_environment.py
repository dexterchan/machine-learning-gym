from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
from q_learning_lab.adapter.intraday_market.data_input import (
    Data_Source,
    File_Data_Source_Factory,
    Random_File_Access_Data_Source
)
# test FeatureRunner
# Path: tests/adapter/intraday_market/test_environment.py

def test_feature_runner(get_feature_schema, get_TrainingDataBundleParameter) -> None:

    train_data_source, eval_data_source = File_Data_Source_Factory.prepare_training_eval_data_source(
        bundle_para=get_TrainingDataBundleParameter
    )
    FeatureRunner(
        data_source=train_data_source,
        feature_generator_type="OHLCV",
        feature_plan=get_feature_schema,
    )
    pass