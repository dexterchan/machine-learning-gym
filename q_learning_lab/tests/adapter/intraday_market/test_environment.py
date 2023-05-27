from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
from crypto_feature_preprocess.port.features import Feature_Output
import numpy as np
import json
# from q_learning_lab.adapter.intraday_market.data_input import (
#     Data_Source,
#     File_Data_Source_Factory,
#     Random_File_Access_Data_Source
# )
# test FeatureRunner
# Path: tests/adapter/intraday_market/test_environment.py

def test_feature_runner(get_feature_schema, get_training_eval_test_data_source) -> None:

    train_data_source, eval_data_source = get_training_eval_test_data_source
    train_runner:FeatureRunner = FeatureRunner(
        data_source=train_data_source,
        feature_generator_type="OHLCV",
        feature_plan=get_feature_schema,
    )
    eval_runner:FeatureRunner = FeatureRunner(
        data_source=eval_data_source,
        feature_generator_type="OHLCV",
        feature_plan=get_feature_schema,
    )

    # Run the reset
    train_runner.reset()
    train_feature:Feature_Output = train_runner.calculate_features()
    eval_feature:Feature_Output = eval_runner.calculate_features()
    assert train_feature.feature_data is not None
    train_row, train_col = train_feature.feature_data.shape
    eval_row, eval_col = eval_feature.feature_data.shape
    assert train_col == train_runner.feature_observation_space_dim == eval_col == eval_runner.feature_observation_space_dim
    assert train_row > 0
    assert eval_row > 0
    #CHeck the time index
    assert train_feature.time_index is not None
    assert eval_feature.time_index is not None
    # Check train_feature time_index consistent with original data source
    assert (train_feature.time_index == train_runner._data_source.get_market_data_candles().index[-train_row:]).all()

    pass