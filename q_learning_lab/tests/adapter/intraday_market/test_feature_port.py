
from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
import numpy as np
from q_learning_lab.adapter.intraday_market.features.feature_port import (
    Feature_Generator_Factory,
    Feature_Generator_Enum,
    Feature_Generator_Interface,
)
from q_learning_lab.utility.logging import get_logger
from q_learning_lab.adapter.intraday_market.data_input import Data_Source
import pytest
import json

from q_learning_lab.utility.tools import timeit
from tests.adapter.intraday_market.test_environment import logger

logger = get_logger(__name__)
from crypto_feature_preprocess.port.features import Feature_Output



def test_feature_factory(get_feature_schema, get_training_eval_test_data_source):
    _training_data_source, _ = get_training_eval_test_data_source
    feature_schema_spec:dict[str, list] = get_feature_schema
    
    training_data_source:Data_Source = _training_data_source
    feature_schema:dict[str, Feature_Generator_Interface] = {}
    for field, detail_list in feature_schema_spec.items():
        feature_schema[field] = Feature_Generator_Factory.create_generator(
            generator_type=Feature_Generator_Enum.OHLCV, feature_list_input=detail_list)
        assert feature_schema[field] is not None
    feature_vectors_dict:dict[str, Feature_Output] = {}
    for field, feature_generator in feature_schema.items():
        ohlcv_candles = training_data_source.get_market_data_candles()
        feature_vectors_dict[field] = feature_generator.generate_feature(data_vector=ohlcv_candles[field])
        assert isinstance(feature_vectors_dict[field], Feature_Output)
    pass


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
    @timeit()
    def __calculate_feature(runner:FeatureRunner) -> Feature_Output:
        return runner.calculate_features()
    # Run the reset
    train_runner.reset()
    train_data = train_runner.get_current_market_data()
    train_feature:Feature_Output =__calculate_feature(train_runner)
    train_feature1_data_id = train_runner._data_source.data_id
    train_execution_time1 = __calculate_feature.execution_time[0]
    logger.info(f"train_execution_time1: {train_execution_time1} with data id: {train_runner._data_source.data_id}")
    eval_feature:Feature_Output = __calculate_feature(eval_runner)
    eval_execution_time = __calculate_feature.execution_time[0]

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
    assert (train_feature.time_index == train_runner.get_current_market_data().index[-train_row:]).all()

    train_feature2:Feature_Output =__calculate_feature(train_runner)
    train_data2 = train_runner.get_current_market_data()
    train_execution_time2 = __calculate_feature.execution_time[0]
    logger.info(f"train_execution_time2: {train_execution_time2} with data id: {train_runner._data_source.data_id}")

    assert train_execution_time1 > train_execution_time2, f"train_execution_time1: {train_execution_time1}, train_execution_time2: {train_execution_time2}"
    assert train_data.equals(train_data2)

    # Check the feature data is consistent with the original data source
    assert (train_feature.feature_data == train_feature2.feature_data).all()

    # Check feature generation after reset
    train_runner.reset()
    train_feature3_data_id = train_runner._data_source.data_id
    assert train_feature3_data_id != train_feature1_data_id
    train_feature3:Feature_Output =__calculate_feature(train_runner)

    train_execution_time3 = __calculate_feature.execution_time[0]
    logger.info(f"train_execution_time3: {train_execution_time3} with data id: {train_runner._data_source.data_id}")
    assert train_execution_time3 > train_execution_time2, f"train_execution_time3: {train_execution_time3}, train_execution_time2: {train_execution_time2}"
    if train_feature.feature_data.shape == train_feature3.feature_data.shape:
        assert np.not_equal(train_feature.feature_data, train_feature3.feature_data).any()
    pass