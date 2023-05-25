
from q_learning_lab.adapter.intraday_market.features.feature_port import (
    Feature_Generator_Factory,
    Feature_Generator_Enum,
    Feature_Generator_Interface,
)
from q_learning_lab.utility.logging import get_logger
from q_learning_lab.adapter.intraday_market.data_input import Data_Source
import pytest
import json

logger = get_logger(__name__)




def test_feature_factory(get_feature_schema, get_training_eval_test_data_source):
    _training_data_source, _ = get_training_eval_test_data_source
    feature_schema_spec:dict[str, list] = get_feature_schema
    
    training_data_source:Data_Source = _training_data_source
    feature_schema:dict[str, Feature_Generator_Interface] = {}
    for field, detail_list in feature_schema_spec.items():
        feature_schema[field] = Feature_Generator_Factory.create_generator(
            generator_type=Feature_Generator_Enum.OHLCV, feature_list_input=detail_list)
        assert feature_schema[field] is not None
    feature_vectors_dict:dict[str, list] = {}
    for field, feature_generator in feature_schema.items():
        ohlcv_candles = training_data_source.get_market_data_candles()
        feature_vectors_dict[field] = feature_generator.generate_feature(price_vector=ohlcv_candles[field])
    pass