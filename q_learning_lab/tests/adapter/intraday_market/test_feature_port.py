
from q_learning_lab.adapter.intraday_market.features.feature_port import (
    Feature_Generator_Factory,
    Feature_Generator_Enum,
    Feature_Generator_Interface,
)
from q_learning_lab.utility.logging import get_logger
import pytest
import json

logger = get_logger(__name__)

@pytest.fixture()
def get_feature_schema() -> dict[str, list]:
    with open("scripts/models/ohlcv_feature_schema.json", "r") as f:
        return json.load(f)


def test_feature_factory(get_feature_schema):

    feature_schema_spec:dict[str, list] = get_feature_schema
    
    feature_schema:dict[str, Feature_Generator_Interface] = {}
    for field, detail_list in feature_schema_spec.items():
        feature_schema[field] = Feature_Generator_Factory.create_generator(
            generator_type=Feature_Generator_Enum.OHLCV, feature_list_input=detail_list)
        assert feature_schema[field] is not None
    pass