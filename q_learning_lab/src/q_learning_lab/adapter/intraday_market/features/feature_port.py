from enum import Enum
from typing import List
class Feature_Generator_Enum(str, Enum):
    OHLCV = "OHLCV"

from .feature_ohlcv import OHLCV_Feature_Generator
from .feature_interface import Feature_Generator_Interface

class Feature_Generator_Factory():
    @staticmethod
    def create_generator(generator_type:Feature_Generator_Enum, feature_list_input:List[dict]) -> Feature_Generator_Interface:
        if generator_type == Feature_Generator_Enum.OHLCV:
            return OHLCV_Feature_Generator(feature_list_input)
        else:
            raise Exception("Feature generator type not supported")
