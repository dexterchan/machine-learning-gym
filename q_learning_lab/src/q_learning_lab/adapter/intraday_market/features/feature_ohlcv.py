from __future__ import annotations
from typing import NamedTuple, Tuple, List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np
from crypto_feature_preprocess.port.interfaces import (
    Feature_Definition,
    RSI_Feature_Interface,
    Log_Price_Feature_Interface,
    SMA_Cross_Feature_Interface,
    Feature_Enum,
)
from crypto_feature_preprocess.port.features import (
    create_feature_from_one_dim_data,
    create_feature_from_one_dim_data_v2,
    Feature_Output
)
from q_learning_lab.utility.tools import deprecated
from .feature_interface import Feature_Generator_Interface

class OHLCV_Feature_Generator(Feature_Generator_Interface):
    def __init__(self, feature_list_input:List[dict]) -> None:
        #Call helper function to parse List[dict] into List[Feature_Definition]
        self.feature_list:List[Feature_Definition] = self._parse_feature_list_input(feature_list_input)
    
    def _parse_feature_list_input(self, feature_list_input:List[dict]) -> List[Feature_Definition]:
        """ Helper function to parse List[dict] into List[Feature_Definition]
            dict format:
            {
                "feature_name": "feature_name",
                "feature_type": "feature_type",
                "feature_params": {
                    "param1": "param1",
                    "param2": "param2"
                    }
            }
        Args:
            feature_list_input (List[dict]): list of dict containing feature definition

        Returns:
            List[Feature_Definition]: List of structured Feature_Definition
        """

        feature_list:List[Feature_Definition] = []

        def __parse_feature_data(_type:Feature_Enum, _data:dict) -> NamedTuple:
            if _type == Feature_Enum.RSI:
                return RSI_Feature_Interface(**_data)
            elif _type == Feature_Enum.SMA_CROSS:
                return SMA_Cross_Feature_Interface(**_data)
            elif _type == Feature_Enum.LOG_PRICE:
                return Log_Price_Feature_Interface(**_data)
            else:
                raise Exception("Feature type not supported")

        for feature_dict in feature_list_input:
            feature_name:str = feature_dict["feature_name"]
            feature_type:Feature_Enum = Feature_Enum(feature_dict["feature_type"])
            feature_params:dict = feature_dict["feature_params"]
            feature_list.append(Feature_Definition(
                meta={
                    "name":feature_name,
                    "type" : feature_type
                    },
                data=__parse_feature_data(feature_type, feature_params)
            ))

        return feature_list
    

    @deprecated()
    def generate_feature_v1(self, price_vector:pd.Series) -> np.ndarray:
        """[Deprecated] generate feature vector from candle data following given feature input requirement

        Args:
            price_vector (pd.Series): Price vector e.g. close price, volume

        Returns:
            np.ndarray: feature array (N x feature_dim)
        """

        feature, _ = create_feature_from_one_dim_data(
            price_vector=price_vector,
            feature_list=self.feature_list
        )

        return feature
    
    def generate_feature(self, data_vector:pd.Series) -> Feature_Output:
        """Generate feature vector from candle data following given feature input requirement

        Args:
            data_vector (pd.Series): Price vector e.g. close price, volume

        Returns:
            Feature_Output: Feature output structured data
        """

        feature_output:Feature_Output = create_feature_from_one_dim_data_v2(
            data_vector=data_vector,
            feature_schema_list=self.feature_list
        )
        return feature_output