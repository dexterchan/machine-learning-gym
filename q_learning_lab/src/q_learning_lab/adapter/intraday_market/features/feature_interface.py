import pandas as pd
import numpy as np
from abc import abstractmethod
from crypto_feature_preprocess.port.features import Feature_Output

class Feature_Generator_Interface:
     @abstractmethod
     def generate_feature(self, data_vector:pd.Series) -> Feature_Output:
          pass