import pandas as pd
import numpy as np
from abc import abstractmethod

class Feature_Generator_Interface:
     @abstractmethod
     def generate_feature(self, price_vector:pd.Series) -> np.ndarray:
          pass