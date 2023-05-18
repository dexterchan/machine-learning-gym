from typing import NamedTuple

class TrainingDataBundleParameter(NamedTuple):
    input_data_dir:str
    exchange:str
    symbol:str
    start_date_ymd:str
    end_date_ymd:str
    data_length_days:int
    data_step:int
    split_ratio:float
    output_data_dir:str
    candle_size_minutes:int