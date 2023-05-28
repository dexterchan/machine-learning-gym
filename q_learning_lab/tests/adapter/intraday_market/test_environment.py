import json

from q_learning_lab.utility.logging import get_logger
from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
from q_learning_lab.adapter.intraday_market.environment import Intraday_Trade_Action_Space, Buy_Sell_Action_Enum
import pandas as pd
import numpy as np
logger = get_logger(__name__)
def test_intraday_market_environment(
        get_feature_schema, 
        get_training_eval_test_data_source,
        get_intraday_config) -> None:
    
    train_data_source, _ = get_training_eval_test_data_source
    train_runner:FeatureRunner = FeatureRunner(
        data_source=train_data_source,
        feature_generator_type="OHLCV",
        feature_plan=get_feature_schema,
    )

    intraday_config_dict:dict = get_intraday_config

    intraday_market_train_env: Intraday_Market_Environment = Intraday_Market_Environment()
    intraday_market_train_env.register_feature_runner(train_runner)
    intraday_market_train_env.register_reward_generator(intraday_config_dict["pnl_config"])

    observation, time_inx = intraday_market_train_env.reset()
    logger.info(f"observation: {observation}")
    # logger.info(f"time_inx: {time_inx}")
    # logger.info(f"time_inx: {type(time_inx)}")

    ##time_inx_timestamp = (pd.to_numeric(time_inx) / 1000000).astype("int64")
    try:
        intraday_market_train_env.step(action=Buy_Sell_Action_Enum.BUY)
        assert False, "should be invalid action"
    except ValueError as e:
        pass
    pass

    #print(intraday_market_train_env._current_data)

    states, reward, done, truncated, info = intraday_market_train_env.step(action=Intraday_Trade_Action_Space.BUY)
    inx = info['inx']
    time_inx = info['time_inx']
    #print(intraday_market_train_env._current_data)
    _data = intraday_market_train_env._current_data

    logger.critical(_data[:time_inx])
    #print(_data[_data["buy"]!=0])

    logger.info(f"states: {states}")
    logger.info(f"reward: {reward}")
    logger.info(f"done: {done}")
    logger.info(f"truncated: {truncated}")
    logger.info(f"mtm: {np.sum(info['mtm_ratio'])}")
    logger.info(f"time_inx: {info['time_inx']}")
    logger.info(f"inx: {info['inx']}")
    logger.info(f"long_trade_outstanding: {info['long_trade_outstanding']}")
    logger.info(f"long_trade_archive: {info['long_trade_archive']}")
    