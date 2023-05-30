import json

from q_learning_lab.utility.logging import get_logger
from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
from q_learning_lab.adapter.intraday_market.environment import Intraday_Trade_Action_Space, Buy_Sell_Action_Enum
import pandas as pd
import numpy as np
import random

logger = get_logger(__name__)
def test_intraday_market_environment(
        get_feature_schema, 
        get_training_eval_test_data_source,
        get_intraday_config) -> None:
    random.seed(0)
    train_data_source, _ = get_training_eval_test_data_source
    train_runner:FeatureRunner = FeatureRunner(
        data_source=train_data_source,
        feature_generator_type="OHLCV",
        feature_plan=get_feature_schema,
    )

    intraday_config_dict:dict = get_intraday_config

    intraday_market_train_env: Intraday_Market_Environment = Intraday_Market_Environment()
    intraday_market_train_env.register_feature_runner(train_runner)
    intraday_market_train_env.register_pnl_calc_config(intraday_config_dict["pnl_config"])

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
    total_episode_length = intraday_market_train_env._current_data.shape[0]

    states, reward, done, truncated, info = intraday_market_train_env.step(action=Intraday_Trade_Action_Space.BUY)

    assert reward == -intraday_config_dict["pnl_config"]["fee_rate"]
    inx = info['inx']
    time_inx = info['time_inx']
    logger.critical(f"{type(time_inx)}")
    assert len(info['mtm_ratio']) == intraday_market_train_env.step_counter 
    assert len(info['long_trade_outstanding']) == 1, "only 1 trade in the first step"
    assert len(info['long_trade_archive'])==0
    assert done == False

    while not done:
        states, reward, done, truncated, info = intraday_market_train_env.step(action=Intraday_Trade_Action_Space.HOLD)
        
        assert len(info['mtm_ratio']) == intraday_market_train_env.step_counter 
        assert len(info['long_trade_outstanding']) == 1, "only 1 trade in the first step"
        assert len(info['long_trade_archive'])==0
        assert done == False, f"done == False at step {intraday_market_train_env.step_counter} of {total_episode_length}"
    
    