import json
import re
from q_learning_lab.utility.logging import get_logger
from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
from q_learning_lab.adapter.intraday_market.environment import Intraday_Trade_Action_Space, Buy_Sell_Action_Enum
from crypto_feature_preprocess.port.features import Feature_Output
from tradesignal_mtm_runner.models import ProxyTrade, Proxy_Trade_Actions
import pandas as pd
import numpy as np
import random
import pytest
import os

logger = get_logger(__name__)

def test_regex_value(get_intraday_local_config):
    intraday_config_dict:dict = get_intraday_local_config
    Intraday_Market_Environment._regex_replace_with_env_variables(
        intraday_config_dict["data_config"],
    )
    regex_processed_value = intraday_config_dict["data_config"]["input_data_dir"]
    assert regex_processed_value == os.environ["DATA_DIR"]
    
    command_var_regex = re.compile(r"\$\((\w+)\)")
    value = "/tmp/output/batch_run/$(RANDOM)"
    m = command_var_regex.search(value)
    assert m is not None
    assert m[1] == "RANDOM"
    abc = command_var_regex.sub("newvalue", value)
    assert abc == "/tmp/output/batch_run/newvalue"

    intraday_config_dict:dict = get_intraday_local_config
    Intraday_Market_Environment._regex_sub_with_command(
        intraday_config_dict["data_config"],
    )
    regex_processed_value = intraday_config_dict["data_config"]["output_data_dir"]
    logger.info(regex_processed_value)
    match_regex = re.compile(r"^/tmp/output/batch_run/([A-Z0-9]+)$") #([A-Z0-9]{10})$
    m = match_regex.match(regex_processed_value)
    assert m is not None
    logger.info(m[1])
    



def test_intraday_market_environment(
        get_intraday_local_config) -> None:
    random.seed(0)
    
    intraday_config_dict:dict = get_intraday_local_config
    intraday_market_train_env, _ = Intraday_Market_Environment.create_from_config(
        raw_data_config=intraday_config_dict["data_config"],
        feature_schema_config={
            "OHLCV":{
                "volume": [
                {
                    "feature_name": "Log Volume movement",
                    "feature_type": "LOG_PRICE",
                    "feature_params": {
                        "dimension": 10,
                        "normalize_value": 5
                    }
                }
            ]
            }
        },
        pnl_config=intraday_config_dict["pnl_config"],
    )

    # intraday_market_train_env: Intraday_Market_Environment = Intraday_Market_Environment()
    # intraday_market_train_env.register_feature_runner(train_runner)
    # intraday_market_train_env.register_pnl_calc_config(intraday_config_dict["pnl_config"])
    data_id1:int = intraday_market_train_env._feature_runner.episode_id

    observation, time_inx = intraday_market_train_env.reset()
    logger.debug(f"observation: {observation}")
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
    max_step = intraday_market_train_env._feature_runner.max_step
    f_out:Feature_Output = intraday_market_train_env._feature_runner.calculate_features()

    assert f_out.feature_data.shape[0] < total_episode_length
    

    states, reward, done, truncated, _ = intraday_market_train_env.step(action=Intraday_Trade_Action_Space.BUY)

    outstanding_long_position_list:list[ProxyTrade] = intraday_market_train_env._trade_order_agent.outstanding_long_position_list
    archive_long_position_list:list[ProxyTrade] = intraday_market_train_env._trade_order_agent.archive_long_positions_list
    
    assert reward == -intraday_config_dict["pnl_config"]["fee_rate"]
    # inx = info['inx']
    # time_inx = info['time_inx']
    
    assert len(intraday_market_train_env._trade_order_agent.mtm_history_value) > 0
    assert len(intraday_market_train_env._trade_order_agent.mtm_history_value) == intraday_market_train_env.step_counter 
    assert len(outstanding_long_position_list) == 1, "only 1 trade in the first step"
    assert len(archive_long_position_list)==0
    assert done == False

    current_data = intraday_market_train_env._current_data
    logger.info(f"start:{current_data.index[0]} to end: {current_data.index[-1]}")
    logger.info(f"total episode length: {total_episode_length}")
    logger.info(f"step_counter: {intraday_market_train_env._feature_runner.read_pointer} of {max_step}")

    monitoring = False
    assert len(outstanding_long_position_list) == 1
    logger.debug(f"outstanding_long_position_list: {outstanding_long_position_list}")
    while intraday_market_train_env._feature_runner.read_pointer<max_step:
        states, reward, done, truncated, info = intraday_market_train_env.step(action=Intraday_Trade_Action_Space.HOLD)
        if len(archive_long_position_list)>0:
            logger.debug("spot trade closed")
            monitoring = True
        if monitoring and len(outstanding_long_position_list)>0:
            logger.critical("trade reopen again")
            logger.critical(f"outstanding_long_position_list: {outstanding_long_position_list}")
            assert False, "trade issue"
            break

        #logger.info(f"step_counter: {intraday_market_train_env._feature_runner.read_pointer} of {num_of_features}")

    assert done == True
    assert len(intraday_market_train_env._trade_order_agent.outstanding_long_position_list) == 0, "all trade should be done"
    assert len(archive_long_position_list) == 1, "only 1 trade should be done"

    #Reset the environment and start another epsiode
    observation, time_inx = intraday_market_train_env.reset()
    logger.debug(f"observation: {observation}")
    assert intraday_market_train_env._feature_runner.read_pointer == 0
    assert len(intraday_market_train_env._trade_order_agent.outstanding_long_position_list) == 0
    assert len(intraday_market_train_env._trade_order_agent.archive_long_positions_list) == 0
    data_id2:int = intraday_market_train_env._feature_runner.episode_id

    assert data_id1 != data_id2, "should be different episode id"
    run_continue = True

    buy_pos = max_step // 3
    sell_pos = buy_pos + 1

    while run_continue:
        _action:Intraday_Trade_Action_Space = Intraday_Trade_Action_Space.HOLD
        if intraday_market_train_env._feature_runner.read_pointer == buy_pos:
            _action = Intraday_Trade_Action_Space.BUY
        elif intraday_market_train_env._feature_runner.read_pointer == sell_pos:
            _action = Intraday_Trade_Action_Space.SELL

        states, reward, done, truncated, info = intraday_market_train_env.step(action=_action)
        if done:
            run_continue = False
    assert len(intraday_market_train_env._trade_order_agent.outstanding_long_position_list) == 0
    assert len(intraday_market_train_env._trade_order_agent.archive_long_positions_list) == 1
    trade:ProxyTrade = intraday_market_train_env._trade_order_agent.archive_long_positions_list[0]
    assert trade.close_reason == Proxy_Trade_Actions.SIGNAL
    logger.info(trade)
    _mtm = intraday_market_train_env._trade_order_agent.mtm_history_value
    _mtm_filtered = [ x for x in _mtm if (x > 0) or (x < 0)]
    logger.info("MtM sum list =  %s", np.sum(_mtm_filtered))
    logger.info("MtM sum =  %s", np.sum(intraday_market_train_env._trade_order_agent.mtm_history_value))

    logger.info("Trade MTM = %s ", (trade.exit_price - trade.entry_price)/trade.entry_price - intraday_config_dict["pnl_config"]["fee_rate"]*2)
    logger.info("Trade MTM = %s ", trade.calculate_pnl_normalized(price=trade.exit_price,fee_included=True))
    logger.info(intraday_config_dict)
    assert abs(trade.calculate_pnl_normalized(trade.exit_price,True) - np.sum(intraday_market_train_env._trade_order_agent.mtm_history_value)) < 0.1
    
    pass


def test_intraday_market_train_env_with_class_function_iterator(get_intraday_local_config) -> None:
    intraday_config_dict:dict = get_intraday_local_config
    intraday_market_train_env, _ = Intraday_Market_Environment.create_from_config(
        raw_data_config=intraday_config_dict["data_config"],
        feature_schema_config=intraday_config_dict["features"],
        pnl_config=intraday_config_dict["pnl_config"]
    )
    #Iterate the intraday market train env
    intraday_market_train_env_itr = iter(intraday_market_train_env)
    #feature_itr = iter(intraday_market_train_env._feature_runner)
    data_set = set()
    cnt = 0
    max_counter = -1
    for cnt, env in enumerate(intraday_market_train_env_itr):
        env.reset()
        runner = env._feature_runner

        #logger.info(f"Feature runner {cnt}: {runner._data_source.data_id}")
        assert runner._read_pointer == 0
        _data_id = int(runner._data_source.data_id)
        max_counter = _data_id if _data_id > max_counter else max_counter
        assert _data_id not in data_set, "data source counter not consistent"
        data_set.add(_data_id)
        #assert cnt == int(env._feature_runner._data_source.data_id), "data source counter not consistent"
        pass
    assert len(data_set) == cnt+1


    pass