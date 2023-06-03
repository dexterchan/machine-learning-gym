import json

from q_learning_lab.utility.logging import get_logger
from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.adapter.intraday_market.environment import FeatureRunner
from q_learning_lab.adapter.intraday_market.environment import Intraday_Trade_Action_Space, Buy_Sell_Action_Enum
from crypto_feature_preprocess.port.features import Feature_Output
from tradesignal_mtm_runner.models import ProxyTrade, Proxy_Trade_Actions
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
    
    assert len(intraday_market_train_env._trade_order_agent.mtm_history) > 0
    assert len(intraday_market_train_env._trade_order_agent.mtm_history) == intraday_market_train_env.step_counter 
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
    _mtm = intraday_market_train_env._trade_order_agent.mtm_history
    _mtm_filtered = [ x for x in _mtm if (x > 0) or (x < 0)]
    logger.info("MtM sum list =  %s", np.sum(_mtm_filtered))
    logger.info("MtM sum =  %s", np.sum(intraday_market_train_env._trade_order_agent.mtm_history))

    logger.info("Trade MTM = %s ", (trade.exit_price - trade.entry_price)/trade.entry_price - intraday_config_dict["pnl_config"]["fee_rate"]*2)
    logger.info("Trade MTM = %s ", trade.calculate_pnl_normalized(price=trade.exit_price,fee_included=True))
    logger.info(intraday_config_dict)
    
        
    
    