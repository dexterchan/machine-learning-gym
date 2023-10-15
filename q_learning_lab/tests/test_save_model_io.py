from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.domain.models.intraday_market_models import DNN_Params, EnvParams

from q_learning_lab.domain.models.agent_params import Agent_Params
from q_learning_lab.domain.models.env_params import BaseEnv_Params
from q_learning_lab.domain.deep_q_learn import Reinforcement_DeepLearning
from q_learning_lab.domain.deep_q_learn import DeepAgent

import pytest
import os
import logging

from urllib.parse import urlparse

#@pytest.mark.skipif(os.environ.get("CI") == "true", reason="CI does not have the data")
#@pytest.mark.skip(reason="skip for now")
def test_saving_model_local_folder(get_intraday_local_config):
    intraday_config_dict:dict = get_intraday_local_config
    run_id:str="test_training"


    # 1. Create training and evaluation environments
    train_env, eval_env = Intraday_Market_Environment.create_from_config(
        raw_data_config=intraday_config_dict["data_config"],
        feature_schema_config=intraday_config_dict["features"],
        pnl_config=intraday_config_dict["pnl_config"],
    )

    #2. Reset the environments
    train_env.reset()
    eval_env.reset()

    #3. create agent parameters - Agent_Params
    dnn_params = DNN_Params(**intraday_config_dict["model_param"]["data"])
    model_struct = dnn_params.get_dnn_structure()
    agent_params = Agent_Params(**intraday_config_dict["agent"])
    train_env_params = EnvParams(**intraday_config_dict["env"])

    model_name = intraday_config_dict["model_param"]["meta"]["name"]

    model_path: str = Reinforcement_DeepLearning.create_model_path_root(
            agent_params=agent_params, model_name=model_name, run_id=run_id
        )
    
    learning_rate = agent_params.learning_rate
    discount_factor = agent_params.gamma
    
    main = Reinforcement_DeepLearning.create_new_deep_agent(
                dnn_structure=model_struct,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                is_verbose=False,
            )
    episode:int = 1
    epsilon:float = agent_params.epsilon
    best_measure:float = 0
    main.save_agent(
                    path=f"{model_path}_{episode}",
                    episode=episode,
                    epsilon=epsilon,
                    best_measure=best_measure,
                    total_rewards_history=[0.1,0,2],
                    eval_rewards_history=[0.3,0,1]
                )
    logging.info(f"model saved to {model_path}_{episode}/MODEL_DUMP.zip")
    assert os.path.exists(f"{model_path}_{episode}/MODEL_DUMP.zip")
    
    assert DeepAgent.check_agent_loadable_from_path(path=f"{model_path}_{episode}")
    
    load = Reinforcement_DeepLearning.create_new_deep_agent(
                dnn_structure=model_struct,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                is_verbose=False,
            )
    agent, para = load.load_agent(path=f"{model_path}_{episode}")
    assert agent is not None
    assert para is not None
    assert para["total_rewards_history"] == [0.1,0,2]
    assert para["eval_rewards_history"] == [0.3,0,1]


    pass

#@pytest.mark.skip(reason="skip for now")
def test_saving_model_s3_folder(get_intraday_s3_config):
    intraday_config_dict:dict = get_intraday_s3_config
    run_id:str="test_training"


    # 1. Create training and evaluation environments
    train_env, eval_env = Intraday_Market_Environment.create_from_config(
        raw_data_config=intraday_config_dict["data_config"],
        feature_schema_config=intraday_config_dict["features"],
        pnl_config=intraday_config_dict["pnl_config"],
    )

    #2. Reset the environments
    train_env.reset()
    eval_env.reset()

    #3. create agent parameters - Agent_Params
    dnn_params = DNN_Params(**intraday_config_dict["model_param"]["data"])
    model_struct = dnn_params.get_dnn_structure()
    agent_params = Agent_Params(**intraday_config_dict["agent"])
    train_env_params = EnvParams(**intraday_config_dict["env"])

    model_name = intraday_config_dict["model_param"]["meta"]["name"]

    model_path: str = Reinforcement_DeepLearning.create_model_path_root(
            agent_params=agent_params, model_name=model_name, run_id=run_id
        )
    
    learning_rate = agent_params.learning_rate
    discount_factor = agent_params.gamma
    
    main = Reinforcement_DeepLearning.create_new_deep_agent(
                dnn_structure=model_struct,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                is_verbose=False,
            )
    episode:int = 1
    epsilon:float = agent_params.epsilon
    best_measure:float = 0
    main.save_agent(
                    path=f"{model_path}_{episode}",
                    episode=episode,
                    epsilon=epsilon,
                    best_measure=best_measure,
                    total_rewards_history=[0.1,0,2,0.3],
                    eval_rewards_history=[0.3,0,2]
                )
    logging.info(f"model saved to {model_path}_{episode}/MODEL_DUMP.zip")
    #assert os.path.exists(f"{model_path}_{episode}/MODEL_DUMP.zip")
    assert DeepAgent.check_agent_loadable_from_path(path=f"{model_path}_{episode}")
    assert not DeepAgent.check_agent_loadable_from_path(path=f"{model_path}_{episode}.notexist")

    load = Reinforcement_DeepLearning.create_new_deep_agent(
                dnn_structure=model_struct,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                is_verbose=False,
            )
    agent, para = load.load_agent(path=f"{model_path}_{episode}")
    assert agent is not None
    assert para is not None
    assert para["total_rewards_history"] == [0.1,0,2,0.3]
    assert para["eval_rewards_history"] == [0.3,0,2]

    pass

@pytest.mark.skip(reason="skip for now")
def test_save_to_s3():
    import boto3
    s3_client = boto3.client('s3')
    path = "s3://boar-tradingbot/q-learning/agent_para/model/test_training/simple_dnn_1/MODEL_DUMP.zip"
    zip_file_path = "/Users/dexter/sandbox/algo_trading/gym/q_learning_lab/_static/model/test/test_training/simple_dnn_1/MODEL_DUMP.zip"
    o = urlparse(path, allow_fragments=False)
    logging.info(f"Upload to S3 bucket: {o.netloc} path: {os.path.join(o.path,os.path.basename(zip_file_path))}")
                # s3_client.put_object(
                #     Bucket=o.netloc,
                #     Key=os.path.join(o.path,os.path.basename(zip_file_path)),
                #     Body=zip_file_path
                #     #request_payer='requester'
                # )
    override = "/Users/dexter/sandbox/algo_trading/gym/q_learning_lab/_static/model/test/test_training/simple_dnn_1/MODEL_DUMP.zip"
    with open(override, "rb") as f:
            s3_client.upload_fileobj(f, o.netloc, os.path.join(o.path[1:],os.path.basename(zip_file_path)))

    pass