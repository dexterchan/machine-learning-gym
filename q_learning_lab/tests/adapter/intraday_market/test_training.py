from __future__ import annotations
from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.domain.models.agent_params import Agent_Params
from q_learning_lab.domain.models.intraday_market_models import DNN_Params, EnvParams
from q_learning_lab.domain.deep_q_learn import Reinforcement_DeepLearning
import pytest

@pytest.mark.skip(reason="Not implemented")
def test_training(get_intraday_config):
    intraday_config_dict:dict = get_intraday_config
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
    agent_params = Agent_Params(**intraday_config_dict["agent"])
    train_env_params = EnvParams(**intraday_config_dict["env"])

    #4. create DNN structure - DNN_Params
    dnn_params = DNN_Params(**intraday_config_dict["model_param"]["data"])
    
    #5. Use Reinforcement_DeepLearning.train to train the agent
    model_name = intraday_config_dict["model_param"]["meta"]["name"]
    deepagent_dict = Reinforcement_DeepLearning.train(
            train_env=train_env,
            agent_params=dnn_params,
            train_env_params=train_env_params,
            dnn_structure=dnn_params.get_dnn_structure(),
            is_verbose=False,
            model_name=model_name,
            eval_env=eval_env
        )


    pass