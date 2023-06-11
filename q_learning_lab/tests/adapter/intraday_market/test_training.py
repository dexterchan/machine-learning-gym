from __future__ import annotations
from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
from q_learning_lab.domain.models.agent_params import Agent_Params
from q_learning_lab.domain.models.intraday_market_models import DNN_Params, EnvParams
from q_learning_lab.domain.deep_q_learn import Reinforcement_DeepLearning
import pytest
import os
from q_learning_lab.utility.logging import get_logger
from q_learning_lab.utility.process_runner import ForkProcessRunner
import math

logger = get_logger(__name__)

@pytest.mark.skipif("INTRADAY_TRAIN" not in os.environ,reason="not yet ready")
def test_training(get_intraday_config):
    intraday_config_dict:dict = get_intraday_config
    train_env:Intraday_Market_Environment = None
    eval_env:Intraday_Market_Environment = None

    # 1. Create training and evaluation environments
    train_env, eval_env = Intraday_Market_Environment.create_from_config(
        raw_data_config=intraday_config_dict["data_config"],
        feature_schema_config=intraday_config_dict["features"],
        pnl_config=intraday_config_dict["pnl_config"],
    )
    #2. Reset the environments
    train_env.reset()
    eval_env.reset()

    logger.info(f"train_env data dimension: {train_env.get_data_dimension()}")
    logger.info(f"eval_env data dimension: {eval_env.get_data_dimension()}")

    #3. create agent parameters - Agent_Params
    agent_params = Agent_Params(**intraday_config_dict["agent"])
    train_env_params = EnvParams(**intraday_config_dict["env"])

    #4. estimate the number of episodes batch required
    n_episodes_batches = math.ceil(int(train_env_params.total_episodes) / int(train_env_params.episode_batch))

    # For step 5 to step 6,
    # we fork the child process to execute the training
    # each fork process runs a batch of episodes
    # the parent process will wait for the child process to finish
    # then fork the next child process to execute the next batch of episodes
    # until all episodes are executed
    # the parent process will then exit
    # the child process will exit after the training is finished

     #fork start here
    def _fork_training_process()->None:
        #5. create DNN structure - DNN_Params
        dnn_params = DNN_Params(**intraday_config_dict["model_param"]["data"])
        
        #6. Use Reinforcement_DeepLearning.train to train the agent
        model_name = intraday_config_dict["model_param"]["meta"]["name"]
        deepagent_dict = Reinforcement_DeepLearning.train(
                train_env=train_env,
                agent_params=agent_params,
                train_env_params=train_env_params,
                dnn_structure=dnn_params.get_dnn_structure(),
                is_verbose=False,
                model_name=model_name,
                eval_env=eval_env
            )
        pass
    #fork end here

    fork_process_runner = ForkProcessRunner()
    for i in range(n_episodes_batches):
        fork_process_runner.fork_run(_fork_training_process)
    #wait until all child process finish

    pass