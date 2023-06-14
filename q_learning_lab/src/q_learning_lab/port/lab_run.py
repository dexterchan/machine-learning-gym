from .environment import create_execute_environment

from ..domain.models.agent_params import Agent_Params
from ..domain.models.env_params import BaseEnv_Params

from ..domain.deep_q_learn import Reinforcement_DeepLearning
from ..utility.logging import get_logger
from typing import Any
import os
import math

from ..utility.process_runner import ForkProcessRunner

logger = get_logger(__name__)


def create_train_materials(lab_name:str, lab_config: dict) -> tuple:
    """
        Create the training materials for the lab.
        Args:
            lab_name: The name of the lab.
            lab_config: The lab configuration dictionary

        Returns:
            dnn_structure: The DNN structure to use for the lab training.
            train_env (Execute_Environment): The training environment.
            eval_env (Execute_Environment): The evaluation environment.
            _env_config (BaseEnv_Params): The environment configuration parameters.
    """
    _env_config_dict: dict = lab_config["env"]
    _env_config:BaseEnv_Params = BaseEnv_Params(**{k: v for k, v in _env_config_dict.items() if k in BaseEnv_Params._fields})

    if lab_name == "cartpole-v1":
        logger.info("Creating training materials for cartpole-v1")
        from ..domain.models.cart_pole_v1_models import get_dnn_structure
        
        assert "n_max_steps" in _env_config_dict, "n_max_steps not found"
        # Create the training environment
        train_env = create_execute_environment(arena=lab_name, params=_env_config_dict)
        # Create the eval environment
        eval_env = create_execute_environment(arena=lab_name, params=_env_config_dict)

        _agent_config: Agent_Params = Agent_Params(**lab_config["agent"])
        # Create the DNN structure
        dnn_structure = get_dnn_structure(
            input_dim=train_env.observation_space_dim,
            output_dim=train_env.action_space_dim,
        )
        return dnn_structure, train_env, eval_env, _env_config, _agent_config
    elif lab_name == "intraday-market-v0":
        logger.info("Preparing intraday-market-v0 training materials")
        intraday_config_dict:dict = lab_config
        from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
        from q_learning_lab.domain.models.intraday_market_models import DNN_Params, EnvParams
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
        dnn_structure=dnn_params.get_dnn_structure()

        return dnn_structure, train_env, eval_env, train_env_params, agent_params

    else:
        raise NotImplementedError(f"lab_name: {lab_name} not implemented")  

def execute_lab_training(lab_name: str, lab_config: dict, is_verbose: bool, force_new:bool, run_id:str) -> None:
    """
        Execute the lab training.
        Args:
            lab_name: The name of the lab.
            lab_config: The lab configuration dictionary
            is_verbose: Whether to print the training progress.
            force_new: Whether to force a new training.
            run_id: "run id" of this trianing
        
        Returns:
            None
    """
    from q_learning_lab.adapter.intraday_market.environment import Intraday_Market_Environment
    from q_learning_lab.domain.models.intraday_market_models import DNN_Params, EnvParams
    intraday_config_dict:dict = lab_config
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

    #4. estimate the number of episodes batch required
    n_episode_batches = math.ceil(int(train_env_params.total_episodes) / int(train_env_params.episode_batch))

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
        model_struct = dnn_params.get_dnn_structure()
        model_name = intraday_config_dict["model_param"]["meta"]["name"]
        
        if not force_new:
            #Construct the model path is loadable
            model_path = os.path.join(agent_params.savemodel_folder, run_id, f"{model_name}_latest")
            logger.info(f"Trying to load latest model from {model_path}")
            #Check if the model path exists
            if Reinforcement_DeepLearning.check_agent_reloadable(model_path=model_path):
                logger.info(f"Ready to continue the training from {model_path}")
                model_struct = model_path
        
        #6. Use Reinforcement_DeepLearning.train to train the agent
        _ = Reinforcement_DeepLearning.train(
                train_env=train_env,
                agent_params=agent_params,
                train_env_params=train_env_params,
                dnn_structure=model_struct,
                is_verbose=False,
                model_name=model_name,
                eval_env=eval_env,
                run_id=run_id
            )
        pass
    #fork end here
    fork_process_runner = ForkProcessRunner()
    for i in range(n_episode_batches):
        logger.info(f"Execute fork batch {i} start")
        fork_process_runner.fork_run(_fork_training_process)
        logger.info(f"Execute fork batch {i} end")
    #wait until all child process finish
    
    pass


def execute_lab_training_old(lab_name: str, lab_config: dict, is_verbose: bool) -> None:
    # Convert parameter dict to Cart_Pole_V1_Params

    dnn_structure, train_env, eval_env, baseEnv_Params, _agent_config = create_train_materials(lab_name=lab_name, lab_config=lab_config)
    
    # Create the model path
    model_path:str = os.path.join(
            _agent_config.savemodel_folder, "training", f"{lab_name}-latest"
    )
    num_of_batch = math.ceil(int(baseEnv_Params.total_episodes) / int(baseEnv_Params.episode_batch))
    
    sub_env_config:BaseEnv_Params = BaseEnv_Params(
        total_episodes=baseEnv_Params.episode_batch,
        n_max_steps=baseEnv_Params.n_max_steps,
        episode_batch=baseEnv_Params.episode_batch,
    )
    for i in range(num_of_batch):
        logger.info(f"Batch {i+1} of {num_of_batch}")
        if i > 0:
            dnn_structure = model_path
        
        deepagent_dict = Reinforcement_DeepLearning.train(
            train_env=train_env,
            eval_env=eval_env,
            agent_params=_agent_config,
            train_env_params=sub_env_config,
            dnn_structure=dnn_structure,
            is_verbose=is_verbose,
            model_name=lab_name,
            run_id="training"
        )
        deepagent_dict["main"].save_agent(
            path=model_path,
            episode=deepagent_dict["episode"],
            epsilon=deepagent_dict["epsilon"],
            total_rewards_history=deepagent_dict["total_rewards_history"],
            eval_rewards_history=deepagent_dict["eval_rewards_history"],
        )
        

    return deepagent_dict
