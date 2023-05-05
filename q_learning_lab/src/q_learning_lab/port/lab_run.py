from .environment import create_execute_environment

from ..domain.models.agent_params import Agent_Params
from ..domain.models.env_params import BaseEnv_Params
from ..domain.models.cart_pole_v1_models import get_dnn_structure
from ..domain.deep_q_learn import Reinforcement_DeepLearning
from ..utility.logging import get_logger

import os
import math
logger = get_logger(__name__)


def execute_lab_training(lab_name: str, lab_config: dict, is_verbose: bool) -> None:
    # Convert parameter dict to Cart_Pole_V1_Params

    _env_config_dict: dict = lab_config["env"]
    # Create the execute environment
    assert "n_max_steps" in _env_config_dict, "n_max_steps not found"
    # logger.error(f"execute with params: {_env_config_dict}")
    env = create_execute_environment(arena=lab_name, params=_env_config_dict)
    _agent_config: Agent_Params = Agent_Params(**lab_config["agent"])
    #Filter _env_config_dict to only include BaseEnv_Params
    _env_config:BaseEnv_Params = BaseEnv_Params(**{k: v for k, v in _env_config_dict.items() if k in BaseEnv_Params._fields})

    # Execute the lab training

    dnn_structure = get_dnn_structure(
        input_dim=env.observation_space_dim,
        output_dim=env.action_space_dim,
    )
    
    model_path:str = os.path.join(
            _agent_config.savemodel_folder, "training", f"{lab_name}-latest"
    )
    num_of_batch = math.ceil(int(_env_config.total_episodes) / int(_env_config.episode_batch))
    
    sub_env_config:BaseEnv_Params = BaseEnv_Params(
        total_episodes=_env_config.episode_batch,
        n_max_steps=_env_config.n_max_steps,
        episode_batch=_env_config.episode_batch,
    )
    for i in range(num_of_batch):
        logger.info(f"Batch {i+1} of {num_of_batch}")
        if i > 0:
            dnn_structure = model_path
        
        deepagent_dict = Reinforcement_DeepLearning.train(
            train_env=env,
            agent_params=_agent_config,
            train_env_params=sub_env_config,
            dnn_structure=dnn_structure,
            is_verbose=is_verbose,
        )
        deepagent_dict["main"].save_agent(
            path=model_path,
            episode=deepagent_dict["episode"],
            epsilon=deepagent_dict["epsilon"],
            total_rewards_history=deepagent_dict["total_rewards_history"],
        )
        

    return deepagent_dict
