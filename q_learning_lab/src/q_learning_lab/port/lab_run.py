from .environment import create_execute_environment

from ..domain.models.agent_params import Agent_Params
from ..domain.models.cart_pole_v1_models import get_dnn_structure
from ..domain.deep_q_learn import Reinforcement_DeepLearning
from ..utility.logging import get_logger

logger = get_logger(__name__)


def execute_lab_training(lab_name: str, lab_config: dict, is_verbose: bool) -> None:
    # Convert parameter dict to Cart_Pole_V1_Params

    _env_config: dict = lab_config["env"]
    _agent_config: dict = Agent_Params(**lab_config["agent"])
    # Create the execute environment
    assert "n_max_steps" in _env_config, "n_max_steps not found"
    # logger.error(f"execute with params: {_env_config}")
    env = create_execute_environment(arena=lab_name, params=_env_config)

    # Execute the lab training

    dnn_structure = get_dnn_structure(
        input_dim=env.observation_space_dim,
        output_dim=env.action_space_dim,
    )

    deepagent_dict = Reinforcement_DeepLearning.train(
        train_env=env,
        agent_params=_agent_config,
        train_env_params=env.env_params,
        dnn_structure=dnn_structure,
        is_verbose=is_verbose,
    )

    return deepagent_dict
