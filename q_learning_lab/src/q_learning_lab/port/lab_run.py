from .environment import create_execute_environment

from ..domain.models.cart_pole_v1_models import (
    Params as Cart_Pole_V1_Params,
)
from ..domain.models.cart_pole_v1_models import get_dnn_structure
from ..domain.deep_q_learn import Reinforcement_DeepLearning
from ..utility.logging import get_logger

logger = get_logger(__name__)


def execute_lab_training(lab_name: str, lab_config: dict, is_verbose: bool) -> None:
    # Convert parameter dict to Cart_Pole_V1_Params

    _lab_config = Cart_Pole_V1_Params(**lab_config)
    # Create the execute environment
    env = create_execute_environment(arena=lab_name, params=_lab_config)

    # Execute the lab training

    dnn_structure = get_dnn_structure(
        input_dim=env.observation_space_dim,
        output_dim=env.action_space_dim,
    )

    deepagent_dict = Reinforcement_DeepLearning.train(
        env=env,
        params=_lab_config,
        dnn_structure=dnn_structure,
        is_verbose=is_verbose,
    )

    return deepagent_dict
