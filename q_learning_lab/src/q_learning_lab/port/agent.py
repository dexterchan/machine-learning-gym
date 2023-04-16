from ..domain.q_learn import Agent
from ..domain.deep_q_learn import DeepAgent, SequentialStructure
from typing import NamedTuple
from pathlib import Path


def create_agent(params: NamedTuple, is_verbose: bool = False) -> Agent:
    return Agent(
        learning_rate=params.learning_rate,
        discount_rate=params.gamma,
        is_verbose=is_verbose,
    )


def create_new_deep_agent(
    params: NamedTuple, structure: SequentialStructure, is_verbose: bool = False
) -> DeepAgent:
    return DeepAgent(
        learning_rate=params.learning_rate,
        discount_factor=params.gamma,
        is_verbose=is_verbose,
        structure=structure,
    )

def load_saved_deep_agent(model_path:str) -> DeepAgent:
    """
    Load a saved agent from the specified path.

    """
    _model_path = model_path
    if isinstance(model_path, Path):
        _model_path = str(model_path)
    agent = DeepAgent.load_agent(path=_model_path)

    return agent


