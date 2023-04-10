from ..domain.q_learn import Agent
from typing import NamedTuple
from .environment import Execute_Environment


def create_agent(params: NamedTuple, is_verbose: bool = False) -> Agent:
    return Agent(
        learning_rate=params.learning_rate,
        discount_rate=params.gamma,
        is_verbose=is_verbose,
    )
