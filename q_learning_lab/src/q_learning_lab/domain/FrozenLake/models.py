from __future__ import annotations
from enum import Enum
from typing import Any


class State:
    def __init__(self) -> None:
        pass


class Action_Space(int, Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class Policy:
    def __init__(self) -> None:
        pass

    def act(self, observation: Any) -> Action_Space:
        raise NotImplementedError("Policy not implemented")


class Reward:
    def __init__(self) -> None:
        pass

    def get_reward(self, state: State) -> float:
        raise NotImplementedError("Reward not implemented")
