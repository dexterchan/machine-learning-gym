from __future__ import annotations
from enum import Enum


class EnvParams(NamedTuple):
    total_episodes: int  # Total episodes
    n_max_steps: int  # Max steps per episode


class Intraday_Trade_Action_Space(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
