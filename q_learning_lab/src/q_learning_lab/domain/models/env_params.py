from typing import NamedTuple

class BaseEnv_Params(NamedTuple):
    total_episodes: int  # Total episodes
    n_max_steps: int  # Max steps per episode
    episode_batch: int = 200 #Episode batch