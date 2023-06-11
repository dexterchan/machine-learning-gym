from typing import NamedTuple

class BaseEnv_Params(NamedTuple):
    total_episodes: int  # Total episodes
    n_max_steps: int  # Max steps per episode if value is not zero
    episode_batch: int = 200 #Episode batch size to restart the training environment
    batch_mode:bool = False #If true, the training will be done in batch mode