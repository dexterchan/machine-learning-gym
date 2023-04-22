from __future__ import annotations
from enum import Enum

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    n_max_steps: int  # Max steps per episode
    learning_rate: float  # Learning rate (alpha)
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    savefig_folder: Path  # Root folder where plots are saved
    savemodel_folder: Path  # Root folder where models are saved
    start_epsilon: float = 1.0  # Starting exploration probability
    min_epsilon: float = 0.05  # Minimum exploration probability
    decay_rate: float = 0.001  # Exponential decay rate for exploration prob
    min_replay_size: int = 1000  # minimum replay size to train the model
    every_n_steps_to_train_main_model: int = 4  # Train the model every n steps
    every_m_steps_to_copy_main_weights_to_target_model: int = (
        100  # Copy weights every m steps
    )
    replay_memory_size: int = 50000  # Maximum size of replay memory
    train_batch_size: int = 64 * 2  # Size of batch taken from replay memory
    
class Intraday_Trade_Action_Space(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2