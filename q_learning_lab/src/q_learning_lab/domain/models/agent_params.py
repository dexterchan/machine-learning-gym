from pathlib import Path
from typing import NamedTuple


class Agent_Params(NamedTuple):
    learning_rate: float  # Learning rate (alpha)
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    savefig_folder: Path  # Root folder where plots are saved
    savemodel_folder: Path  # Root folder where models are saved
    save_best_only: bool = True  # support early stopping and save best model only
    start_epsilon: float = 1.0  # Starting exploration probability
    min_epsilon: float = 0.05  # Minimum exploration probability
    decay_rate: float = 0.001  # Exponential decay rate for exploration prob
    min_replay_size: int = 1000  # minimum replay size to train the model
    every_n_steps_to_train_main_model: int = 4  # Train the model every n steps
    every_m_steps_to_copy_main_weights_to_target_model: int = (
        100  # Copy weights every m steps
    )
    worse_than_best_reward_count_limit:int = 100
    replay_memory_size: int = 50000  # Maximum size of replay memory
    train_batch_size: int = 64 * 2  # Size of batch taken from replay memory
