from __future__ import annotations
from enum import Enum
from typing import Any, NamedTuple
from pathlib import Path
import tensorflow as tf
import time


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


class Env_Params(NamedTuple):
    total_episodes: int  # Total episodes
    n_max_steps: int  # Max steps per episode


class Action_Space(int, Enum):
    LEFT = 0
    RIGHT = 1


from ..deep_q_learn import (
    SequentialStructure,
    ProcessLayer,
    InputLayer,
    DeepAgent,
)


def get_dnn_structure(input_dim: tuple, output_dim: int) -> SequentialStructure:
    """Get pre-defined DNN structure for CartPole-v1

    Args:
        input_dim (tuple): input dimenstion of state space
        output_dim (int): output dimension of action space

    Returns:
        SequentialStructure: Sequential Structure of DNN
    """

    init = tf.keras.initializers.HeUniform(seed=int(time.time()))
    return SequentialStructure(
        initializer=init,
        input_layer=InputLayer(
            units=24,
            input_shape=input_dim,
            activation="relu",
            kernel_initializer=init,
        ),
        process_layers=[
            ProcessLayer(units=12, activation="relu", kernel_initializer=init),
            ProcessLayer(
                units=output_dim, activation="linear", kernel_initializer=init
            ),
        ],
        loss_function=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
