from __future__ import annotations
import tensorflow as tf
import tensorflow.keras.initializers as kernel_initializer
from tensorflow import keras
from keras.losses import LossFunctionWrapper
from keras.optimizers import Optimizer
from typing import NamedTuple, Any

# Reference https://github.com/mswang12/minDQN/blob/main/minDQN.py


class InputLayer(NamedTuple):
    units: int
    input_shape: tuple[int, int]
    activation: str
    kernel_initializer: str


class ProcessLayer(NamedTuple):
    units: int
    activation: str
    kernel_initializer: str


class SequentialStructure(NamedTuple):
    initializer: kernel_initializer
    input_layer: InputLayer
    process_layers: list[ProcessLayer]
    loss_function: LossFunctionWrapper
    optimizer: keras.optimizers.Optimizer


class DeepAgent:
    def __init__(
        self,
        structure: SequentialStructure,
        learning_rate: float,
        discount_rate: float,
        is_verbose: bool = False,
    ) -> None:
        self.is_verbose: bool = is_verbose
        self.learning_rate: float = learning_rate
        self.discounting_rate: float = discount_rate
        self.model = self._create_sequential_model(structure=structure)
        pass

    def _create_sequential_model(
        self, structure: SequentialStructure
    ) -> keras.Sequential:
        """create the Sequetial model from the structure given

        Args:
            structure (SequentialStructure): _description_

        Returns:
            keras.Sequential: Sequential model
        """
        model = keras.Sequential()
        model.add(
            keras.layers.Dense(
                units=structure.input_layer.units,
                input_shape=structure.input_layer.input_shape,
                activation=structure.input_layer.activation,
                kernel_initializer=structure.initializer,
            )
        )
        for layer in structure.process_layers:
            model.add(
                keras.layers.Dense(
                    units=layer.units,
                    activation=layer.activation,
                    kernel_initializer=structure.initializer,
                )
            )
        model.compile(
            loss=structure.loss_function,
            optimizer=structure.optimizer,
            metrics=["accuracy"],
        )
        return model

    def copy_weights(self, other: DeepAgent) -> None:
        """copy weights from other agent

        Args:
            other (DeepAgent): _description_
        """
        self.model.set_weights(other.model.get_weights())
