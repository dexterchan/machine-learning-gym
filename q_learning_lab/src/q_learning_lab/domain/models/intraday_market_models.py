from __future__ import annotations
from tradesignal_mtm_runner.models import Buy_Sell_Action_Enum
from typing import NamedTuple, Any
from enum import Enum
from .env_params import BaseEnv_Params

from ..deep_q_learn import (
    SequentialStructure,
    ProcessLayer,
    InputLayer,
)
from time import time
import tensorflow as tf

class EnvParams(BaseEnv_Params):
    pass


class Intraday_Trade_Action_Space(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

    @staticmethod
    def convert_from_buy_sell_action_enum(actionEnum:Buy_Sell_Action_Enum)->Intraday_Trade_Action_Space:

        if actionEnum.BUY:
            return Intraday_Trade_Action_Space.BUY
        elif actionEnum.SELL:
            return Intraday_Trade_Action_Space.SELL
        elif actionEnum.HOLD:
            return Intraday_Trade_Action_Space.HOLD
        else:
            raise NotImplementedError(f"No conversion found for action enum: {actionEnum}")

class DNN_Params(NamedTuple):
    input_feacture_dim: tuple #e.g. (16,)
    first_layer_struct:dict = {"units": 16*6, "activation": "relu"}
    mid_layers_struct:list[dict] = [
        {"units": 16*2, "activation": "relu"},
        {"units": 16*2, "activation": "relu"},
    ]
    output_layer_struct:dict = {"units": len(Intraday_Trade_Action_Space), "activation": "linear"}
    pass

def get_dnn_structure(input_dim: tuple, output_dim: int) -> SequentialStructure:
    """Get pre-defined DNN structure for Intraday Trade

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
