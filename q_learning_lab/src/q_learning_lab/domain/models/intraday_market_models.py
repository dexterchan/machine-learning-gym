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
import time
import tensorflow as tf

class EnvParams(BaseEnv_Params):
    pass


class Intraday_Trade_Action_Space(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

    @staticmethod
    def convert_to_buy_sell_action_enum(intradayEnum:Intraday_Trade_Action_Space)->Buy_Sell_Action_Enum:
        if intradayEnum.BUY:
            return Buy_Sell_Action_Enum.BUY
        elif intradayEnum.SELL:
            return Buy_Sell_Action_Enum.SELL
        elif intradayEnum.HOLD:
            return Buy_Sell_Action_Enum.HOLD
        else:
            raise NotImplementedError(f"No conversion found for intraday enum: {intradayEnum}")
            

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
    
    def abc(self) -> None:
        pass

    def get_dnn_structure(self) -> SequentialStructure:
        """Get pre-defined DNN structure for Intraday Trade
        Returns:
            SequentialStructure: Sequential Structure of DNN
        """

        init = tf.keras.initializers.HeUniform(seed=int(time.time()))
        return SequentialStructure(
            initializer=init,
            input_layer=InputLayer(
                units=self.first_layer_struct["units"],
                input_shape=self.input_feacture_dim,
                activation=self.first_layer_struct["activation"],
            ),
            process_layers=[
                ProcessLayer(**self.first_layer_struct),
                *[ProcessLayer(**layer_struct) for layer_struct in self.mid_layers_struct],
                ProcessLayer(**self.output_layer_struct),
            ],
            loss_function=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        )
    pass
