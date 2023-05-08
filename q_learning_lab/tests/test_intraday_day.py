from q_learning_lab.domain.models.intraday_market_models import (
    DNN_Params,
    Intraday_Trade_Action_Space,
)

from  q_learning_lab.domain.deep_q_learn import (
    SequentialStructure,
    DeepAgent
)

def test_load_dnn_struct() -> None:
    input_dim:int = 16
    dnn_params:DNN_Params = DNN_Params(
        input_feacture_dim=(input_dim,),
        first_layer_struct={"units": input_dim*6, "activation": "relu"},
        mid_layers_struct=[
            {"units": input_dim*2, "activation": "relu"},
            {"units": input_dim*2, "activation": "relu"},
        ],
        output_layer_struct={"units": len(Intraday_Trade_Action_Space), "activation": "linear"},
    )
    seq_struct:SequentialStructure = dnn_params.get_dnn_structure()
    #Try to create the kera sequential model
    deep_agent = DeepAgent(
        structure=seq_struct,
        learning_rate=0.001,
        discount_factor=0.99,
    )
    _seq_model = deep_agent._create_sequential_model(structure=seq_struct)
    assert _seq_model is not None
    pass