from __future__ import annotations
import tensorflow as tf
from pathlib import Path

from q_learning_lab.port.environment import create_execute_environment
from q_learning_lab.port.agent import create_deep_agent
from q_learning_lab.domain.models.cart_pole_v1_models import (
    Params as Cart_Pole_V1_Params,
)
import unittest
import pytest
import numpy as np

from q_learning_lab.utility.logging import get_logger
from q_learning_lab.domain.deep_q_learn import (
    DeepAgent,
    SequentialStructure,
)
from q_learning_lab.domain.models.cart_pole_v1_models import get_dnn_structure

from q_learning_lab.domain.deep_q_learn import Reinforcement_DeepLearning

logger = get_logger(name=__name__, level="DEBUG")


class TestDeepQLearning(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        optimizer_learning_rate = 0.001
        self.params = Cart_Pole_V1_Params(
            total_episodes=200,
            n_max_steps=3000,
            learning_rate=0.7,
            gamma=0.618,
            epsilon=0.1,
            savefig_folder=Path("_static/img/tutorials/"),
            start_epsilon=1.0,  # Starting exploration probability
            min_epsilon=0.05,  # Minimum exploration probability
            decay_rate=0.001,
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_create_env(self):
        """Test something."""
        env = create_execute_environment(arena="CartPole-v1", params=self.params)
        assert env is not None
        logger.debug(env.get_description())
        print(type(env.get_description()))

        action = env.sample_action_space()
        print(action)
        assert action in [0, 1]
        assert env.action_space_dim == 2
        assert env.observation_space_dim[0] == 4

    def test_create_agent(self):
        env = create_execute_environment(arena="CartPole-v1", params=self.params)
        assert env is not None

        dnn_structure: SequentialStructure = get_dnn_structure(
            input_dim=env.observation_space_dim,
            output_dim=env.action_space_dim,
        )
        main = create_deep_agent(
            params=self.params, structure=dnn_structure, is_verbose=False
        )
        assert main is not None
        assert isinstance(main, DeepAgent)
        target = create_deep_agent(
            params=self.params, structure=dnn_structure, is_verbose=False
        )
        target.copy_weights(other=main)

        state = env.reset()
        action = main.predict(state=state[0])
        # logger.info(action.flatten())
        assert action.shape == (2,)
        logger.info(action)
        logger.info(np.argmax(action))

    @pytest.mark.skip(reason="not implemented yet")
    def test_training(self):
        env = create_execute_environment(arena="CartPole-v1", params=self.params)
        from q_learning_lab.domain.models.cart_pole_v1_models import get_dnn_structure

        dnn_structure = get_dnn_structure(
            input_dim=env.observation_space_dim,
            output_dim=env.action_space_dim,
        )

        deepagent_dict = Reinforcement_DeepLearning.train(
            env=env,
            params=self.params,
            dnn_structure=dnn_structure,
            is_verbose=False,
        )

        pass
