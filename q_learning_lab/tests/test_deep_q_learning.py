from __future__ import annotations
import tensorflow as tf
from pathlib import Path
import os
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

test_cases = ["test_train_main_model", "test_training"]


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
            savemodel_folder=Path("_static/model/tutorials/"),
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

    @pytest.mark.skipif(
        "test_train_main_model" not in test_cases, reason="skipped test_training"
    )
    def test_train_main_model(self):
        # Setup CartPole-v1 Environment
        env = create_execute_environment(arena="CartPole-v1", params=self.params)
        assert env is not None
        # Setup DNN Structure
        dnn_structure: SequentialStructure = get_dnn_structure(
            input_dim=env.observation_space_dim,
            output_dim=env.action_space_dim,
        )
        # 1a. initialize the main model, (updated every "every_n_steps_to_train_main_model" steps)
        main = DeepAgent(
            structure=dnn_structure,
            learning_rate=self.params.learning_rate,
            discount_factor=self.params.gamma,
        )
        # 1b. initialize the target model, (updated every "every_m_steps_to_copy_main_weights_to_target_model" steps)
        target = DeepAgent(
            structure=dnn_structure,
            learning_rate=self.params.learning_rate,
            discount_factor=self.params.gamma,
        )
        target.copy_weights(other=main)

        # Prepare a random test data of numpy array of shape (100, 4)
        current_states = np.random.random((100, 4))
        future_states = np.random.random((100, 4))
        # Create a random action space of shape (100, 1)
        actions = np.random.randint(0, 2, size=(100, 1))
        # Create a ONE reward space of shape (100, 1)
        rewards = np.ones((100, 1))
        # Create a random boolean done space of shape (100,1) with 95% of False and 5% of True
        dones = np.random.random(size=(100, 1)) > 0.95

        # Construct mini-batch with random data:
        # 1. current_states
        # 2. actions
        # 3. rewards
        # 4. future_states
        # 5. dones
        mini_batch = []
        for inx, current_state in enumerate(current_states):
            mini_batch.append(
                (
                    current_state,
                    actions[inx][0],
                    rewards[inx][0],
                    future_states[inx],
                    dones[inx][0],
                )
            )
        # Convert mini_batch_array to a list of tuples
        Reinforcement_DeepLearning._train_main_model(
            main=main,
            target=target,
            mini_batch=mini_batch,
            current_states=current_states,
            next_states=future_states,
            learning_rate=self.params.learning_rate,
            discount_factor=self.params.gamma,
        )

        model_path = os.path.join(
            self.params.savemodel_folder, "dummy", "CartPole-v1-dummy"
        )
        main.save_agent(path=model_path)
        pass

    @pytest.mark.skipif(
        "test_training" not in test_cases, reason="skipped test_training"
    )
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

        assert deepagent_dict is not None
        model_path = os.path.join(
            self.params.savemodel_folder, "unittest", "CartPole-v1-unittest"
        )
        deepagent_dict["main"].save_agent(path=model_path)
        pass
