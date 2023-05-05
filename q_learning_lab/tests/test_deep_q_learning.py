from __future__ import annotations
import tensorflow as tf
from pathlib import Path
import os
from q_learning_lab.port.environment import create_execute_environment
from q_learning_lab.port.agent import create_new_deep_agent
from q_learning_lab.domain.models.cart_pole_v1_models import Env_Params
from q_learning_lab.domain.models.agent_params import Agent_Params
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
        # optimizer_learning_rate = 0.001
        self.env_params = Env_Params(total_episodes=20, n_max_steps=3000)
        self.env_params_dict: dict = self.env_params._asdict()

        self.agent_params: dict = Agent_Params(
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
        env = create_execute_environment(
            arena="CartPole-v1", params=self.env_params_dict
        )
        assert env is not None
        logger.debug(env.get_description())
        print(type(env.get_description()))

        action = env.sample_action_space()
        print(action)
        assert action in [0, 1]
        assert env.action_space_dim == 2
        assert env.observation_space_dim[0] == 4

    def test_create_agent(self):
        env = create_execute_environment(
            arena="CartPole-v1", params=self.env_params_dict
        )
        assert env is not None

        dnn_structure: SequentialStructure = get_dnn_structure(
            input_dim=env.observation_space_dim,
            output_dim=env.action_space_dim,
        )
        main = create_new_deep_agent(
            params=self.agent_params, structure=dnn_structure, is_verbose=False
        )
        assert main is not None
        assert isinstance(main, DeepAgent)
        target = create_new_deep_agent(
            params=self.agent_params, structure=dnn_structure, is_verbose=False
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
        env = create_execute_environment(
            arena="CartPole-v1", params=self.env_params_dict
        )
        assert env is not None
        # Setup DNN Structure
        dnn_structure: SequentialStructure = get_dnn_structure(
            input_dim=env.observation_space_dim,
            output_dim=env.action_space_dim,
        )
        # 1a. initialize the main model, (updated every "every_n_steps_to_train_main_model" steps)
        main = DeepAgent(
            structure=dnn_structure,
            learning_rate=self.agent_params.learning_rate,
            discount_factor=self.agent_params.gamma,
        )
        # 1b. initialize the target model, (updated every "every_m_steps_to_copy_main_weights_to_target_model" steps)
        target = DeepAgent(
            structure=dnn_structure,
            learning_rate=self.agent_params.learning_rate,
            discount_factor=self.agent_params.gamma,
        )
        target.copy_weights(other=main)

        def prepare_fake_mini_batch() -> list[tuple]:
            # Prepare a random test data of numpy array of shape (100, 4)
            _current_states = np.random.random((100, 4))
            _future_states = np.random.random((100, 4))
            # Create a random action space of shape (100, 1)
            actions = np.random.randint(0, 2, size=(100, 1))
            # Create a ONE reward space of shape (100, 1)
            rewards = np.ones((100, 1))
            # Create a random boolean done space of shape (100,1) with 95% of False and 5% of True
            dones = np.random.random(size=(100, 1)) > 0.95

            # Construct mini-batch with random data:
            # 1. _current_states
            # 2. actions
            # 3. rewards
            # 4. _future_states
            # 5. dones
            _mini_batch = []
            for inx, current_state in enumerate(_current_states):
                _mini_batch.append(
                    (
                        current_state,
                        actions[inx][0],
                        rewards[inx][0],
                        _future_states[inx],
                        dones[inx][0],
                    )
                )
            return _mini_batch, _current_states, _future_states
        
        mini_batch, current_states, future_states = prepare_fake_mini_batch()

        # Convert mini_batch_array to a list of tuples
        Reinforcement_DeepLearning._train_main_model(
            main=main,
            target=target,
            mini_batch=mini_batch,
            current_states=current_states,
            next_states=future_states,
            learning_rate=self.agent_params.learning_rate,
            discount_factor=self.agent_params.gamma,
        )

        model_path = os.path.join(
            self.agent_params.savemodel_folder, "dummy", "CartPole-v1-dummy"
        )
        main.save_agent(
                path=model_path,
                    episode=1,
                    epsilon=1,
                    total_reward=1,)
        
        #test load agent
        cloned_agent, last_run_para = DeepAgent.load_agent(path=model_path)
        cloned_agent2, last_run_para2 = DeepAgent.load_agent(path=model_path)
        assert cloned_agent is not None
        assert isinstance(cloned_agent, DeepAgent)
        assert last_run_para["episode"] == 1
        assert last_run_para["epsilon"] == 1
        assert last_run_para["total_reward"] == 1
        #Compare load agent with main agent
        assert type(cloned_agent.model) == type(main.model)

        assert cloned_agent.learning_rate == main.learning_rate
        assert cloned_agent.discounting_factor == main.discounting_factor
        
        # Compare two keras.engine.sequential.Sequential objects are equal
        # assert (cloned_agent.model) == (main.model) not working
        assert cloned_agent.model.get_config() == main.model.get_config()
        # Compare two keras.engine.sequential.Sequential objects are equal
        for i, layer in enumerate(cloned_agent.model.layers):
            #logger.debug("layer {} weights: {}".format(i, layer.get_weights()))
            weights_list:list[np.array] = layer.get_weights()
            for j, weights in enumerate(weights_list):
                
                assert (weights == main.model.layers[i].get_weights()[j]).all()
        
        # Test the run
        test_states = np.random.random((200, env.observation_space_dim[0]))
        output_main_result = main.predict_batch(
            states=test_states
        )
        assert (200, env.action_space_dim) == output_main_result.shape

        output_clone_result = cloned_agent.predict_batch(
            states=test_states
        )
        assert (200, env.action_space_dim) == output_clone_result.shape

        assert (output_main_result == output_clone_result).all()

        # Test if still can train
        mini_batch, current_states, future_states = prepare_fake_mini_batch()
        new_traing_agent = Reinforcement_DeepLearning._train_main_model(
            main=cloned_agent,
            target=cloned_agent2,
            mini_batch=mini_batch,
            current_states=current_states,
            next_states=future_states,
            learning_rate=last_run_para["learning_rate"],
            discount_factor=last_run_para["discounting_factor"],
        )
        assert main.model.get_config() == new_traing_agent.model.get_config()
        for i, layer in enumerate(main.model.layers):
            #logger.debug("layer {} weights: {}".format(i, layer.get_weights()))
            weights_list:list[np.array] = layer.get_weights()
            for j, weights in enumerate(weights_list):
                assert (weights != new_traing_agent.model.layers[i].get_weights()[j]).any()



    @pytest.mark.skipif(
        "test_training" not in test_cases, reason="skipped test_training"
    )
    def test_training(self):
        env = create_execute_environment(
            arena="CartPole-v1", params=self.env_params_dict
        )
        from q_learning_lab.domain.models.cart_pole_v1_models import get_dnn_structure

        dnn_structure = get_dnn_structure(
            input_dim=env.observation_space_dim,
            output_dim=env.action_space_dim,
        )

        deepagent_dict = Reinforcement_DeepLearning.train(
            env=env,
            agent_params=self.agent_params,
            env_params=self.env_params,
            dnn_structure=dnn_structure,
            is_verbose=False,
        )

        assert deepagent_dict is not None
        model_path = os.path.join(
            self.agent_params.savemodel_folder, "unittest", "CartPole-v1-unittest"
        )

        pass
