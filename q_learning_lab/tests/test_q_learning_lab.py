#!/usr/bin/env python

"""Tests for `q_learning_lab` package."""


import unittest

from q_learning_lab.port.environment import create_execute_environment
from q_learning_lab.utility.logging import get_logger
from q_learning_lab.domain.q_learn import Agent
from q_learning_lab.domain.models.frozen_lake_models import Params
from pathlib import Path
from q_learning_lab.port.frozen_lake import create_agent

logger = get_logger(name=__name__, level="DEBUG")


class TestQ_learning_lab(unittest.TestCase):
    """Tests for `q_learning_lab` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.params = Params(
            total_episodes=2000,
            n_max_steps=100,
            learning_rate=0.8,
            gamma=0.95,
            epsilon=0.1,
            map_size=4,
            seed=123,
            is_slippery=False,
            n_runs=20,
            action_size=None,
            state_size=None,
            proba_frozen=0.9,
            savefig_folder=Path("_static/img/tutorials/"),
            start_epsilon=1.0,  # Starting exploration probability
            min_epsilon=0.05,  # Minimum exploration probability
            decay_rate=0.001,
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_create_env(self):
        """Test something."""
        env = create_execute_environment(arena="frozen_lake", params=self.params)
        assert env is not None
        logger.debug(env.get_description())
        print(type(env.get_description()))

        action = env.sample_action_space()
        print(action)
        print(type(action))
        assert env.observation_space_dim == 16
        assert env.action_space_dim == 4

    def test_random_walk(self):
        """Test something."""
        env = create_execute_environment(arena="frozen_lake", params=self.params)
        assert env is not None
        agent = Agent(
            learning_rate=self.params.learning_rate,
            discount_rate=self.params.gamma,
            is_verbose=False,
        )
        agent.random_walk(env, random_steps=10)

    def test_train(self):
        """Test something."""
        env = create_execute_environment(arena="frozen_lake", params=self.params)
        assert env is not None
        agent = Agent(
            learning_rate=self.params.learning_rate,
            discount_rate=self.params.gamma,
            is_verbose=False,
        )
        qtable = agent.train(
            env=env,
            n_episodes=self.params.total_episodes,
            n_max_steps=self.params.n_max_steps,
            start_epsilon=self.params.start_epsilon,
            min_epsilon=self.params.min_epsilon,
            decay_rate=self.params.decay_rate,
        )
        # logger.info(qtable)
        assert qtable is not None

    def test_evaluate(self):
        """Test something."""
        env = create_execute_environment(arena="frozen_lake", params=self.params)
        assert env is not None
        # agent = Agent(
        #     learning_rate=self.params.learning_rate,
        #     discount_rate=self.params.gamma,
        #     is_verbose=False,
        # )
        agent = create_agent(params=self.params, is_verbose=False)
        qtable = agent.train(
            env=env,
            n_episodes=self.params.total_episodes,
            n_max_steps=self.params.n_max_steps,
            start_epsilon=self.params.start_epsilon,
            min_epsilon=self.params.min_epsilon,
            decay_rate=self.params.decay_rate,
        )
        logger.debug(qtable)
        assert qtable is not None
        mean_reward, std_reward = agent.evaluate_agent(
            env=env, Qtable=qtable, n_eval_episodes=10, n_max_steps=100
        )
        logger.info(f"mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
