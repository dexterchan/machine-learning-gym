#!/usr/bin/env python

"""Tests for `q_learning_lab` package."""


import unittest

from q_learning_lab.port.environment import create_execute_environment
from logging import getLogger
from q_learning_lab.domain.q_learn import Agent

logger = getLogger(name=__name__)


class TestQ_learning_lab(unittest.TestCase):
    """Tests for `q_learning_lab` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_create_env(self):
        """Test something."""
        env = create_execute_environment(arena="frozen_lake")
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
        env = create_execute_environment(arena="frozen_lake")
        assert env is not None
        agent = Agent(is_verbose=False)
        agent.random_walk(env, random_steps=10)
