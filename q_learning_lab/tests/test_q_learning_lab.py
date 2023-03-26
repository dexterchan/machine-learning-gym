#!/usr/bin/env python

"""Tests for `q_learning_lab` package."""


import unittest

from q_learning_lab.port.environment import create_execute_environment


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
        print(env.get_description())
        print(type(env.get_description()))
