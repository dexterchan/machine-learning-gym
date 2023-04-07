from __future__ import annotations
from ..port.environment import Execute_Environment
from .frozen_lake.models import Action_Space
from ..port.environment import Execute_Environment
# from ..utility.logging import getLogger
from logging import getLogger
import time
import numpy as np


# Reference: https://toxwardsdatascience.com/q-learning-algorithm-how-to-successfully-teach-an-intelligent-agent-to-play-a-game-933595fd1abf
# https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#sphx-glr-tutorials-training-agents-frozenlake-tuto-py
class Q_Table:
    def __init__(self, env:Execute_Environment) -> None:
        self.values = np.zeros((env.observation_space_dim, env.action_space_dim))
        pass


logger = getLogger(name=__name__)


class Agent:
    def __init__(self, is_verbose: bool = False) -> None:
        self.is_verbose: bool = is_verbose
        pass

    def random_walk(self, env: Execute_Environment, random_steps: int) -> None:
        """random run before any training

        Args:
            env (Execute_Environment): _description_
            random_steps (int): _description_
        """
        state, info = env.reset()
        for _ in range(random_steps):
            if self.is_verbose:
                env.render()
            # Pass the random action into the step function
            random_action: Action_Space = env.sample_action_space()
            result = env.step(random_action)
            observation_space, reward, terminated, truncated, info = result

            # Wait a little bit before the next frame
            time.sleep(0.2)
            if terminated:
                if self.is_verbose:
                    env.render()
                # Reset environment
                state, info = env.reset()
        env.close()
        pass
