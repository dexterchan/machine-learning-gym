import gymnasium as gym
from typing import Any


import matplotlib
import matplotlib.pyplot as plt  # for displaying environment states
from IPython import display
from typing import NamedTuple


from ...domain.models.frozen_lake_models import Action_Space


class Fronzen_Lake_Environment:
    def __init__(self, params: NamedTuple):
        self.env = gym.make(
            id="FrozenLake-v1",  # Choose one of the existing environments
            desc=None,  # Used to specify custom map for frozen lake. E.g., desc=["SFFF", "FHFH", "FFFH", "HFFG"].
            map_name=f"{params.map_size}x{params.map_size}",  # ID to use any of the preloaded maps. E.g., '4x4', '8x8'
            is_slippery=params.is_slippery,  # True/False. If True will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.
            max_episode_steps=params.n_max_steps,  # default=None, Maximum length of an episode (TimeLimit wrapper).
            autoreset=False,  # default=None, Whether to automatically reset the environment after each episode (AutoResetWrapper).
            disable_env_checker=None,  # default=None, If to run the env checker
            render_mode="rgb_array",  # The set of supported modes varies per environment. (And some third-party environments may not support rendering at all.)
        )

    def render(self):
        """Render and display current state of the environment"""
        plt.imshow(self.env.render())  # render current state and pass to pyplot
        plt.axis("off")
        display.display(plt.gcf())  # get current figure and display
        display.clear_output(wait=True)  # clear output before showing the next frame
        # self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action: Action_Space):
        return self.env.step(int(action))

    def close(self):
        self.env.close()

    def get_description(self) -> Any:
        return self.env.desc

    def sample_action_space(self) -> Action_Space:
        return Action_Space(self.env.action_space.sample())

    @property
    def observation_space_dim(self) -> int:
        return self.env.observation_space.n

    @property
    def action_space_dim(self) -> int:
        return self.env.action_space.n


from ...domain.models.cart_pole_v1_models import (
    Action_Space as Cart_Pole_Action_Space,
    Env_Params,
)


class Cart_Pole_v1_Environment:
    def __init__(self, params: Env_Params):
        self.env = gym.make(
            id="CartPole-v1",  # Choose one of the existing environments
            max_episode_steps=params.n_max_steps,  # default=None, Maximum length of an episode (TimeLimit wrapper).
            autoreset=False,  # default=None, Whether to automatically reset the environment after each episode (AutoResetWrapper).
            disable_env_checker=None,  # default=None, If to run the env checker
            render_mode="rgb_array",  # The set of supported modes varies per environment. (And some third-party environments may not support rendering at all.)
        )

    def render(self):
        """Render and display current state of the environment"""
        plt.imshow(self.env.render())  # render current state and pass to pyplot
        plt.axis("off")
        display.display(plt.gcf())  # get current figure and display
        display.clear_output(wait=True)  # clear output before showing the next frame
        # self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action: Cart_Pole_Action_Space):
        return self.env.step(int(action))

    def close(self):
        self.env.close()

    def get_description(self) -> Any:
        if hasattr(self.env, "desc"):
            return self.env.desc
        else:
            return "No description available."

    def sample_action_space(self) -> Cart_Pole_Action_Space:
        return Cart_Pole_Action_Space(self.env.action_space.sample())

    @property
    def observation_space_dim(self) -> int:
        if hasattr(self.env.observation_space, "n"):
            return self.env.observation_space.n
        elif hasattr(self.env.observation_space, "shape"):
            return self.env.observation_space.shape
        else:
            raise NotImplementedError("No dimension found for observation space.")

    @property
    def action_space_dim(self) -> int:
        return self.env.action_space.n
