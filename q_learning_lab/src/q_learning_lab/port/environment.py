from typing import Any, NamedTuple


class Execute_Environment:
    def __init__(self, arena: str, params: NamedTuple):
        if arena == "frozen_lake":
            from ..adapter.gym.gym_environment import Fronzen_Lake_Environment

            self.env = Fronzen_Lake_Environment(params=params)
        else:
            raise NotImplementedError("Arena not implemented")

    def render(self):
        self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action: int):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def get_description(self) -> Any:
        return self.env.get_description()

    def get_action_space(self) -> Any:
        return self.env.get_action_space()

    def sample_action_space(self) -> Any:
        return self.env.sample_action_space()

    @property
    def observation_space_dim(self) -> int:
        return self.env.observation_space_dim

    @property
    def action_space_dim(self) -> int:
        return self.env.action_space_dim


def create_execute_environment(arena: str, params: NamedTuple) -> Execute_Environment:
    return Execute_Environment(arena=arena, params=params)
