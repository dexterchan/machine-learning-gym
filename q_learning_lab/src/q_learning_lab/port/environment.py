from typing import Any, NamedTuple
from ..utility.logging import get_logger

logger = get_logger(__name__)


class Execute_Environment:
    def __init__(self, arena: str, params: dict):
        if arena.lower() == "frozen_lake":
            from ..adapter.gym.gym_environment import Fronzen_Lake_Environment
            from ..domain.models.frozen_lake_models import Params as forzen_lake_params

            self.env_params = forzen_lake_params(**params)
            self.env = Fronzen_Lake_Environment(params=self.env_params)

        elif arena.lower() == "cartpole-v1":
            from ..adapter.gym.gym_environment import Cart_Pole_v1_Environment
            from ..adapter.gym.gym_environment import Env_Params as cartpole_v1_params

            self.env_params = cartpole_v1_params(**params)
            self.env = Cart_Pole_v1_Environment(params=self.env_params)

        elif arena.lower() == "dummy":
            from ..adapter.dummy.dummy_environment import Dummy_Environment
            from ..adapter.gym.gym_environment import Env_Params as cartpole_v1_params

            self.env_params = cartpole_v1_params(**params)
            self.env = Dummy_Environment(params={})

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

    def sample_action_space(self) -> Any:
        return self.env.sample_action_space()

    @property
    def observation_space_dim(self) -> int:
        return self.env.observation_space_dim

    @property
    def action_space_dim(self) -> int:
        return self.env.action_space_dim

    @property
    def env_params(self) -> Any:
        return self._env_params

    @env_params.setter
    def env_params(self, value: Any) -> None:
        self._env_params = value
    
    @property
    def measure_result(self) -> float:
        return 0
    
    def __iter__(self):
        yield self


def create_execute_environment(arena: str, params: dict) -> Execute_Environment:
    return Execute_Environment(arena=arena, params=params)
