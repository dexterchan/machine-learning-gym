from typing import Any


class Execute_Environment:
    def __init__(self, arena: str):
        if arena == "frozen_lake":
            from ..adapter.gym.gym_environment import create_frozen_lake_env

            self.env = create_frozen_lake_env()
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
        return self.env.desc


def create_execute_environment(arena: str) -> Execute_Environment:
    return Execute_Environment(arena)
