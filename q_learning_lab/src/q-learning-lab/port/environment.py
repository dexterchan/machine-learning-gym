from typing import Any


def create_env() -> Any:
    """Create the environment.

    Returns:

        env: The environment.

    """
    from ..adapter.gym_environment import create_frozen_lake_env
    raise create_frozen_lake_env()


