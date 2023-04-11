from __future__ import annotations
from ..port.environment import Execute_Environment

import random
import tensorflow as tf
import tensorflow.keras.initializers as kernel_initializer
from tensorflow import keras
from keras.losses import LossFunctionWrapper
from keras.optimizers import Optimizer
from typing import NamedTuple, Any
from collections import deque
import numpy as np
from ..utility.logging import get_logger

logger = get_logger(__name__)
# Reference https://github.com/mswang12/minDQN/blob/main/minDQN.py


class InputLayer(NamedTuple):
    units: int
    input_shape: tuple[int, int]
    activation: str
    kernel_initializer: str


class ProcessLayer(NamedTuple):
    units: int
    activation: str
    kernel_initializer: str


class SequentialStructure(NamedTuple):
    initializer: kernel_initializer
    input_layer: InputLayer
    process_layers: list[ProcessLayer]
    loss_function: LossFunctionWrapper
    optimizer: keras.optimizers.Optimizer


class DeepAgent:
    def __init__(
        self,
        structure: SequentialStructure,
        learning_rate: float,
        discount_factor: float,
        is_verbose: bool = False,
    ) -> None:
        self.is_verbose: bool = is_verbose
        self.learning_rate: float = learning_rate
        self.discounting_factor: float = discount_factor
        self.model = self._create_sequential_model(structure=structure)
        pass

    def _create_sequential_model(
        self, structure: SequentialStructure
    ) -> keras.Sequential:
        """create the Sequetial model from the structure given

        Args:
            structure (SequentialStructure): _description_

        Returns:
            keras.Sequential: Sequential model
        """
        model = keras.Sequential()
        model.add(
            keras.layers.Dense(
                units=structure.input_layer.units,
                input_shape=structure.input_layer.input_shape,
                activation=structure.input_layer.activation,
                kernel_initializer=structure.initializer,
            )
        )
        for layer in structure.process_layers:
            model.add(
                keras.layers.Dense(
                    units=layer.units,
                    activation=layer.activation,
                    kernel_initializer=structure.initializer,
                )
            )
        model.compile(
            loss=structure.loss_function,
            optimizer=structure.optimizer,
            metrics=["accuracy"],
        )
        return model

    def copy_weights(self, other: DeepAgent) -> None:
        """copy weights from other agent

        Args:
            other (DeepAgent): _description_
        """
        self.model.set_weights(other.model.get_weights())

    def predict(self, state: np.ndarray) -> Any:
        """_summary_

        Args:
            state (np.ndarray): column based vector (N,1) in shape

        Returns:
            Any: action space with Q values (2,1)
        """
        return self.model.predict(state.reshape(1, state.shape[0])).flatten()

    def epsilon_greedy(
        self,
        env: Execute_Environment,
        state: np.ndarray,
        epsilon: float,
        is_exploit_only: bool = False,
    ) -> int:
        """Explore using the Epsilon Greedy Exploration Strategy

        Args:
            env (Execute_Environment): Execute environment
            state (np.ndarray): State in column vector (N,1) shape
            epsilon (float): epsilon value for exploration
            is_exploit_only (bool, optional): exploit only. Defaults to False.

        Returns:
            int: action value
        """

        def _exploit() -> int:
            output = self.predict(state=state)
            return np.argmax(output)

        if not is_exploit_only:
            randnum = np.random.uniform(0, 1)
            if randnum < epsilon:
                action = int(env.sample_action_space())  # explore
            else:
                action = _exploit(state=state)  # exploit
        else:
            action = _exploit(state=state)  # exploit

        return action


class Reinforcement_DeepLearning:
    @staticmethod
    def train(
        env: Execute_Environment,
        params: NamedTuple,
        dnn_structure: SequentialStructure,
        is_verbose: bool = False,
    ) -> dict[str, DeepAgent]:
        """_summary_

        Args:
            env (Execute_Environment): _description_
            params (NamedTuple): _description_
            is_verbose (bool, optional): render the environment during training. Defaults to False.

        Returns:
            dict[str,DeepAgent]: _description_
        """
        epsilon = (
            params.start_epsilon
        )  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        max_epsilon = (
            params.start_epsilon
        )  # You can't explore more than 100% of the time
        min_epsilon = (
            params.min_epsilon
        )  # At a minimum, we'll always explore 1% of the time
        decay = params.decay_rate

        every_n_steps_to_train_main_model = params.every_n_steps_to_train_main_model
        every_m_steps_to_copy_main_weights_to_target_model = (
            params.every_m_steps_to_copy_main_weights_to_target_model
        )
        train_batch_size = params.train_batch_size
        min_replay_size = params.min_replay_size
        total_episodes = params.total_episodes

        learning_rate = params.learning_rate
        discount_factor = params.gamma
        # 1a. initialize the main model, (updated every "every_n_steps_to_train_main_model" steps)
        main = DeepAgent(
            structure=dnn_structure,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        # 1b. initialize the target model, (updated every "every_m_steps_to_copy_main_weights_to_target_model" steps)
        target = DeepAgent(
            structure=dnn_structure,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        target.copy_weights(main)

        replay_memory_list = deque(maxlen=params.replay_memory_size)

        steps_to_update_target_model: int = 0

        for episode in range(total_episodes):
            total_training_rewards: float = 0
            state, _ = env.reset()
            terminated: bool = False

            while not terminated:
                steps_to_update_target_model += 1
                if is_verbose:
                    env.render()

                # 2. Explore with Epsilon Greedy exploration
                action = main.epsilon_greedy(state=state, epsilon=epsilon)

                next_state, reward, terminated, truncated, info = env.step(
                    action=action
                )
                replay_memory_list.append(
                    [state, action, reward, next_state, terminated]
                )

                # 3. Train the main model for every_n_steps_to_train_main_model
                if (
                    steps_to_update_target_model % every_n_steps_to_train_main_model
                    == 0
                    or terminated
                ):
                    if (
                        len(replay_memory_list) > min_replay_size
                        and len(replay_memory_list) > train_batch_size
                    ):
                        # sample a minibatch from the replay memory
                        mini_batch = random.sample(replay_memory_list, train_batch_size)
                        main = Reinforcement_DeepLearning._train_main_model(
                            main=main,
                            target=target,
                            mini_batch=mini_batch,
                            current_states=np.array(
                                [state for state, _, _, _, _ in mini_batch]
                            ),
                            next_states=np.array(
                                [next_state for _, _, _, next_state, _ in mini_batch]
                            ),
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
                        )
                state = next_state
                total_training_rewards += reward

                if terminated:
                    logger.info(
                        f"Total training rewards: {total_training_rewards} after n steps = {steps_to_update_target_model} with final reward = {total_training_rewards}"
                    )
                    if (
                        steps_to_update_target_model
                        >= every_m_steps_to_copy_main_weights_to_target_model
                    ):
                        logger.info("Copying main network weights to target network")
                        target.copy_weights(main)
                        steps_to_update_target_model = 0
                    break
                pass
            pass
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay * episode
            )
        return {"main": main, "target": target}

    @staticmethod
    def _train_main_model(
        main: DeepAgent,
        target: DeepAgent,
        mini_batch: list[list],
        current_states: np.array,
        next_states: np.array,
        learning_rate: float,
        discount_factor: float,
    ) -> DeepAgent:
        """_summary_

        Args:
            main (DeepAgent): _description_
            target (DeepAgent): _description_
            mini_batch (list[list]): _description_
            current_states (np.array): _description_
            next_states (np.array): _description_
            learning_rate (float): _description_
            discount_factor (float): _description_

        Returns:
            DeepAgent: _description_
        """

        current_qs_list = main.model.predict(current_states)
        future_qs_list = target.model.predict(next_states)

        X = []
        Y = []
        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if done:
                max_future_q = reward
            else:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            current_qs = current_qs_list[index]
            current_qs[action] = current_qs[action] + learning_rate * (
                max_future_q - current_qs[action]
            )
            X.append(state)
            Y.append(current_qs)
        main.model.fit(np.array(X), np.array(Y), batch_size=len(X), verbose=0)
        return main
