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
import json
import os

from ..utility.logging import get_logger

logger = get_logger(__name__)
# Reference https://github.com/mswang12/minDQN/blob/main/minDQN.py
import time

np.random.seed(int(time.time()))
tf.random.set_seed(int(time.time()))


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
        if structure is not None:
            self.model = self._create_sequential_model(structure=structure)
        pass

    @property
    def verbose(self) -> bool:
        return self.is_verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.is_verbose = value

    def save_agent(
        self, path: str, episode: int, epsilon: float, total_reward: float
    ) -> None:
        """save the agent into file
            it will save into two files:
            - path + ".tfm" : tensorflow model
            - path + ".json" : agent parameters

        Args:
            path (str): file path
        """
        tensorflow_model_path = path + ".tfm"
        agent_path = path + ".json"
        self.model.save(tensorflow_model_path)
        model_dict: dict = {
            "episode": episode,
            "learning_rate": self.learning_rate,
            "discounting_factor": self.discounting_factor,
            "epsilon": epsilon,
            "total_reward": total_reward,
        }
        with open(agent_path, "w") as f:
            json.dump(model_dict, f)
        pass

    @classmethod
    def load_agent(cls, path: str) -> DeepAgent:
        """load the agent from file
            it will load from two files:
            - path + ".tfm" : tensorflow model
            - path + ".json" : agent parameters

        Args:
            path (str): file path
        """
        tensorflow_model_path = path + ".tfm"
        agent_path = path + ".json"
        _sequential_model = keras.models.load_model(tensorflow_model_path)
        with open(agent_path, "r") as f:
            model_dict: dict = json.load(f)
        learning_rate = model_dict["learning_rate"]
        discounting_factor = model_dict["discounting_factor"]
        instance = cls(
            structure=None,
            learning_rate=learning_rate,
            discount_factor=discounting_factor,
        )
        instance.model = _sequential_model
        return instance

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
                kernel_initializer=structure.input_layer.kernel_initializer,
            )
        )
        for layer in structure.process_layers:
            model.add(
                keras.layers.Dense(
                    units=layer.units,
                    activation=layer.activation,
                    kernel_initializer=layer.kernel_initializer,
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

    def predict_batch(self, states: np.ndarray) -> Any:
        """_summary_

        Args:
            states (np.ndarray): column based vector (M,N) in shape

        Returns:
            Any: action space with Q values (2,1)
        """
        return self.model.predict(states)

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

        def _exploit(state: np.array) -> int:
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

    def play(self, env: Execute_Environment, max_step: int) -> tuple[float, bool]:
        """play the game"""

        state, _ = env.reset()
        total_reward = 0
        COMPLETE: bool = False
        for step in range(1, max_step + 1):
            if self.is_verbose:
                env.render()
                time.sleep(0.2)
            # Get the action
            action = self.epsilon_greedy(
                env=env, state=state, epsilon=0, is_exploit_only=True
            )

            next_state, reward, terminated, truncated, info = env.step(action=action)
            total_reward += reward
            state = next_state
            COMPLETE = step >= max_step

            if terminated:
                COMPLETE = step >= max_step
                if COMPLETE:
                    logger.info("Finish with max step")
                break
            pass
        logger.info(
            "Finished playing with total reward: %s Finish state: %s , Complete: %s, step: %s",
            total_reward,
            terminated,
            COMPLETE,
            step,
        )
        return total_reward, COMPLETE


class Reinforcement_DeepLearning:
    @staticmethod
    def train(
        env: Execute_Environment,
        params: NamedTuple,
        dnn_structure: SequentialStructure,
        is_verbose: bool = False,
        model_name: str = "CartPole-v1",
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

        model_path: str = os.path.join(params.savemodel_folder, "training", model_name)

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
            is_verbose=is_verbose,
        )
        # 1b. initialize the target model, (updated every "every_m_steps_to_copy_main_weights_to_target_model" steps)
        target = DeepAgent(
            structure=dnn_structure,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            is_verbose=False,
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
                action = main.epsilon_greedy(env=env, state=state, epsilon=epsilon)

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
                        f"Episode{episode}: Total training rewards: {total_training_rewards} after n steps = {steps_to_update_target_model} with final reward = {total_training_rewards}"
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
            if episode % 10 == 0:
                main.save_agent(
                    path=f"{model_path}_{episode}",
                    episode=episode,
                    epsilon=epsilon,
                    total_reward=total_training_rewards,
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

        current_qs_list = main.predict_batch(current_states)
        future_qs_list = target.predict_batch(next_states)

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
