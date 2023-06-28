from __future__ import annotations
from ..port.environment import Execute_Environment

import random
import tensorflow as tf
import tensorflow.keras.initializers as kernel_initializer
from datetime import datetime
from tensorflow import keras
from keras.losses import LossFunctionWrapper
from keras.optimizers import Optimizer
from typing import NamedTuple, Any
from collections import deque
import numpy as np
import json
import os
from .models.env_params import BaseEnv_Params
from .models.agent_params import Agent_Params

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


class ProcessLayer(NamedTuple):
    units: int
    activation: str


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
            self._model = self._create_sequential_model(structure=structure)
        pass

    @property
    def verbose(self) -> bool:
        return self.is_verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.is_verbose = value

    @property
    def model(self) -> keras.Model:
        return self._model

    def save_agent(
        self, path: str, episode: int, 
        epsilon: float, 
        total_rewards_history: list[float],
        eval_rewards_history: list[dict]
    ) -> None:
        """save the agent into file
            it will save into two files:
            - path + ".tfm" : tensorflow model
            - path + ".json" : agent parameters

        Args:
            path (str): file path
            episode (int): last episode
            epsilon (float): last epsilon
            total_rewards_history (list[float]): total rewards history
            eval_rewards_history (list[dict]): evaluation rewards history
        """
        tensorflow_model_path = path + ".tfm"
        agent_path = path + ".json"
        self._model.save(tensorflow_model_path)
        model_dict: dict = {
            "episode": episode,
            "learning_rate": self.learning_rate,
            "discounting_factor": self.discounting_factor,
            "epsilon": epsilon,
            "total_rewards_history": total_rewards_history,
            "eval_rewards_history": eval_rewards_history,
        }
        with open(agent_path, "w") as f:
            json.dump(model_dict, f)
        pass

    @classmethod
    def load_agent(cls, path: str) -> tuple[DeepAgent, dict]:
        """
        load the agent from file
        it will load from two files:
        - path + ".tfm" : tensorflow model
        - path + ".json" : agent parameters
        Args:
            path (str): file path
        
        Returns:    
            tuple[DeepAgent, dict]: agent, last run parameters
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
        instance._model = _sequential_model
        return instance, model_dict

    @classmethod
    def check_agent_loadable_from_path(cls, path:str) -> bool:
        """
        Check if the agent loadable from path
        it will check two file paths:
        - path + ".tfm" : tensorflow model path
        - path + ".json" : agent parameters in a json file
        Args:
            path (str): file path
        
        Returns:
            bool: boolean - true likely loadable; false not loadable
        """
        tensorflow_model_path = path + ".tfm"
        agent_path = path + ".json"

        if not os.path.exists(tensorflow_model_path) or not os.path.isdir(tensorflow_model_path):
            logger.warning(f"Tensorflow model path {tensorflow_model_path} not found")
            return False
        
        if not os.path.exists(agent_path) or not os.path.isfile(agent_path):
            logger.warning(f"Tensorflow model training meta data not exists: {agent_path}")
            return False
        return True

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
        for i, layer in enumerate(structure.process_layers):
            model.add(
                keras.layers.Dense(
                    units=layer.units,
                    activation=layer.activation,
                    kernel_initializer=structure.initializer,
                )
            )
            model.add(
                keras.layers.Dropout(0.2, name=f'layers_{i}_dropout'),
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
        self._model.set_weights(other.model.get_weights())

    def predict(self, state: np.ndarray) -> Any:
        """_summary_

        Args:
            state (np.ndarray): column based vector (N,1) in shape

        Returns:
            Any: action space with Q values (2,1)
        """
        return self._model.predict(state.reshape(1, state.shape[0])).flatten()

    def predict_batch(self, states: np.ndarray) -> Any:
        """_summary_

        Args:
            states (np.ndarray): column based vector (M,N) in shape

        Returns:
            Any: action space with Q values (2,N)
        """
        return self._model.predict(states)

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

    def play(self, env: Execute_Environment, max_step: int, epsilon:float=0, is_exploit_only=False) -> tuple[float, bool]:
        """
        Play the game with the agent
        
        Args:
            env (Execute_Environment): Execute environment
            max_step (int): maximum step to play
            epsilon (float, optional): epsilon value for exploration (0,1). Smaller value mean less likely exploration. Defaults to 0.
            is_exploit_only (bool, optional): exploit only. Defaults to False.
            
        Returns:
            tuple[float, bool]: total reward, is_complete
        """

        state, _ = env.reset()
        total_reward = 0
        terminated: bool = False
        step: int = 1
        COMPLETE: bool = False
        logger.info("Start playing with max_step: %s", max_step)
        while step <= max_step or (not terminated):
        #for step in range(1, max_step + 1):
            if self.is_verbose:
                env.render()
                time.sleep(0.2)
            # Get the action
            action = self.epsilon_greedy(
                env=env, state=state, epsilon=epsilon, is_exploit_only=is_exploit_only
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
            step += 1
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
    def create_new_deep_agent(
        dnn_structure:SequentialStructure, 
        learning_rate:float, 
        discount_factor:float, 
        is_verbose:bool=False) -> DeepAgent:
        """ Create new sequential structur
        """
        return  DeepAgent(
            structure=dnn_structure,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            is_verbose=is_verbose,
        )
    
    @staticmethod
    def load_existing_agent(
        model_path:str
    ) -> tuple[DeepAgent, int, float]:
        """ Load existing agent
        
        Args:
            model_path (str): model path
            
        Returns:
            tuple[DeepAgent, int, float, list[float] ]: agent, last episode, last epsilon, total_rewards_history, last_eval_rewards_history
        """
        cloned_agent, last_run_para = DeepAgent.load_agent(path=model_path)
        episode = last_run_para["episode"]
        epsilon = last_run_para["epsilon"]
        total_rewards_history = last_run_para["total_rewards_history"]
        last_eval_rewards_history = last_run_para["eval_rewards_history"]

        return cloned_agent, episode, epsilon, total_rewards_history, last_eval_rewards_history

    @staticmethod
    def check_agent_reloadable(model_path:str) -> bool:
        """ Check agent loadable in the path
        Args:
            model_path (str): model path
        
        Returns:
            boolean: True -> probably loadable; False -> NOT loadable
        """
        return DeepAgent.check_agent_loadable_from_path(path=model_path)

    @classmethod
    def create_model_path_root(cls, agent_params:Agent_Params, model_name:str, run_id:str) -> str:
        return os.path.join(
            agent_params.savemodel_folder, run_id, model_name
        )
    
    @classmethod
    def create_model_path_fit_log_dir(cls, agent_params:Agent_Params, model_name:str, run_id:str, episode:int) -> str:
        path_str:str = cls.create_model_path_root(
            agent_params=agent_params, model_name=model_name, run_id=run_id
        )
        log_dir = f'logs/fit/epoch{episode}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        return os.path.join(path_str, log_dir)

    @classmethod
    def do_eval(cls, episode:int, eval_env:Execute_Environment, agent:DeepAgent, max_step_allowed:int, min_epsilon:float) -> tuple[dict, float]:
        eval_env_itr = iter(eval_env)
        #Iterate the eval env to run scenario
        _env:Execute_Environment = None
        eval_reward_lst = []
        measure_result_lst = []
        for _env in eval_env_itr:
            eval_reward, _ = agent.play(
                env=_env,
                max_step=max_step_allowed,
                epsilon=min_epsilon,
                is_exploit_only=False
            )
            eval_reward_lst.append(eval_reward)
            #calcualte sharpe ratio
            measure_result_lst.append(_env.measure_result)
            pass
        #Summarize eval_result
        measure_result = np.percentile(measure_result_lst, 10)
        return {
            "episode": episode,
            "10th_percentile_reward" : np.percentile(eval_reward_lst, 10),
            "10th_percentile_measure" : np.percentile(measure_result_lst, 10),
            "median_reward": np.median(eval_reward_lst),
            "median_measure_outcome": np.median(measure_result_lst),
            "90th_percentile_reward" : np.percentile(eval_reward_lst, 90),
            "90th_percentile_measure" : np.percentile(measure_result_lst, 90)
        }, measure_result
        

    @classmethod
    def train(
        cls,
        train_env: Execute_Environment,
        agent_params: Agent_Params,
        train_env_params: BaseEnv_Params,
        dnn_structure: SequentialStructure|str,
        model_name: str,
        run_id:str,
        is_verbose: bool = False,
        eval_env: Execute_Environment = None
    ) -> dict[str, DeepAgent]:
        """ training the model in batch

        Args:
            train_env (Execute_Environment): execution environment with training data
            agent_params (Agent_Params): agent parameters
            train_env_params (BaseEnv_Params): environment parameters
            dnn_structure (SequentialStructure|str): sequential structure or model path
            model_name (str, optional): model name of the environment.
            is_verbose (bool, optional): render the environment during training. Defaults to False.
            

        Returns:
            dict[str,DeepAgent]: dictionary of agents -> {"main": main, "episode": episode, "epsilon": epsilon, "eval_rewards_history": eval_rewards_history}
        """
        epsilon = (
            agent_params.start_epsilon
        )  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        max_epsilon = (
            agent_params.start_epsilon
        )  # You can't explore more than 100% of the time
        min_epsilon = (
            agent_params.min_epsilon
        )  # At a minimum, we'll always explore 1% of the time
        decay = agent_params.decay_rate

        model_path: str = Reinforcement_DeepLearning.create_model_path_root(
            agent_params=agent_params, model_name=model_name, run_id=run_id
        )

        every_n_steps_to_train_main_model = (
            agent_params.every_n_steps_to_train_main_model
        )
        every_m_steps_to_copy_main_weights_to_target_model = (
            agent_params.every_m_steps_to_copy_main_weights_to_target_model
        )
        train_batch_size = agent_params.train_batch_size
        min_replay_size = agent_params.min_replay_size

        learning_rate = agent_params.learning_rate
        discount_factor = agent_params.gamma

        #Support early dropout by saving best model
        save_best_only:bool = agent_params.save_best_only
        worse_than_best_reward_count_limit:int = agent_params.worse_than_best_reward_count_limit

        episode:int = 1
        total_training_rewards_history = []
        eval_rewards_history = []
        
        if isinstance(dnn_structure, str):
            # Overwriting episode and epsilon here after loading mode
            #Load the model from a file directory in dnn_structure
            
            logger.info(f"Load existing {model_name} model from {dnn_structure}")
            main, _episode, _epsilon, _reward_history, _eval_history = Reinforcement_DeepLearning.load_existing_agent(
                model_path=dnn_structure
            )

            target, _, _, _ ,_ = Reinforcement_DeepLearning.load_existing_agent(
                model_path=dnn_structure
            )
            logger.info(
                "Load existing model from %s, episode: %s, epsilon: %s",
                dnn_structure,
                _episode,
                _epsilon,
            )
            episode = 1 + _episode
            epsilon = _epsilon
            total_training_rewards_history.extend(_reward_history)
            eval_rewards_history.extend(_eval_history)
        else:
            # 1a. initialize the main model, (updated every "every_n_steps_to_train_main_model" steps)
            main = Reinforcement_DeepLearning.create_new_deep_agent(
                dnn_structure=dnn_structure,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                is_verbose=is_verbose,
            )
            # 1b. initialize the target model, (updated every "every_m_steps_to_copy_main_weights_to_target_model" steps)
            target = Reinforcement_DeepLearning.create_new_deep_agent(
                dnn_structure=dnn_structure,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                is_verbose=False,
            )
        target.copy_weights(main)

        replay_memory_list = deque(maxlen=agent_params.replay_memory_size)

        steps_to_update_target_model: int = 0

        total_episodes = train_env_params.total_episodes
        episode_batch = train_env_params.episode_batch
        max_steps_allowed = train_env_params.n_max_steps
        best_result:float = -float('inf')
        measure_result:float = -float('inf')
        worse_than_best_reward_count:int = 0
        end_iteration:int = min(total_episodes+1, episode + episode_batch) if train_env_params.batch_mode else total_episodes+1

        logger.info(f"Start training {model_name} model iteration {episode} to {end_iteration-1}")

        for episode in range(episode, end_iteration):
            total_training_rewards: float = 0
            step_count:int = 0
            state, _ = train_env.reset()
            terminated: bool = False
            logger.info(f"Batch Training Episode: {episode}/{total_episodes}")

            while not terminated:
                logger.debug(f"training at {step_count} step")
                step_count += 1
                if max_steps_allowed!=0 and step_count >= max_steps_allowed:
                    logger.info(f"Reach max step allowed:{step_count}")
                    break
                steps_to_update_target_model += 1
                if is_verbose:
                    train_env.render()

                # 2. Explore with Epsilon Greedy exploration
                action = main.epsilon_greedy(env=train_env, state=state, epsilon=epsilon)

                next_state, reward, terminated, truncated, info = train_env.step(
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


                        training_fig_log:str = (cls.create_model_path_fit_log_dir(
                            agent_params=agent_params,
                            model_name=model_name,
                            run_id=run_id,
                            episode=episode
                        ) if episode % agent_params.save_agent_every_n_episode == 0 else None)

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
                            training_fit_log=training_fig_log,
                            validation_split=agent_params.validation_split,
                            training_epoch=agent_params.dnn_training_epoch
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
            total_training_rewards_history.append(total_training_rewards)
            # Save the model every n episodes
            if episode % agent_params.save_agent_every_n_episode == 0:
                logger.info(f"Run the evaluation at episode {episode}")
                #Do the evaluation
                if eval_env is not None:
                    eval_result, measure_result = cls.do_eval(
                            episode=episode,
                            eval_env=eval_env,
                            agent=main,
                            max_step_allowed=max_steps_allowed,
                            min_epsilon=min_epsilon
                        )
                    eval_rewards_history.append(
                        eval_result
                    )
                    

                    
                main.save_agent(
                    path=f"{model_path}_{episode}",
                    episode=episode,
                    epsilon=epsilon,
                    total_rewards_history=total_training_rewards_history,
                    eval_rewards_history=eval_rewards_history
                )
                main.save_agent(
                    path=f"{model_path}_latest",
                    episode=episode,
                    epsilon=epsilon,
                    total_rewards_history=total_training_rewards_history,
                    eval_rewards_history=eval_rewards_history
                )

            #4. check best reward
            if save_best_only:
                if best_result < measure_result:
                    logger.info(f"Save best result at episode {episode}")
                    best_result = measure_result
                    worse_than_best_reward_count = 0
                    #update the best reward model
                    main.save_agent(
                        path=f"{model_path}_best",
                        episode=episode,
                        epsilon=epsilon,
                        total_rewards_history=total_training_rewards_history,
                        eval_rewards_history=eval_rewards_history
                    )
                else:
                    if (worse_than_best_reward_count < worse_than_best_reward_count_limit):
                        worse_than_best_reward_count += 1
                    else:
                        logger.info(f"Reach worse_than_best_reward_count_limit:{worse_than_best_reward_count_limit}")
                        break
            pass
            
        return {
                "main": main, 
                "episode": episode, 
                "epsilon": epsilon, 
                "total_rewards_history": total_training_rewards_history,
                "eval_rewards_history": eval_rewards_history
                }

    @classmethod
    def _train_main_model(
        cls,
        main: DeepAgent,
        target: DeepAgent,
        mini_batch: list[list],
        current_states: np.array,
        next_states: np.array,
        learning_rate: float,
        discount_factor: float,
        validation_split:float = 0,
        training_epoch:int = 10,
        training_fit_log:str = None
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
            validation_split(float): fraction of data for validation
            training_epoch (int): training epoch
            training_fit_log (str): training fit log

        Returns:
            DeepAgent: _description_
        """

        current_qs_list = main.predict_batch(current_states)
        future_qs_list = target.predict_batch(next_states)

        callbacks = None

        if training_fit_log is not None:
            #Tensorboard
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=training_fit_log, 
            histogram_freq=1)
            callbacks = [tensorboard_callback]
        

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
        logger.info(f"fit model validation_split{validation_split} training_fit_log{training_fit_log}")
        main.model.fit(
            np.array(X), np.array(Y), 
            batch_size=len(X), 
            epochs=training_epoch, 
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0)
        return main
