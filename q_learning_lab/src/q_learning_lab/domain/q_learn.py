from __future__ import annotations
from ..port.environment import Execute_Environment
from .frozen_lake.models import Action_Space
from ..port.environment import Execute_Environment
from ..utility.logging import get_logger

import time
import numpy as np

logger = get_logger(name=__name__)


# Reference: https://toxwardsdatascience.com/q-learning-algorithm-how-to-successfully-teach-an-intelligent-agent-to-play-a-game-933595fd1abf
# https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#sphx-glr-tutorials-training-agents-frozenlake-tuto-py


def create_q_table(env: Execute_Environment) -> np.ndarray:
    return np.zeros((env.observation_space_dim, env.action_space_dim))


class Agent:
    def __init__(
        self, learning_rate: float, discount_rate: float, is_verbose: bool = False
    ) -> None:
        self.is_verbose: bool = is_verbose
        self.learning_rate: float = learning_rate
        self.discounting_rate: float = discount_rate
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

    def execute(self, env: Execute_Environment, max_steps: int, Qtable) -> float:
        """execute the agent

        Args:
            env (Execute_Environment): environment for execution
            max_steps (int): max step per episode
            Qtable (_type_): _description_

        Returns:
            float: total reward
        """
        state, info = env.reset()
        total_episode_reward = 0
        for step in range(max_steps):
            logger.debug(
                "execution step: %s, total_episode_reward:%s",
                step,
                total_episode_reward,
            )
            if self.is_verbose:
                env.render()
            # use greedy policy to evaluate
            action = self.epsilon_greedy(
                Qtable=Qtable, env=env, state=state, epsilon=0.0, is_exploit_only=True
            )

            # Pass action into step function
            observation_space, reward, terminated, truncated, info = env.step(
                action=action
            )
            # Sum episode reward
            total_episode_reward += reward

            # Update current state
            state = observation_space

            # Wait a little bit before the next frame
            if self.is_verbose:
                time.sleep(0.2)
            if terminated:
                if self.is_verbose:
                    env.render()
                break
        return total_episode_reward

    def evaluate_agent(self, n_max_steps, n_eval_episodes, env, Qtable) -> float:
        """evaluate the agent

        Args:
            n_max_steps (int): max step per episode
            n_eval_episodes (int): number of episode
            env (Execute_Environment): environment for execution
            Qtable (_type_): _description_

        Returns:
            float: average reward
        """
        episode_rewards = []
        for episode in range(n_eval_episodes):
            episode_reward = self.execute(env=env, max_steps=n_max_steps, Qtable=Qtable)
            episode_rewards.append(episode_reward)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    def train(
        self,
        env: Execute_Environment,
        n_episodes,
        n_max_steps,
        start_epsilon,
        min_epsilon,
        decay_rate,
    ) -> None:
        """train the agent

        Args:
            env (Execute_Environment): environment for execution
            n_episodes (_type_): number of episode
            n_max_steps (_type_): max step per episode
            start_epsilon (_type_): start epsilon
            min_epsilon (_type_): min epsilon
            decay_rate (_type_): decay rate of epsilon
        """
        # Create qtable from env
        Qtable = create_q_table(env=env)

        # iterate each episode
        for episode in range(n_episodes):
            # Reset the environment at the start of each episode
            state, info = env.reset()
            # Set the termination flag to false

            # calculate the epsilon value based on decay rate
            epsilon = max(
                min_epsilon,
                (start_epsilon) * np.exp(-decay_rate * episode),
            )
            for step in range(n_max_steps):
                logger.debug(f"Episode {episode}: step {step}")
                # Select an action using the epsilon greedy policy
                action: int = self.epsilon_greedy(
                    Qtable=Qtable,
                    env=env,
                    state=state,
                    epsilon=epsilon,
                )
                # Execute the action and get the reward and the next state
                next_state, reward, terminated, truncated, info = env.step(action)

                # Update the Q table
                Qtable = self.update_Q(
                    Qtable=Qtable,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                )
                # Update the state
                state = next_state
                if terminated:
                    break
        # return the final table
        return Qtable

    def epsilon_greedy(
        self,
        Qtable,
        env: Execute_Environment,
        state: int,
        epsilon: float,
        is_exploit_only: bool = False,
    ) -> int:
        """This is our acting policy (epsilon-greedy), for the agent to do exploration and exploitation during training

        Args:
            Qtable (_type_): _description_
            env (Execute_Environment): _description_
            state (int): _description_
            epsilon (float): _description_
            is_exploit_only (bool, optional): only exploit and ignore epsilon. Defaults to False.

        Returns:
            int: _description_
        """

        def _exploit(Qtable, state):
            return np.argmax(Qtable[state, :])

        # Generate a random number and compare to epsilon, if lower then explore, otherwuse exploit
        if not is_exploit_only:
            randnum = np.random.uniform(0, 1)
            if randnum < epsilon:
                action = int(env.sample_action_space())  # explore
            else:
                action = _exploit(Qtable=Qtable, state=state)  # exploit
        else:
            action = _exploit(Qtable=Qtable, state=state)  # exploit
        return action

    # This is our updating policy (greedy)
    # i.e., always select the action with the highest value for that state: np.max(Qtable[next_state])
    def update_Q(self, Qtable, state, action, reward, next_state):
        """This is our updating policy (greedy)
            i.e., always select the action with the highest value for that state: np.max(Qtable[next_state])
            Q(S_t,A_t) = Q(S_t,A_t) + alpha [R_t+1 + gamma * max Q(S_t+1,a) - Q(S_t,A_t)]

        Args:
            Qtable (_type_): _description_
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        Qtable[state][action] = Qtable[state][action] + self.learning_rate * (
            reward
            + self.discounting_rate * np.max(Qtable[next_state])
            - Qtable[state][action]
        )
        return Qtable
