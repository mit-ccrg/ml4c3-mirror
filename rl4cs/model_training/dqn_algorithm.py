# Imports: standard library
import math
import random
from typing import List, Optional

# Imports: third party
import numpy as np
import tensorflow as tf

# Imports: first party
from rl4cs.model_training.memory import Memory
from rl4cs.model_training.nn_model import NNModel
from rl4cs.environments.cv_env_typed import CVEnvTyped

# pylint: disable=unused-variable,no-member


class DQNAlgorithm:
    """
    Deep Q Network algorithm for training a policy.
    """

    def __init__(
        self,
        q_nn_model: NNModel,
        q_target: NNModel,
        env: Optional[CVEnvTyped],
        memory: Memory,
        max_eps: float = 1,
        min_eps: float = 0.01,
        decay: float = 0.01,
        discount: float = 0.99,
        reply_start_size: int = 100,
        sync_steps: int = 100,
        dqn_method: str = "unique",
    ):
        """
        Init DQNAlgorithm class.

        :param q_nn_model: <NNModel> Q function with Neural Network model.
        :param q_target: <NNModel> Target Q function with NN model.
        :param env: <CVEnvTyped> Gym environment.
        :param memory: <Memory> Memory to store trajectories for training.
        :param max_eps: <float> Max epsilon for eps-greedy approach of action selection.
        :param min_eps: <float> Min epsilon for eps-greedy approach of action selection.
        :param decay: <float> Decay for eps-greedy approach of action selection.
        :param discount: <float> Discount factor for Q update.
        :param reply_start_size: <int> Initial size of the buffer to start
                                       experience reply.
        :param sync_steps: <int> Number of steps between sync of q_nn and q_target.
        :param dqn_method: <str> Either unique or all. Either taking unique elements
                                 that change from the q function or all in order to
                                 update the policy.
        """
        # Environment and policy
        self.env = env
        self.q_nn_model = q_nn_model
        self.q_target = q_target

        # Hyperparameters related to the DQN
        self.memory = memory
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.decay = decay
        self.discount = discount
        self.reply_start_size = reply_start_size
        self.sync_steps = sync_steps
        self.eps = max_eps

        # Start new variables to count/store
        self.steps = 0
        self.reward_store: List[float] = []
        self.loss_store: List[float] = []

        # Render capability
        self.render = False

        # Choice of dqn method
        if dqn_method not in ("unique", "all"):
            raise ValueError(
                f"DQN method {dqn_method} not supported. Possibilities "
                f"are: unique or all.",
            )
        self.dqn_method = dqn_method

    def run(self):
        """
        Run DQN algorithm.
        """
        states = self.env.reset()
        total_reward = 0
        while True:
            self.steps += 1

            if self.render:
                self.env.render()

            action = self._choose_action(states)
            next_states, reward, done, info = self.env.step(action)

            # If game completed set next state to None for storage sake
            if done:
                next_states = None

            # Add samples to memory
            self.memory.add_sample((states, action, reward, next_states))

            # Syncronize q_nn and q_target
            if self.steps % self.sync_steps == 0:
                self.sync_q_networks()

            # Train Q function (Neural Network model) with experience replay
            # after a certain amount of steps have passed
            loss = self.replay()
            self.loss_store.append(loss)

            # Exponentially decay the eps value (epsilon-greedy)
            # Exploration vs exploitation
            self.eps = self.min_eps + (self.max_eps - self.min_eps) * math.exp(
                -self.decay * self.steps,
            )

            # Move the agent to the next state and accumulate the reward
            states = next_states
            total_reward += reward

            # If done, break the loop
            if done:
                self.reward_store.append(total_reward)
                break

    def sync_q_networks(self):
        """
        Synchronize target q function with trained q function.
        """
        self.q_target.model.set_weights(self.q_nn_model.model.get_weights())

    def _choose_action(self, states: np.ndarray) -> int:
        """
        Choose action either randomly or taking the policy into account.
        (Exploration vs exploitation)

        :param states: <np.ndarray> States (observations) of the RL problem.
        :return: <int> Index of the action.
        """
        if random.random() < self.eps:
            return random.randint(0, self.q_nn_model.num_actions - 1)
        return int(np.argmax(self.q_nn_model.predict_one(states).numpy()))

    def replay(self) -> float:
        """
        Apply experience replay. Take one batch of experiences and train q NN.

        :return: <float> Loss of the NN after one training step.
        """
        if len(self.memory.samples) < self.reply_start_size:
            return -1.0

        # Get batch of experiences
        batch = self.memory.take_samples(int(self.q_nn_model.batch_size))
        states = np.array([val[0] for val in batch])
        next_states = np.array(
            [
                (
                    np.zeros(self.q_nn_model.num_states)
                    if val[3] is None
                    else np.array(val[3])
                )
                for val in batch
            ],
        )
        # Predict several Q(s,a) given the batch of states
        q_s_a = self.q_nn_model.predict_batch(states).numpy()

        # Predict several Q(s', a') given the batch of next states
        q_s_a_next = self.q_target.predict_batch(next_states).numpy()

        # Setup training arrays
        x_imp = np.zeros((len(batch), self.q_nn_model.num_states))
        y_out = np.zeros((len(batch), self.q_nn_model.num_actions))
        if self.dqn_method == "unique":
            y_out = y_out[:, 0]
        actions = np.zeros(len(batch)).astype(int)
        for idx, sample in enumerate(batch):
            states, action, reward, next_states = (
                sample[0],
                sample[1],
                sample[2],
                sample[3],
            )
            # Get the current q values for all actions in state
            current_q = q_s_a[idx]
            # Game completed, only reward is got, otherwise also max Q(s', a')
            if next_states is None:
                current_q[action] = reward
            else:
                current_q[action] = reward + self.discount * np.amax(q_s_a_next[idx])
            x_imp[idx] = states
            if self.dqn_method == "all":
                y_out[idx] = current_q
            if self.dqn_method == "unique":
                y_out[idx] = np.sum(
                    current_q * tf.one_hot(action, self.q_nn_model.num_actions),
                )
            actions[idx] = int(action)

        # Train Q NN model and return its loss
        loss = self.q_nn_model.train_batch(x_imp, y_out, actions)
        return float(loss.numpy())
