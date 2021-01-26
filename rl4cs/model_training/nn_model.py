# Imports: standard library
from typing import List, Optional

# Imports: third party
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, losses, optimizers


class NNModel:
    """
    Neural network model for DQN RL algorithm.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        batch_size: int = 1,
        hidden_arch: Optional[List] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        """
        Init NN class.

        :param num_states: <int> Number of states of the RL problem.
        :param num_actions: <int> Number of actions of the RL problem.
        :param batch_size: <int> Batch of samples to use during training.
        :param hidden_arch: <List> Hidden layers and neurons.
        :param activation: <str> Activation function for neurons.
        :param learning_rate: <float> Learning rate of neural network training.
        :param beta1: <float> beta1 parameter of Adam optimizer.
        :param beta2: <float> beta2 parameter of Adam optimizer.
        """
        # Customizable params
        if hidden_arch is None:
            hidden_arch = [1, 1]
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.hidden_arch = hidden_arch

        if activation == "relu":
            self.activation = tf.nn.relu
        elif activation == "leaky_relu":
            self.activation = tf.nn.leaky_relu
        elif activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        elif activation == "elu":
            self.activation = tf.nn.elu
        elif activation == "tanh":
            self.activation = tf.nn.tanh
        else:
            raise ValueError(
                "This activation does not exist, available are:"
                "relu, elu, tanh, leaky_relu and sigmoid",
            )

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        # Model
        self.model = Model()

        # Operations
        self.loss = None
        self.optimizer = None

        # Setup the model and functions
        self._define_model_functions()

    def _define_model_functions(self):
        """
        Define NN model for DQN training.
        """
        # Input of neurons (Batch size x Number of states)
        states = Input(shape=(self.num_states,), dtype=tf.float32, name="states")

        # Hidden layers
        layer_1 = layers.Dense(self.hidden_arch[0], activation=self.activation)(states)
        layers_n = [None for _ in range(len(self.hidden_arch))]
        layers_n[0] = layer_1
        for idx, n_neurons in enumerate(self.hidden_arch[1:]):
            layers_n[idx + 1] = layers.Dense(
                n_neurons,
                activation=self.activation,
            )(layers_n[idx])

        # Output of neurons is q(s, a) function
        q_s_a = layers.Dense(self.num_actions, name="q_s_a")(layers_n[-1])

        # Get the model
        self.model = Model(inputs=states, outputs=q_s_a)

        # Loss function and optimizer
        self.loss = losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

        self.optimizer = optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
            name="Adam",
        )

    @tf.function
    def predict_one(self, state: np.ndarray):
        """
        Make prediction of one sample.

        :param state: <np.ndarray> Input state.
        :return: <tf.python.framework.ops.EagerTensor> Result tensor.
        """
        return self.model(tf.expand_dims(state, axis=0), training=False)

    @tf.function
    def predict_batch(self, states: np.ndarray):
        """
        Make prediction of batch of samples.

        :param states: <np.ndarray> Input batch of states.
        :return: <tf.python.framework.ops.EagerTensor> Run session.
        """
        return self.model(states, training=False)

    @tf.function
    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray):
        """
        Train NN model.

        :param x_batch: <np.ndarray> Input batch of states.
        :param y_batch: <np.ndarray> Output batch of actions.
        :return: <tf.python.framework.ops.EagerTensor> Loss.
        """
        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            loss_value = self.loss(y_batch, logits)  # type: ignore

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(  # type: ignore
            zip(grads, self.model.trainable_weights),
        )
        return loss_value
