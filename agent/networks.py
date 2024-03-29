import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 

class CriticNetwork(keras.Model):
    """
    A class representing the Critic Network in a reinforcement learning agent.

    Parameters:
    - n_actions (int): Number of possible actions.
    - fc1_dims (int): Number of units in the first fully connected layer (default: 512).
    - fc2_dims (int): Number of units in the second fully connected layer (default: 256).
    - name (str): Name of the network (default: 'critic').
    - chkpt_dir (str): Directory to save checkpoints (default: 'tmp/sac').

    Attributes:
    - fc1_dims (int): Number of units in the first fully connected layer.
    - fc2_dims (int): Number of units in the second fully connected layer.
    - n_actions (int): Number of possible actions.
    - model_name (str): Name of the network.
    - checkpoint_dir (str): Directory to save checkpoints.
    - checkpoint_file (str): File path to save the checkpoint.

    Methods:
    - call(state, action): Forward pass of the network.

    """

    def __init__(self, n_actions, fc1_dims=512, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='leaky_relu')
        self.fc2 = Dense(self.fc2_dims, activation='leaky_relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        """
        Forward pass of the Critic Network.

        Parameters:
        - state (tf.Tensor): Input state tensor.
        - action (tf.Tensor): Input action tensor.

        Returns:
        - q (tf.Tensor): Output Q-value tensor.

        """
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ValueNetwork(keras.Model):
    """
    ValueNetwork class represents a value network model used in reinforcement learning algorithms.

    Args:
        fc1_dims (int): Number of units in the first fully connected layer. Default is 512.
        fc2_dims (int): Number of units in the second fully connected layer. Default is 256.
        name (str): Name of the model. Default is 'value'.
        chkpt_dir (str): Directory to save model checkpoints. Default is 'tmp/sac'.

    Attributes:
        fc1_dims (int): Number of units in the first fully connected layer.
        fc2_dims (int): Number of units in the second fully connected layer.
        model_name (str): Name of the model.
        checkpoint_dir (str): Directory to save model checkpoints.
        checkpoint_file (str): File path to save model checkpoints.

    Methods:
        call(state): Forward pass of the value network.

    """

    def __init__(self, fc1_dims=512, fc2_dims=256,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='leaky_relu')
        self.fc2 = Dense(fc2_dims, activation='leaky_relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        """
        Forward pass of the value network.

        Args:
            state (tf.Tensor): Input state tensor.

        Returns:
            tf.Tensor: Value estimation for the input state.

        """
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

class ActorNetwork(keras.Model):
    """
    The ActorNetwork class represents an actor network used in the Soft Actor-Critic (SAC) algorithm.

    Parameters:
    - max_action (float): The maximum value of the action.
    - fc1_dims (int): The number of units in the first fully connected layer. Default is 512.
    - fc2_dims (int): The number of units in the second fully connected layer. Default is 256.
    - n_actions (int): The number of actions. Default is 2.
    - name (str): The name of the actor network. Default is 'actor'.
    - chkpt_dir (str): The directory to save checkpoints. Default is 'tmp/sac'.

    Attributes:
    - fc1 (Dense): The first fully connected layer.
    - fc2 (Dense): The second fully connected layer.
    - mu (Dense): The mean output layer.
    - sigma (Dense): The standard deviation output layer.

    Methods:
    - call(state): Forward pass of the actor network.
    - sample_normal(state): Sample an action from a normal distribution.

    """

    def __init__(self, max_action, fc1_dims=512, fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation='leaky_relu')
        self.fc2 = Dense(self.fc2_dims, activation='leaky_relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation='softplus')

    def call(self, state):
        """
        Forward pass of the actor network.

        Parameters:
        - state (tf.Tensor): The input state.

        Returns:
        - mu (tf.Tensor): The mean of the action distribution.
        - sigma (tf.Tensor): The standard deviation of the action distribution.

        """
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = tf.math.add(sigma, self.noise)

        return mu, sigma

    def sample_normal(self, state):
        """
        Sample an action from a normal distribution.

        Parameters:
        - state (tf.Tensor): The input state.

        Returns:
        - action (tf.Tensor): The sampled action.
        - log_probs (tf.Tensor): The log probabilities of the sampled action.

        """
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        actions = probabilities.sample()

        action = tf.math.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        action = tf.math.multiply(action, self.max_action)

        return action, log_probs
