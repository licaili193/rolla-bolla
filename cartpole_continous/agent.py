import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import HeNormal
import numpy as np
import tensorflow_probability as tfp

config = {
    "gamma": 0.99,
    "alpha_actor": 0.0000002,
    "alpha_critic": 0.000004,
    "network_size": 128,
    "std_epsilon": 1e-5,
}

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = config["gamma"]  # Discount factor for future rewards
        self.alpha_actor = config["alpha_actor"]  # Learning rate for the actor
        self.alpha_critic = config["alpha_critic"]  # Learning rate for the critic

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        # Define actor model that maps states to action distributions (mean and standard deviation)
        state_input = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(config["network_size"], activation='relu', kernel_initializer=HeNormal())(state_input)
        dense2 = layers.Dense(config["network_size"], activation='relu', kernel_initializer=HeNormal())(dense1)
        
        # Output layer for action mean (no scaling here)
        action_mean = layers.Dense(self.action_size, activation=None)(dense2)
        
        # Output layer for action standard deviation
        epsilon = config["std_epsilon"]  # Small constant to ensure standard deviation is never zero
        action_std = layers.Dense(self.action_size, activation='softplus')(dense2)
        action_std = layers.Lambda(lambda x: x + epsilon)(action_std)

        model = Model(inputs=state_input, outputs=[action_mean, action_std])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_actor))
        return model

    def build_critic(self):
        # Define critic model that maps (state, action) pairs to their Q-values
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))

        # Concatenate state and action inputs
        concat_input = layers.Concatenate()([state_input, action_input])

        dense1 = layers.Dense(config["network_size"], activation='relu', kernel_initializer=HeNormal())(concat_input)
        dense2 = layers.Dense(config["network_size"], activation='relu', kernel_initializer=HeNormal())(dense1)
        q_value_output = layers.Dense(1, activation=None)(dense2)  # Q-value output

        model = Model(inputs=[state_input, action_input], outputs=q_value_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_critic))
        return model

    def act(self, state):
        # Predict action mean and std, sample action
        state_tensor = tf.convert_to_tensor(state.reshape(1, self.state_size))
        action_mean, action_std = self.actor(state_tensor)
        action_dist = tfp.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        self.log_prob = action_dist.log_prob(action)
        return tf.squeeze(action).numpy(), tf.squeeze(action_mean).numpy(), tf.squeeze(action_std).numpy()

    def learn(self, state, action, reward, next_state, done, tape):
        state_tensor = tf.convert_to_tensor(state.reshape(1, self.state_size))
        next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, self.state_size))
        action_tensor = tf.convert_to_tensor(action.reshape(1, self.action_size))

        # Predict the future and current Q-values with action included
        next_q_value = self.critic([next_state_tensor, action_tensor])
        q_value = self.critic([state_tensor, action_tensor])
        
        # Compute the target and the advantage (delta)
        delta = reward + self.gamma * next_q_value * (1 - done) - q_value

        # Calculate the critic loss as the mean squared error of delta
        critic_loss = tf.reduce_mean(delta ** 2)
        # Calculate the actor loss
        actor_loss = -tf.reduce_mean(self.log_prob * delta)

        # Compute gradients and apply them
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))