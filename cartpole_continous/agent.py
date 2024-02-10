import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9  # Discount factor for future rewards
        self.alpha_actor = 0.0025  # Learning rate for the actor
        self.alpha_critic = 0.005  # Learning rate for the critic

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        # Define actor model that maps states to action distributions (mean and standard deviation)
        state_input = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(64, activation='relu')(state_input)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        
        # Output layer for action mean (no scaling here)
        action_mean = layers.Dense(self.action_size, activation=None)(dense2)
        
        # Output layer for action standard deviation
        epsilon = 1e-5  # Small constant to ensure standard deviation is never zero
        action_std = layers.Dense(self.action_size, activation='softplus')(dense2)
        action_std = layers.Lambda(lambda x: x + epsilon)(action_std)

        model = Model(inputs=state_input, outputs=[action_mean, action_std])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_actor))
        return model

    def build_critic(self):
        # Define critic model that maps (state, action) pairs to their Q-values
        state_input = layers.Input(shape=(self.state_size,))

        dense1 = layers.Dense(64, activation='relu')(state_input)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        q_value_output = layers.Dense(1, activation=None)(dense2)  # Q-value output

        model = Model(inputs=state_input, outputs=q_value_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_critic))
        return model

    def act(self, state):
        # Predict action mean and std, sample action
        state = np.reshape(state, [1, self.state_size])
        action_mean, action_std = self.actor.predict(state, verbose=0)
        action_dist = tfp.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        return tf.squeeze(action).numpy(), tf.squeeze(action_mean).numpy(), tf.squeeze(action_std).numpy()

    def learn(self, state, action, reward, next_state, done):
        with tf.GradientTape(persistent=True) as tape:
            state_tensor = tf.convert_to_tensor(state.reshape(1, self.state_size))
            next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, self.state_size))

            # Predict the future and current Q-values
            next_q_value = self.critic(next_state_tensor)
            q_value = self.critic(state_tensor)
            
            # Compute the target and the advantage (delta)
            delta = reward + self.gamma * next_q_value * (1 - done) - q_value

            # Predict action mean and std using the actor model
            action_mean, action_std = self.actor(state_tensor)
            action_dist = tfp.distributions.Normal(action_mean, action_std)

            # Calculate the critic loss as the mean squared error of delta
            critic_loss = delta ** 2
            # Calculate the actor loss
            actor_loss = -action_dist.log_prob(action) * delta

            loss = tf.reduce_mean(critic_loss + actor_loss)

        # Compute gradients and apply them
        actor_grads = tape.gradient(loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(loss, self.critic.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # return loss
        return loss
