import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # Discount factor for future rewards
        self.alpha_actor = 0.001  # Learning rate for the actor
        self.alpha_critic = 0.005  # Learning rate for the critic

        # Memory for experiences
        self.memory = []

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        # Define actor model that maps states to action distributions (mean and standard deviation)
        state_input = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(64, activation='relu')(state_input)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        
        # Output layer for action mean (no scaling here)
        action_mean = layers.Dense(self.action_size, activation='tanh')(dense2)
        
        # Output layer for action standard deviation
        action_std = layers.Dense(self.action_size, activation='sigmoid')(dense2)  # Make std positive from 0 to 1

        model = Model(inputs=state_input, outputs=[action_mean, action_std])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_actor))
        return model

    def build_critic(self):
        # Define critic model that maps (state, action) pairs to their Q-values
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))
        concat = layers.Concatenate()([state_input, action_input])

        dense1 = layers.Dense(64, activation='relu')(concat)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        q_value_output = layers.Dense(1, activation='softplus')(dense2)  # Q-value output

        model = Model(inputs=[state_input, action_input], outputs=q_value_output)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_critic))
        return model

    def act(self, state):
        # Predict action mean and std, sample action
        state = np.reshape(state, [1, self.state_size])
        action_mean, action_std = self.actor.predict(state)
        action_dist = tfp.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        # Cap the action from -1 to 1
        action = np.clip(action, -1, 1)
        return action[0], action_mean[0], action_std[0]

    def learn(self, state, action, reward, next_state, done):
        np_state = np.array([state])
        np_next_state = np.array([next_state])
        np_action = np.array([action])

        # 1. Update critic
        # Correctly predict the next Q-value
        next_value = self.critic.predict([np_next_state, np_action])
        # Calculate target for current state
        target_q_value = reward + (self.gamma * next_value[0] * (1 - done))
        self.critic.train_on_batch([np_state, np_action], target_q_value)

        # 2. Update actor
        with tf.GradientTape() as tape:
            states_tensor = tf.convert_to_tensor(np_state, dtype=tf.float32)  # Ensure the state is a tensor
            # Directly use the actor model for prediction inside the tape to capture gradients
            action_mean, action_std = self.actor(states_tensor)
            action_dist = tfp.distributions.Normal(action_mean, action_std)
            action_pred = action_dist.sample()  # Sample an action from the distribution
            action_pred = tf.clip_by_value(action_pred, -1, 1)  # Ensure actions are within expected range

            # Ensure critic receives tensors with correct shapes
            critic_value = self.critic([states_tensor, action_pred])
            actor_loss = -tf.math.reduce_mean(critic_value)

            # Penalize high std values to encourage the actor to reduce exploration over time
            std_penalty = tf.reduce_mean(action_std)  # Calculate the mean of action standard deviations
            std_weight = 0.0001  # Define the weight of the std penalty term
            actor_loss += std_weight * std_penalty  # Add the std penalty to the actor loss

        # Compute gradients and apply them to the actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss, target_q_value        
