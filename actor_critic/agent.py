import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

config = {
    "gamma": 0.99,
    "alpha": 0.01,
    "network_size": 128,
}

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]

        self.env = env

        self.network = self.build_network()
    
    def build_network(self):
        state_input = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(config["network_size"], activation='relu')(state_input)

        action_probs = layers.Dense(self.action_size, activation='softmax')(dense1)
        value = layers.Dense(1, activation=None)(dense1)

        model = Model(inputs=state_input, outputs=[action_probs, value])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss="mean_squared_error")
        return model
    
    def act(self, state):
        state_tensor = tf.convert_to_tensor(state.reshape(1, self.state_size))
        action_probs, value = self.network(state_tensor)
        action_dist = tfp.distributions.Categorical(probs=action_probs)
        action = action_dist.sample()
        action = int(tf.squeeze(action).numpy())
        value = tf.squeeze(value).numpy()
        return action, float(action_probs[0, action]), value
    
    def run_episode(self, max_steps):
        state = self.env.reset()
        state = state[0]
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for step in tf.range(max_steps):
            state_tensor = tf.convert_to_tensor(state.reshape(1, self.state_size))
            action_prob, value = self.network(state_tensor)
            action_log_probs = tf.math.log(action_prob + 1e-5)
            action = tf.random.categorical(action_log_probs, 1)[0, 0]
            value = tf.squeeze(value).numpy()
            chosen_action_prob = tf.gather(action_prob[0], action)
            next_state, reward, done, _, _ = self.env.step(int(action.numpy()))

            rewards = rewards.write(step, reward)
            values = values.write(step, value)
            action_probs = action_probs.write(step, chosen_action_prob)

            state = next_state
            if done:
                break

        return (
            rewards.stack(),
            action_probs.stack(),
            values.stack(),
        )
    
    def compute_expected_return(self, rewards, standardize=True):
      expected_return = []
      cumulative_return = 0
      for i in range(len(rewards)):
        cumulative_return = rewards[i] + self.gamma * cumulative_return
        expected_return.append(cumulative_return)
      expected_return.reverse()
      
      if standardize:
          returns = np.array(expected_return)
          eps = 1e-5
          returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
          return returns.tolist()
      else:
          return expected_return
      
    def compute_loss(self, action_probs, values, returns):
        advantage = returns - values
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = huber_loss(values, returns)
        return actor_loss + critic_loss
    
    def train_episode(self, max_steps_per_episode):
        with tf.GradientTape() as tape:
            rewards, action_probs, values = self.run_episode(max_steps_per_episode)
            returns = self.compute_expected_return(rewards)
            loss = self.compute_loss(tf.convert_to_tensor(action_probs), tf.convert_to_tensor(values), tf.convert_to_tensor(returns))

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return sum(rewards)
