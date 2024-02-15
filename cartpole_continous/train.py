import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from agent import Agent
from environment import Environment, config

filename = "action_log.pkl"

env = Environment()
agent = Agent(state_size=4, action_size=1)

# Logs for additional data
rewards_log = []
episode_rewards_log = []
action_log = []
action_means_log = []
action_stdevs_log = []
actions_per_episode = []

num_episodes = 20000
for episode in range(num_episodes):
    print(f"Training Episode {episode + 1}/{num_episodes}...")
    state = env.reset()
    done = False
    episode_reward = 0

    step = 0
    while not done:
        step += 1

        with tf.GradientTape(persistent=True) as tape:
            action, action_mean, action_std = agent.act(state)
            with tape.stop_recording():
                next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done, tape)
        state = next_state
    
        # Log data
        episode_reward += reward
        action_log.append(action)
        action_means_log.append(action_mean)
        action_stdevs_log.append(action_std)
        rewards_log.append(episode_reward)

        # Print current step, overwriting the same line
        print(f"Step {step} Std {action_std}", end='\r', flush=True)

    print(f"{step} step(s) for episode {episode + 1}, score {episode_reward}")
    episode_rewards_log.append(episode_reward)
    actions_per_episode.append(len(action_log))

print("Done")

with open(filename, 'wb') as file:
    pickle.dump({
        "action_log": action_log,
        "actions_per_episode": actions_per_episode,
        "episode_rewards_log": episode_rewards_log,
    }, file)

def add_plot(ax, index, data, title):
    ax[index].plot(data)
    ax[index].set_title(title)

# Plot action logs in subplots
fig, ax = plt.subplots(4, 1, figsize=(10, 8))  # Adjusted for better visualization
add_plot(ax, 0, episode_rewards_log, "Episode Rewards")
add_plot(ax, 1, action_log, "Action Log")
add_plot(ax, 2, action_means_log, "Action Means Log")
add_plot(ax, 3, action_stdevs_log, "Action Standard Deviations Log")

plt.tight_layout()  # Adjust layout to not overlap plots

plt.savefig("train.png")
plt.show()  # Show plot
