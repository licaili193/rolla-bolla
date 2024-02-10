import matplotlib.pyplot as plt
import pickle

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
losses_log = []

num_episodes = 50
for episode in range(num_episodes):
    print(f"Training Episode {episode + 1}/{num_episodes}...")
    state = env.reset()
    done = False
    episode_reward = 0

    step = 0
    while not done:
        step += 1

        action, action_mean, action_std = agent.act(state)
        next_state, reward, done = env.step(action)
        loss = agent.learn(state, action, reward, next_state, done)
        state = next_state
    
        # Log data
        episode_reward += reward
        action_log.append(action)
        action_means_log.append(action_mean)
        action_stdevs_log.append(action_std)
        rewards_log.append(episode_reward)
        losses_log.append(loss)

        # Print current step, overwriting the same line
        print(f"Step {step}", end='\r', flush=True)

    print(f"{step} step(s) for episode {episode + 1}, score {episode_reward}")
    episode_rewards_log.append(episode_reward)

print("Done")

with open(filename, 'wb') as file:
    pickle.dump(action_log, file)

def add_plot(ax, index, data, title):
    ax[index].plot(data)
    ax[index].set_title(title)

# Plot action logs in subplots
fig, ax = plt.subplots(5, 1, figsize=(10, 12))  # Adjusted for better visualization
add_plot(ax, 0, episode_rewards_log, "Episode Rewards")
add_plot(ax, 1, action_log, "Action Log")
add_plot(ax, 2, action_means_log, "Action Means Log")
add_plot(ax, 3, action_stdevs_log, "Action Standard Deviations Log")
add_plot(ax, 4, losses_log, "Losses Log")

plt.tight_layout()  # Adjust layout to not overlap plots
plt.show()  # Show plot
