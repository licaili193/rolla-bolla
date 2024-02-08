import matplotlib.pyplot as plt

from agent import Agent
from environment import Environment, config

env = Environment()
agent = Agent(state_size=4, action_size=1)

# Logs for additional data
rewards_log = []
episode_rewards_log = []
action_log = []
action_means_log = []
action_stdevs_log = []
actor_losses_log = []
target_qs_log = []

num_episodes = 20
for episode in range(num_episodes):
    print(f"Training Episode {episode + 1}/{num_episodes}...")
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, action_mean, action_std = agent.act(state)
        next_state, reward, done = env.step(env.scale_action(action))
        actor_loss, target_q = agent.learn(state, action, reward, next_state, done)
        state = next_state
    
        # Log data
        episode_reward += reward
        action_log.append(action)
        action_means_log.append(action_mean)
        action_stdevs_log.append(action_std)
        rewards_log.append(episode_reward)
        actor_losses_log.append(actor_loss)
        target_qs_log.append(target_q)

    episode_rewards_log.append(episode_reward)

print("Done")

# Plot Episode Rewards
plt.figure(figsize=(10, 4))  # Optional: Specify figure size
plt.plot(episode_rewards_log)
plt.title("Episode Rewards")
plt.xlabel("Episode")  # Adding x-axis label
plt.ylabel("Total Reward")  # Adding y-axis label
plt.show()  # Show plot

# Plot action logs in subplots
fig, ax = plt.subplots(4, 1, figsize=(10, 12))  # Adjusted for better visualization
ax[0].plot(action_log)
ax[0].set_title("Action Log")
ax[0].set_xlabel("Step")  # Labeling x-axis
ax[0].set_ylabel("Action Value")  # Labeling y-axis

ax[1].plot(action_means_log)
ax[1].set_title("Action Means Log")
ax[1].set_xlabel("Step")
ax[1].set_ylabel("Mean Action Value")

ax[2].plot(action_stdevs_log)
ax[2].set_title("Action Standard Deviations Log")
ax[2].set_xlabel("Step")
ax[2].set_ylabel("Action Std Dev Value")

ax[3].plot(actor_losses_log)
ax[3].set_title("Actor Losses Log")
ax[3].set_xlabel("Step")
ax[3].set_ylabel("Loss Value")

plt.tight_layout()  # Adjust layout to not overlap plots
plt.show()  # Show plot
