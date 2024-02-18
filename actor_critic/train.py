# Reference: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic

import matplotlib.pyplot as plt
from agent import ActorCriticAgent

from environment import Environment, config
env = Environment()

# import gym
# env = gym.make("CartPole-v1")

agent = ActorCriticAgent(state_size=4, action_size=2, env=env)

scores = []

max_episodes = 10000
max_steps_per_episode = 200
solving_score = 195
solving_count = 0
for episode in range(max_episodes):
    score = agent.train_episode(max_steps_per_episode)
    scores.append(score)
    print(f"Training Episode {episode + 1}/{max_episodes}... Score: {score}")

    if score >= solving_score:
        solving_count += 1
    else:
        solving_count = 0
    
    if solving_count >= 10:
        print(f"Solved at episode {episode + 1}!")
        break

plt.plot(scores)
plt.title("Episode Rewards")
plt.savefig("train.png")

# Record 800 steps of the trained agent
print("Recording...")
render_env = Environment(True)
render_env.start_recording()

state = render_env.reset()[0]
for step in range(800):
    action, _, _ = agent.act(state)
    state, _, _, _, _ = render_env.step(action)

render_env.end_recording()
print("Done recording")