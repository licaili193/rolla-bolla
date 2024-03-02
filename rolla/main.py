import matplotlib.pyplot as plt
import numpy as np
from agent import SACAgent
from environment import Environment, config

if __name__ == '__main__':
    env = Environment()
    agent = SACAgent(input_dims=[config["state_dimention"]], env=env, n_actions=config["action_dimention"], action_scale=config["action_scale"])
    n_games = 15000

    score_history = []
    
    solving_score = 595
    solving_count = 0
    max_steps_per_episode = 600
    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            steps += 1
            if steps >= max_steps_per_episode:
                break
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        if score >= solving_score:
            solving_count += 1
        else:
            solving_count = 0

        if solving_count >= 10:
            print(f"Solved at episode {i + 1}!")
            break

    plt.plot(score_history)
    plt.title("Episode Rewards")
    plt.savefig("train.png")

    # Record 800 steps of the trained agent
    print("Recording...")
    render_env = Environment(True)
    render_env.start_recording()

    state = render_env.reset()[0]
    for step in range(1800):
        action = agent.choose_action(state)
        state, _, _, _, _ = render_env.step(action)

    render_env.end_recording()
    print("Done recording")