import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse
from agent import SACAgent
from environment import create_environment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC Agent Training')
    parser.add_argument('--recover', action='store_true', help='Recover training from the latest save')
    parser.add_argument('--env_name', type=str, default='DefaultEnvironment', help='Name of the environment')
    args = parser.parse_args()
    recover = args.recover
    env_name = args.env_name

    save_folder = "save"
    if not recover:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    env = create_environment(env_name)
    env.reset()
    agent = SACAgent(input_dims=[env.config["state_dimention"]], env=env, n_actions=env.config["action_dimention"], action_scale=env.config["action_scale"])
    n_games = 30000

    score_history = []

    if recover:
        latest_episode = max([int(f.split('ep')[1]) for f in os.listdir(save_folder) if f.startswith("models_ep")])
        agent.load_models(os.path.join(save_folder, f"models_ep{latest_episode}"))
        agent.load_replay_buffer(os.path.join(save_folder, f"replay_buffer_ep{latest_episode}"))
        score_history = list(np.load(os.path.join(save_folder, f"score_history_{latest_episode}.npy")))
        print(f"Recovered from episode {latest_episode}")
        start_episode = latest_episode + 1
    else:
        start_episode = 0

    solving_score = 395
    solving_count = 0
    max_steps_per_episode = 400
    for i in range(start_episode, n_games):
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

        if i % 500 == 0 and i > 0:
            agent.save_models(os.path.join(save_folder, f"models_ep{i}"))
            agent.save_replay_buffer(os.path.join(save_folder, f"replay_buffer_ep{i}"))
            np.save(os.path.join(save_folder, f"score_history_{i}.npy"), np.array(score_history))

    agent.save_models(os.path.join(save_folder, f"models_ep{n_games}"))
    agent.save_replay_buffer(os.path.join(save_folder, f"replay_buffer_ep{n_games}"))
    np.save(os.path.join(save_folder, "score_history_final.npy"), np.array(score_history))

    plt.plot(score_history)
    plt.title("Episode Rewards")
    plt.savefig("train.png")

    # Record 800 steps of the trained agent
    print("Recording...")
    render_env = create_environment(env_name, record=True)
    render_env.start_recording()

    state = render_env.reset()[0]
    for step in range(1600):
        action = agent.choose_action(state)
        state, _, _, _, _ = render_env.step(action)

    render_env.end_recording()
    print("Done recording")
