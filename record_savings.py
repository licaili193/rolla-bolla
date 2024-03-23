import os
from agent import SACAgent
from environment import Environment, config

save_folder = "save"
output_folder = "recordings"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(save_folder):
    if filename.startswith("models_ep"):
        episode = filename.split('_ep')[1]
        print(f"Recording model from episode {episode}...")

        env = Environment(True)
        agent = SACAgent(input_dims=[config["state_dimention"]], env=env, n_actions=config["action_dimention"], action_scale=config["action_scale"])
        agent.load_models(os.path.join(save_folder, f"models_ep{episode}"))

        env.start_recording(os.path.join(output_folder, f"recording_ep{episode}.mp4"))

        state = env.reset()[0]
        for step in range(1600):
            action = agent.choose_action(state)
            state, _, _, _, _ = env.step(action)

        env.end_recording()
        print(f"Done recording episode {episode}")
