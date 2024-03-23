import sys
import pygame
import pymunk
import pymunk.pygame_util
import argparse
from environment import create_environment

def main(env_name):
    # Pygame setup
    pygame.init()
    pygame.font.init()  # Initialize the font module
    font = pygame.font.SysFont('Arial', 12)  # Choose the font and size

    # Create the environment
    env = create_environment(env_name)
    env.reset()
    screen_size = (env.config["x_size"], env.config["y_size"])
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()

    # Set up pymunk pygame drawing
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    pymunk.pygame_util.positive_y_is_up = False

    # Colors
    background_color = pygame.Color('white')

    def handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    # Main loop
    running = True
    action = [0] * env.config["action_dimention"]
    while running:
        handle_events()

        # Get the state of all keyboard keys
        keys = pygame.key.get_pressed()

        # Initialize the action list with zeros
        action = [0] * env.config["action_dimention"]

        # Loop through the action dimensions
        for i in range(env.config["action_dimention"]):
            # Check if the corresponding number key is pressed
            if keys[getattr(pygame, f"K_{i + 1}")]:
                action[i] = env.config["action_scale"]
        
        # Step the environment with the current action
        next_state, reward, done, _, _ = env.step(action)

        # Reset if the episode is done
        if done:
            action = [0] * env.config["action_dimention"]
            env.reset()
        
        # Clear screen
        screen.fill(background_color)
        
        # Draw the pymunk space
        env.space.debug_draw(draw_options)

        # Rendering text information
        text_lines = [
            f"State: {next_state}",
            f"Action: {action}",
            f"Steps: {env.steps_since_reset}",
            f"Reward: {reward:.2f}"
        ]

        # Position for the first line of text
        text_x = 10
        text_y = 10
        line_height = 20

        # Loop through each line of text and blit it to the screen
        for line in text_lines:
            text_surface = font.render(line, True, (0, 0, 0))  # Render the text in black
            screen.blit(text_surface, (text_x, text_y))
            text_y += line_height  # Move down to the next line
        
        pygame.display.flip()
        
        # Control the frame rate for visibility
        clock.tick(env.config["fps"])

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a reinforcement learning environment.")
    parser.add_argument("env_name", type=str, help="Name of the environment to run.")
    args = parser.parse_args()
    main(args.env_name)
