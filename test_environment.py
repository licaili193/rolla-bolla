import pygame
import sys
import pymunk
import pymunk.pygame_util
from environment import Environment, config

# Pygame setup
pygame.init()
pygame.font.init()  # Initialize the font module
font = pygame.font.SysFont('Arial', 12)  # Choose the font and size
screen_size = (config["x_size"], config["y_size"])
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()
env = Environment()

# Set up pymunk pygame drawing
draw_options = pymunk.pygame_util.DrawOptions(screen)
pymunk.pygame_util.positive_y_is_up = False

# Colors
background_color = pygame.Color('white')

def handle_events():
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action += config["action_boundaries"][0] / 2 # Apply a force to the left
            elif event.key == pygame.K_RIGHT:
                action += config["action_boundaries"][1] / 2 # Apply a force to the right
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                action -= config["action_boundaries"][0] / 2 # Stop a force to the left
                action = max(action, 0)
            elif event.key == pygame.K_RIGHT:
                action -= config["action_boundaries"][1] / 2 # Stop a force to the right
                action = min(action, 0)
    return action

# Main loop
running = True
while running:
    action = handle_events()
    
    # Step the environment with the current action
    next_state, reward, done = env.step(action)

    # Reset if the episode is done
    if done:
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
    clock.tick(50)

pygame.quit()