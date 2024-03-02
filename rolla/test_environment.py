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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

# Main loop
running = True
action = [0] * 8
while running:
    handle_events()

    # Get the state of all keyboard keys
    keys = pygame.key.get_pressed()

    action = [0] * 8
    if keys[pygame.K_1]:
        action[0] = config["action_scale"]
    if keys[pygame.K_2]:
        action[1] = config["action_scale"]
    if keys[pygame.K_3]:
        action[2] = config["action_scale"]
    if keys[pygame.K_4]:
        action[3] = config["action_scale"]
    if keys[pygame.K_5]:
        action[4] = config["action_scale"]
    if keys[pygame.K_6]:
        action[5] = config["action_scale"]
    if keys[pygame.K_7]:
        action[6] = config["action_scale"]
    if keys[pygame.K_8]:
        action[7] = config["action_scale"]
    
    # Step the environment with the current action
    next_state, reward, done, _, _ = env.step(action)

    # Reset if the episode is done
    if done:
        action = [0] * 8
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
    clock.tick(config["fps"])

pygame.quit()