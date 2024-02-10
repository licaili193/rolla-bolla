import cv2
import numpy as np
import pickle
import pygame
import pymunk.pygame_util
from environment import Environment, config

# Initialize Pygame
pygame.init()
screen_size = (config["x_size"], config["y_size"])
screen = pygame.display.set_mode(screen_size)

# Initialize environment
env = Environment()

# Load the action log
with open('action_log.pkl', 'rb') as file:
    action_log = pickle.load(file)

# Setup Pymunk pygame drawing utilities
draw_options = pymunk.pygame_util.DrawOptions(screen)
pymunk.pygame_util.positive_y_is_up = False

# Setup video writer using OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video = cv2.VideoWriter('simulation.mp4', fourcc, config["fps"], screen_size)  # 50 FPS

def pygame_to_cvimage(surface):
    """Convert Pygame surface to an OpenCV image."""
    view = pygame.surfarray.array3d(surface)
    view = view.transpose([1, 0, 2])
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    return img_bgr

env.reset()
for i, action in enumerate(action_log):
    next_state, reward, done = env.step(action)

    # Clear screen with white background
    screen.fill((255, 255, 255))
    
    # Draw the environment using Pymunk's debug draw
    env.space.debug_draw(draw_options)
    
    pygame.display.flip()  # Update the full display Surface to the screen

    # Convert Pygame screen to OpenCV image and write to video
    cv_image = pygame_to_cvimage(screen)
    video.write(cv_image)
    
    if done:
        env.reset()

    # Optional: Print progress less frequently
    if i % 100 == 0:  # Adjust the frequency of progress updates as needed
        print(f"Processing action {i + 1} of {len(action_log)}")

video.release()  # Release the video writer
pygame.quit()  # Quit Pygame
