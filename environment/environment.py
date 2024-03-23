import pymunk
import pygame
import cv2

def pygame_to_cvimage(surface):
    """Convert Pygame surface to an OpenCV image."""
    view = pygame.surfarray.array3d(surface)
    view = view.transpose([1, 0, 2])
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    return img_bgr

class BaseEnvironment:
    # Dummy configuration for the base class
    config = {
        "x_size": 600,
        "y_size": 600,
        "fps": 50,
    }

    def __init__(self, enable_rendering=False):
        self.enable_rendering = enable_rendering  # Control rendering initialization
        if self.enable_rendering:
            # Initialize Pygame for rendering
            pygame.init()
            self.screen_size = (self.config["x_size"], self.config["y_size"])
            self.screen = pygame.display.set_mode(self.screen_size)
            self.offscreen_surface = None  # Surface for off-screen rendering during recording
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            pymunk.pygame_util.positive_y_is_up = False
            self.video_writer = None

    def reset(self):
        """Resets the environment to an initial state."""
        self._create_objects()
        self.steps_since_reset = 0  # Reset step counter
        return (self._get_state(), {})  # Match the return type of gym's reset function

    def _step_simulation(self, action):
        """Apply forces or torques and advance in simulation."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def step(self, action):
        """Simulates taking an action in the environment."""
        action = list(action)

        self._step_simulation(action)
        
        # Step the simulation
        self.space.step(1/self.config["fps"])

        # Increment step count
        self.steps_since_reset += 1
        
        next_state = self._get_state()
        done = self._check_done(next_state)
        reward = self._calculate_reward(next_state)

        # If recording, add the current frame to the video
        if self.enable_rendering and self.video_writer:
            # Clear the off-screen surface with white background
            self.offscreen_surface.fill((255, 255, 255))
            
            # Draw the environment using Pymunk's debug draw
            self.draw_options.surface = self.offscreen_surface  # Set the draw surface to the off-screen surface
            self.space.debug_draw(self.draw_options)
            
            # Convert the off-screen surface to an OpenCV image and write to video
            cv_image = pygame_to_cvimage(self.offscreen_surface)
            self.video_writer.write(cv_image)

            # Reset the draw surface to the Pygame screen
            self.draw_options.surface = self.screen
        
        return next_state, reward, done, False, {}  # Match the return type of gym's step function

    def _create_objects(self):
        """Init environment."""
        raise NotImplementedError("Subclasses must implement this method")

    def _get_state(self):
        """Returns the current state of the system."""
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_reward(self, state):
        """Calculates the reward based on the current state."""
        raise NotImplementedError("Subclasses must implement this method")

    def _check_done(self, state):
        """Checks if the episode is done."""
        raise NotImplementedError("Subclasses must implement this method")

    def start_recording(self, filename='simulation.mp4', fps=None):
        """Starts recording the simulation to a video file."""
        if self.enable_rendering:
            if fps is None:
                fps = self.config["fps"]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, self.screen_size)
            # Create an off-screen surface for rendering during recording
            self.offscreen_surface = pygame.Surface(self.screen_size)

    def end_recording(self):
        """Ends the recording and releases the video writer."""
        if self.enable_rendering and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.offscreen_surface = None

    def get_config(self):
        """Returns the configuration of the environment."""
        return self.config
