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
    """
    Base class for creating custom environments.

    Attributes:
        config (dict): Configuration parameters for the environment.
        enable_rendering (bool): Flag to enable rendering of the environment.
        screen_size (tuple): Size of the Pygame screen for rendering.
        screen (pygame.Surface): Pygame screen object for rendering.
        offscreen_surface (pygame.Surface): Surface for off-screen rendering during recording.
        draw_options (pymunk.pygame_util.DrawOptions): Draw options for Pymunk's debug draw.
        video_writer (cv2.VideoWriter): Video writer object for recording the simulation.
        steps_since_reset (int): Counter for the number of steps since the last reset.
        space (pymunk.Space): Pymunk physics space for the environment.
    """

    config = {
        "x_size": 600,
        "y_size": 600,
        "fps": 50,
    }

    def __init__(self, enable_rendering=False):
        """
        Initializes the BaseEnvironment.

        Args:
            enable_rendering (bool, optional): Flag to enable rendering of the environment. Defaults to False.
        """
        self.enable_rendering = enable_rendering
        if self.enable_rendering:
            pygame.init()
            self.screen_size = (self.config["x_size"], self.config["y_size"])
            self.screen = pygame.display.set_mode(self.screen_size)
            self.offscreen_surface = None
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            pymunk.pygame_util.positive_y_is_up = False
            self.video_writer = None

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            tuple: A tuple containing the initial state and an empty dictionary.
        """
        self._create_objects()
        self.steps_since_reset = 0
        return (self._get_state(), {})

    def _step_simulation(self, action):
        """
        Applies forces or torques and advances in simulation.

        Args:
            action (list): List of actions to be applied.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, action):
        """
        Simulates taking an action in the environment.

        Args:
            action (list): List of actions to be applied.

        Returns:
            tuple: A tuple containing the next state, reward, done flag, False flag, and an empty dictionary.
        """
        action = list(action)

        self._step_simulation(action)

        self.space.step(1 / self.config["fps"])

        self.steps_since_reset += 1

        next_state = self._get_state()
        done = self._check_done(next_state)
        reward = self._calculate_reward(next_state)

        if self.enable_rendering and self.video_writer:
            self.offscreen_surface.fill((255, 255, 255))
            self.draw_options.surface = self.offscreen_surface
            self.space.debug_draw(self.draw_options)
            cv_image = pygame_to_cvimage(self.offscreen_surface)
            self.video_writer.write(cv_image)
            self.draw_options.surface = self.screen

        return next_state, reward, done, False, {}

    def _create_objects(self):
        """
        Initializes the environment.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_state(self):
        """
        Returns the current state of the system.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_reward(self, state):
        """
        Calculates the reward based on the current state.

        Args:
            state: The current state of the system.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _check_done(self, state):
        """
        Checks if the episode is done.

        Args:
            state: The current state of the system.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def start_recording(self, filename='simulation.mp4', fps=None):
        """
        Starts recording the simulation to a video file.

        Args:
            filename (str, optional): Name of the video file. Defaults to 'simulation.mp4'.
            fps (int, optional): Frames per second for the video. Defaults to None.
        """
        if self.enable_rendering:
            if fps is None:
                fps = self.config["fps"]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, self.screen_size)
            self.offscreen_surface = pygame.Surface(self.screen_size)

    def end_recording(self):
        """Ends the recording and releases the video writer."""
        if self.enable_rendering and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.offscreen_surface = None

    def get_config(self):
        """
        Returns the configuration of the environment.

        Returns:
            dict: The configuration parameters of the environment.
        """
        return self.config
