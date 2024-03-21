import numpy as np
import pymunk
import pygame
import pymunk.pygame_util
import cv2
import math

humanoid_config = {
    "torso_mass": 30,
    "limb_mass": 5,
    "torso_length": 50,
    "torso_width": 30,
    "upper_leg_length": 35,
    "lower_leg_length": 30,
    "leg_width": 8,
    "friction": 1.0,
    "elasticity": 0.1,
    "torso_anchor": (300, 453),
}

environment_config = {
    "platform_width": 600,
    "platform_height": 10,
    "platform_anchor": (300, 550),
}

config = {
    "x_size": 600,
    "y_size": 600,
    "humanoid_config": humanoid_config,
    "environment_config": environment_config,
    "gravity": 981,  # pymunk uses pixels/s^2
    "fps": 50,
    "state_dimention": 30,
    "action_dimention": 4,
    "action_scale": 2000,
    "torso_fall_threshold": 4.85,
}


# Humanoid

def create_limb(
    space,
    mass,
    length,
    width,
    friction,
    elasticity,
    collision_type,
    parent_body=None,
    parent_anchor=(0, 0),
    child_anchor=(0, 0),
    angle=0,
    ):
    """
    Creates a limb segment and adds it to the space, attaching it to a parent body with a joint.
    The child body is positioned so that its child_anchor aligns with the parent's parent_anchor in world coordinates.
    """

    # Calculate the adjusted length for padding
    adjusted_length = length

    # Create limb body
    moment = pymunk.moment_for_box(mass, (adjusted_length, width))
    body = pymunk.Body(mass, moment)
    body.angle = angle  # Set the body's initial angle

    if parent_body:

        # Calculate the world position for the parent's anchor
        parent_world_anchor = parent_body.local_to_world(parent_anchor)

        # Calculate the initial world position for the child's anchor
        # This requires converting the child_anchor to world space, considering the body's initial angle
        child_world_anchor = pymunk.Vec2d(child_anchor[0],
                child_anchor[1]).rotated(body.angle)

        # The body's position is then adjusted so the child's anchor aligns with the parent's anchor
        body.position = parent_world_anchor - child_world_anchor
    else:

        # If no parent, the position is set directly
        body.position = parent_anchor

    # Create limb shape
    shape = pymunk.Poly.create_box(body, (adjusted_length, width))
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = collision_type
    space.add(body, shape)

    # Connect to the parent body with a joint if specified
    if parent_body:
        joint = pymunk.PivotJoint(body, parent_body, child_anchor,
                                  parent_anchor)
        space.add(joint)

    return (body, shape)


def create_humanoid(space, config):
    # Collision types for different parts
    collision_types = {
        'torso': 1,
        'upper_arm': 2,
        'lower_arm': 3,
        'upper_leg': 4,
        'lower_leg': 5,
        }

    for (_, i) in collision_types.items():
        for (_, j) in collision_types.items():
            if i != j:
                space.add_collision_handler(i, j).begin = \
                    lambda arbiter, space, data: False

    # Create torso
    (torso, _) = create_limb(
        space,
        config["torso_mass"],
        config["torso_length"],
        config["torso_width"],
        config["friction"],
        config["elasticity"],
        collision_types['torso'],
        parent_anchor=config["torso_anchor"],
        angle=math.pi / 2,
        )

    # Upper legs - Note the adjustments for anchor points
    (upper_leg_l, _) = create_limb(
        space,
        config["limb_mass"],
        config["upper_leg_length"],
        config["leg_width"],
        config["friction"],
        config["elasticity"],
        collision_types['upper_leg'],
        parent_body=torso,
        parent_anchor=(20, -config["torso_width"] / 2),
        child_anchor=(-config["upper_leg_length"] / 2, 0),
        angle=math.pi / 2,
        )
    (lower_leg_l, _) = create_limb(
        space,
        config["limb_mass"],
        config["lower_leg_length"],
        config["leg_width"],
        config["friction"],
        config["elasticity"],
        collision_types['lower_leg'],
        parent_body=upper_leg_l,
        parent_anchor=(config["upper_leg_length"] / 2, 0),
        child_anchor=(-config["lower_leg_length"] / 2, 0),
        angle=math.pi / 2,
        )

    (upper_leg_r, _) = create_limb(
        space,
        config["limb_mass"],
        config["upper_leg_length"],
        config["leg_width"],
        config["friction"],
        config["elasticity"],
        collision_types['upper_leg'],
        parent_body=torso,
        parent_anchor=(20, config["torso_width"] / 2),
        child_anchor=(-config["upper_leg_length"] / 2, 0),
        angle=math.pi / 2,
        )
    (lower_leg_r, _) = create_limb(
        space,
        config["limb_mass"],
        config["lower_leg_length"],
        config["leg_width"],
        config["friction"],
        config["elasticity"],
        collision_types['lower_leg'],
        parent_body=upper_leg_r,
        parent_anchor=(config["upper_leg_length"] / 2, 0),
        child_anchor=(-config["lower_leg_length"] / 2, 0),
        angle=math.pi / 2,
        )
    
    return [upper_leg_l, lower_leg_l, upper_leg_r, lower_leg_r], torso


def apply_angular_impulse_to_body(body, angular_impulse):
    """
    Apply an angular impulse to a body, changing its angular velocity.
    
    Parameters:
    - body: The pymunk.Body to which the angular impulse is applied.
    - angular_impulse: The angular impulse value. Positive values will rotate counter-clockwise,
      and negative values will rotate clockwise.
    """
    # Calculate the change in angular velocity
    delta_angular_velocity = angular_impulse / body.moment
    
    # Apply the change to the body's angular velocity
    body.angular_velocity += delta_angular_velocity


def apply_angular_impulse_to_bodies(bodies, angular_impulse_list):
    """
    Apply an angular impulse to a list of bodies, changing their angular velocity.
    
    Parameters:
    - bodies: A list of pymunk.Body objects to which the angular impulse is applied.
    - angular_impulse: The list of angular impulse values. Positive values will 
      rotate counter-clockwise, and negative values will rotate clockwise.
    """
    for i in range(len(bodies)):
        apply_angular_impulse_to_body(bodies[i], angular_impulse_list[i])

# Humanoid
        
# Environment
        
def create_platform(
    space,
    position,
    length,
    height,
    ):
    """Creates a horizontal platform and adds it to the space"""

    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = position
    shape = pymunk.Segment(body, (-length / 2, 0), (length / 2, 0),
                           height)
    shape.elasticity = 0.8  # Adjust elasticity as needed
    shape.friction = 0.5
    space.add(body, shape)
    return shape

def create_environment(space, config):
    platform = create_platform(space, config["platform_anchor"], config["platform_width"], config["platform_height"])
    return []
        
# Environment

def get_object_status(object):
    # Normalize the x and y position and velocity
    return [object.position.x / 100, object.position.y / 100, object.angle, object.velocity.x / 100, object.velocity.y / 100, object.angular_velocity]

def pygame_to_cvimage(surface):
    """Convert Pygame surface to an OpenCV image."""
    view = pygame.surfarray.array3d(surface)
    view = view.transpose([1, 0, 2])
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    return img_bgr

class Environment:
    def __init__(self, enable_rendering=False):
        self.reset()

        self.enable_rendering = enable_rendering  # Control rendering initialization
        if self.enable_rendering:
            # Initialize Pygame for rendering
            pygame.init()
            self.screen_size = (config["x_size"], config["y_size"])
            self.screen = pygame.display.set_mode(self.screen_size)
            self.offscreen_surface = None  # Surface for off-screen rendering during recording
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            pymunk.pygame_util.positive_y_is_up = False
            self.video_writer = None

    def _create_objects(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, config["gravity"])

        self.limb_list, self.torso = create_humanoid(self.space, config["humanoid_config"])
        self.prop_list = create_environment(self.space, config["environment_config"])

    def reset(self):
        """Resets the environment to an initial state."""
        self._create_objects()
        self.steps_since_reset = 0  # Reset step counter
        return (self._get_state(), {}) # Match the return type of gym's reset function

    def step(self, action):
        """Simulates taking an action in the environment."""
        action = list(action)
        assert len(action) == len(self.limb_list)

        apply_angular_impulse_to_bodies(self.limb_list, action)
        
        # Step the simulation
        self.space.step(1/config["fps"])

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

    def _get_state(self):
        """Returns the current state of the system."""
        status = []
        # Torso
        status += get_object_status(self.torso)
        # Limbs
        status += get_object_status(self.limb_list[0])
        status += get_object_status(self.limb_list[1])
        status += get_object_status(self.limb_list[2])
        status += get_object_status(self.limb_list[3])
        return np.array(status)

    def _calculate_reward(self, state):
        """Calculates the reward based on the current state."""
        return 1.0

    @staticmethod
    def _check_tip_over(angle, threshold, offset=0):
        # Get the angle value and check
        return abs(angle - offset) > threshold
    
    @staticmethod
    def _check_deviation(x_1, x_2, threshold):
        # Get the x value and check
        return abs(x_1 - x_2) > threshold

    def _check_done(self, state):
        """Checks if the episode is done."""
        if state[1] > config["torso_fall_threshold"]:
            return True
        return False
    
    def start_recording(self, filename='simulation.mp4', fps=config["fps"]):
        """Starts recording the simulation to a video file."""
        if self.enable_rendering:
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