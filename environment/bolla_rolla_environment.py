import pymunk
import numpy as np
import math

from environment.environment import BaseEnvironment

humanoid_config = {
    "torso_mass": 30,
    "limb_mass": 5,
    "torso_length": 50,
    "torso_width": 30,
    "upper_arm_length": 30,
    "lower_arm_length": 25,
    "arm_width": 8,
    "upper_leg_length": 35,
    "lower_leg_length": 30,
    "leg_width": 8,
    "friction": 1.0,
    "elasticity": 0.1,
    "torso_anchor": (300, 390),
}

environment_config = {
    "platform_width": 600,
    "platform_height": 10,
    "platform_anchor": (300, 550),
    "plank_anchor": (300, 482.5),
    "plank_mass": 2,
    "plank_width": 120,
    "plank_height": 5,
    "ball_radius": 25,
    "ball_anchor": (300, 515),
    "ball_mass": 10,
}

config = {
    "x_size": 600,
    "y_size": 600,
    "humanoid_config": humanoid_config,
    "environment_config": environment_config,
    "gravity": 981,  # pymunk uses pixels/s^2
    "fps": 50,
    "state_dimention": 60,
    "action_dimention": 8,
    "action_scale": 2000,
    "plank_tipover_threshold": 0.8,
    "torso_tipover_threshold": 1.5,
    "plank_ball_deviation_threshold": 40,
    "plank_torso_deviation_threshold": 40,
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

    # Upper arms - Note the adjustments for anchor points
    (upper_arm_l, _) = create_limb(
        space,
        config["limb_mass"],
        config["upper_arm_length"],
        config["arm_width"],
        config["friction"],
        config["elasticity"],
        collision_types['upper_arm'],
        parent_body=torso,
        parent_anchor=(-15, -config["torso_width"] / 2),
        child_anchor=(-config["upper_arm_length"] / 2, 0),
        )
    (lower_arm_l, _) = create_limb(
        space,
        config["limb_mass"],
        config["lower_arm_length"],
        config["arm_width"],
        config["friction"],
        config["elasticity"],
        collision_types['lower_arm'],
        parent_body=upper_arm_l,
        parent_anchor=(config["upper_arm_length"] / 2, 0),
        child_anchor=(-config["lower_arm_length"] / 2, 0),
        )

    (upper_arm_r, _) = create_limb(
        space,
        config["limb_mass"],
        config["upper_arm_length"],
        config["arm_width"],
        config["friction"],
        config["elasticity"],
        collision_types['upper_arm'],
        parent_body=torso,
        parent_anchor=(-15, config["torso_width"] / 2),
        child_anchor=(-config["upper_arm_length"] / 2, 0),
        angle=math.pi,
        )
    (lower_arm_r, _) = create_limb(
        space,
        config["limb_mass"],
        config["lower_arm_length"],
        config["arm_width"],
        config["friction"],
        config["elasticity"],
        collision_types['lower_arm'],
        parent_body=upper_arm_r,
        parent_anchor=(config["upper_arm_length"] / 2, 0),
        child_anchor=(-config["lower_arm_length"] / 2, 0),
        angle=math.pi,
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
    
    return [upper_arm_l, lower_arm_l, upper_arm_r, lower_arm_r, upper_leg_l, lower_leg_l, upper_leg_r, lower_leg_r], torso


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

def create_plank(
    space,
    position,
    length,
    height,
    mass,
    ):
    """Creates a horizontal platform and adds it to the space"""

    moment = pymunk.moment_for_box(mass, (length, height))
    body = pymunk.Body(mass, moment)
    body.position = position
    shape = pymunk.Segment(body, (-length / 2, 0), (length / 2, 0),
                           height)
    shape.elasticity = 0.8  # Adjust elasticity as needed
    shape.friction = 1.0
    shape.color = (139, 69, 19, 255)
    space.add(body, shape)
    return body

def create_ball(
    space,
    position,
    radius,
    mass,
    ):
    """Creates a horizontal platform and adds it to the space"""

    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.8  # Adjust elasticity as needed
    shape.friction = 1.0
    shape.color = (32, 175, 205, 255)
    space.add(body, shape)
    return body

def create_environment(space, config):
    platform = create_platform(space, config["platform_anchor"], config["platform_width"], config["platform_height"])
    plank = create_plank(space, config["plank_anchor"], config["plank_width"], config["plank_height"], config["plank_mass"])
    ball = create_ball(space, config["ball_anchor"], config["ball_radius"], config["ball_mass"])
    return [plank, ball]
        
# Environment

def get_object_status(object):
    # Normalize the x and y position and velocity
    return [object.position.x / 100, object.position.y / 100, object.angle, object.velocity.x / 100, object.velocity.y / 100, object.angular_velocity]

class BollaRollaEnvironment(BaseEnvironment):
    """
    The BollaRollaEnvironment class represents an environment for the BollaRolla game.
    It inherits from the BaseEnvironment class and implements the necessary methods
    for simulation, state retrieval, reward calculation, and episode termination checks.
    """

    config = config

    def _create_objects(self):
        """
        Creates the objects in the environment, such as the humanoid and the environment props.
        """

        self.space = pymunk.Space()
        self.space.gravity = (0, self.config["gravity"])

        self.limb_list, self.torso = create_humanoid(self.space, self.config["humanoid_config"])
        self.prop_list = create_environment(self.space, self.config["environment_config"])

    def _step_simulation(self, action):
        """
        Steps the simulation by applying the given action to the humanoid limbs.
        """

        apply_angular_impulse_to_bodies(self.limb_list, action)
    
    def _get_state(self):
        """
        Retrieves the current state of the environment.
        """

        status = []
        # Ball
        temp_status = get_object_status(self.prop_list[1])
        status += [temp_status[0], temp_status[3]]
        # Plank
        temp_status = get_object_status(self.prop_list[0])
        status += [temp_status[0], temp_status[2], temp_status[3], temp_status[5]]
        # Torso
        status += get_object_status(self.torso)
        # Limbs
        status += get_object_status(self.limb_list[0])
        status += get_object_status(self.limb_list[1])
        status += get_object_status(self.limb_list[2])
        status += get_object_status(self.limb_list[3])
        status += get_object_status(self.limb_list[4])
        status += get_object_status(self.limb_list[5])
        status += get_object_status(self.limb_list[6])
        status += get_object_status(self.limb_list[7])
        return np.array(status)

    def _calculate_reward(self, state):
        """
        Calculates the reward based on the current state.
        """

        return 1.0

    @staticmethod
    def _check_tip_over(angle, threshold, offset=0):
        """
        Checks if the given angle indicates tipping over, based on the threshold and offset.
        """

        return abs(angle - offset) > threshold
    
    @staticmethod
    def _check_deviation(x_1, x_2, threshold):
        """
        Checks if the given x values deviate by more than the threshold.
        """

        return abs(x_1 - x_2) > threshold

    def _check_done(self, state):
        """
        Checks if the episode is done based on the current state.
        """

        status = get_object_status(self.prop_list[1])
        # Check if the plank has tipped over
        if BollaRollaEnvironment._check_tip_over(state[3], self.config["plank_tipover_threshold"]):
            return True
        # Check if the torso has tipped over
        if BollaRollaEnvironment._check_tip_over(state[8], self.config["torso_tipover_threshold"], 1.57):
            return True
        # Check if the torso has left the plank
        if BollaRollaEnvironment._check_deviation(state[6], state[2], self.config["plank_torso_deviation_threshold"]):
            return True
        # Check if the ball has left the plank
        if BollaRollaEnvironment._check_deviation(state[0], state[2], self.config["plank_ball_deviation_threshold"]):
            return True

        return False