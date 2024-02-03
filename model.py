import pymunk
import math

default_humanoid_config = {
    "torso_mass": 10,
    "limb_mass": 1,
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
    "torso_anchor": (300, 300),
}


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


def create_humanoid(space, config=default_humanoid_config):
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
        parent_anchor=(300, 300),
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
    
    return [torso, upper_arm_l, lower_arm_l, upper_arm_r, lower_arm_r, upper_leg_l, lower_leg_l, upper_leg_r, lower_leg_r]


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