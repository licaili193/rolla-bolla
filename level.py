import pymunk

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