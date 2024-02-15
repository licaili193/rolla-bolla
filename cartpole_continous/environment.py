import numpy as np
import pymunk
from pymunk.vec2d import Vec2d

config = {
    "x_size": 600,
    "y_size": 600,
    "platform_position": (300, 550),
    "platform_height": 10,
    "cart_mass": 1.0,
    "pole_mass": 0.1,
    "pole_length": 100.0,  # Half length for pymunk
    "pole_width": 10.0,
    "friction": 0.0,
    "elasticity": 1.0,
    "cart_width": 50,
    "cart_height": 30,
    "gravity": 981,  # pymunk uses pixels/s^2
    "fps": 50,
    "action_boundaries": (-600, 600),
    "max_initial_impluse": 50,
    "angle_threshold": 1.2,  # Maximum angle (in radians) before considering the episode done
    "max_steps": 200,  # Maximum steps in an episode
}

class Environment:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, config["gravity"])
        
        self.cart = None
        self.pole = None
        self.pivot_joint = None
        self.steps_since_reset = 0  # Track the number of steps since the last reset
        self._create_objects()

    def _create_objects(self):
        """Initializes the cart, pole, and pivot joint in the environment."""
        # Create the static line (platform)
        static_body = self.space.static_body
        static_line = pymunk.Segment(static_body, (0, config["platform_position"][1]), (config["x_size"], config["platform_position"][1]), config["platform_height"] / 2)
        static_line.elasticity = config["elasticity"]
        self.space.add(static_line)
        
        # Create the cart
        cart_body = pymunk.Body(config["cart_mass"], pymunk.moment_for_box(config["cart_mass"], (config["cart_width"], config["cart_height"])))
        # Random x pose
        # random_x = config["platform_position"][0] + np.random.uniform(-config["x_size"] / 4, config["x_size"] / 4)
        random_x = config["platform_position"][0]
        cart_body.position = random_x, config["platform_position"][1] - config["cart_height"] / 2 - config["platform_height"] / 2
        cart_shape = pymunk.Poly.create_box(cart_body, (config["cart_width"], config["cart_height"]))
        cart_shape.friction = config["friction"]
        self.space.add(cart_body, cart_shape)
        self.cart = cart_body
        
        # Create the pole
        pole_body = pymunk.Body(config["pole_mass"], pymunk.moment_for_box(config["pole_mass"], (config["pole_width"], config["pole_length"])))
        pole_body.position = cart_body.position.x, cart_body.position.y - config["pole_length"] / 2 - config["cart_height"] / 2
        pole_shape = pymunk.Poly.create_box(pole_body, (config["pole_width"], config["pole_length"]))
        pole_shape.friction = config["friction"]
        pole_shape.color = (139, 69, 19, 255) # Set color
        self.space.add(pole_body, pole_shape)
        self.pole = pole_body
        
        # Connect the cart and pole with a PivotJoint
        self.pivot_joint = pymunk.PivotJoint(cart_body, pole_body, (cart_body.position.x, cart_body.position.y - config["cart_height"] / 2))
        self.space.add(self.pivot_joint)

        # Ensure no collision between the cart and the pole
        cart_shape.collision_type = 1
        pole_shape.collision_type = 2
        handler = self.space.add_collision_handler(1, 2)
        handler.begin = lambda *args, **kwargs: False

        # Step once with an action so that it's not an equilibrium state
        # Random impluse
        # random_impluse = np.random.uniform(-config["max_initial_impluse"], config["max_initial_impluse"])
        # if random_impluse < config["max_initial_impluse"] / 10 and random_impluse > -config["max_initial_impluse"] / 10:
        #     if random_impluse > 0:
        #         random_impluse = config["max_initial_impluse"] / 10
        #     else:
        #         random_impluse = -config["max_initial_impluse"] / 10
        random_impluse = config["max_initial_impluse"]
        self.cart.apply_impulse_at_local_point((random_impluse, 0), (0, 0))

    def reset(self):
        """Resets the environment to an initial state."""
        self.space.remove(self.cart, self.pole, self.pivot_joint, *self.cart.shapes, *self.pole.shapes)
        self._create_objects()
        self.steps_since_reset = 0  # Reset step counter
        return self._get_state()

    def step(self, action):
        """Simulates taking an action in the environment."""
        state = self._get_state()

        # Check for action boundaries
        action = max(config["action_boundaries"][0], min(action, config["action_boundaries"][1]))

        # Apply force to the cart
        force = (action, 0)
        self.cart.apply_force_at_local_point(force, (0, 0))
        
        # Step the simulation
        self.space.step(1/config["fps"])

        # Increment step count
        self.steps_since_reset += 1
        
        next_state = self._get_state()
        reward = self._calculate_reward(next_state)
        done = self._check_done(next_state)
        
        return next_state, reward, done

    def _get_state(self):
        """Returns the current state of the system."""
        # For simplicity, the state could include the cart's position, velocity,
        # the pole's angle, and angular velocity. This needs to be defined based on your specific requirements.
        return np.array([self.cart.position.x, self.cart.velocity.x, self.pole.angle, self.pole.angular_velocity])

    def _calculate_reward(self, state):
        """Calculates the reward based on the current state."""
        # Extract the pole's angle from the state
        pole_angle = abs(state[2])  # Take the absolute value to consider deviation in both directions

        # Define a small threshold angle within which the reward is 1
        # It could be a fraction of the config["angle_threshold"]
        upright_threshold = config["angle_threshold"] * 0.1  # Example: 10% of the angle_threshold
        zero_reward_threshold = config["angle_threshold"] * 0.8

        if pole_angle <= upright_threshold:
            reward = 1
        elif pole_angle >= zero_reward_threshold:
            reward = 0
        else:
            # Linearly interpolate the reward between the upright_threshold and angle_threshold
            reward = 1 - (pole_angle - upright_threshold) / (zero_reward_threshold - upright_threshold)

        return reward

    def _check_done(self, state):
        """Checks if the episode is done."""
        # Example: Done if the pole falls over too much or the cart moves off screen 
        return abs(state[2]) > config["angle_threshold"] or state[0] < 0 or state[0] > config["x_size"] or self.steps_since_reset >= config["max_steps"]