import pymunk
import numpy as np

from environment.environment import BaseEnvironment

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
        "initial_impluse": 50,
        "gravity": 981,  # pymunk uses pixels/s^2
        "fps": 50,
        "action_scale": 1000,
        "angle_threshold": 0.21,  # Maximum angle (in radians) before considering the episode done
        "state_dimention": 4,
        "action_dimention": 1,
    }

class CartpoleEnvironment(BaseEnvironment):
    config =  config
    
    def _create_objects(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, self.config["gravity"])

        # Create the static line (platform)
        static_body = self.space.static_body
        static_line = pymunk.Segment(static_body, (0, self.config["platform_position"][1]), (self.config["x_size"], self.config["platform_position"][1]), self.config["platform_height"] / 2)
        static_line.elasticity = self.config["elasticity"]
        self.space.add(static_line)
        
        # Create the cart
        cart_body = pymunk.Body(self.config["cart_mass"], pymunk.moment_for_box(self.config["cart_mass"], (self.config["cart_width"], self.config["cart_height"])))
        random_x = self.config["platform_position"][0]
        cart_body.position = random_x, self.config["platform_position"][1] - self.config["cart_height"] / 2 - self.config["platform_height"] / 2
        cart_shape = pymunk.Poly.create_box(cart_body, (self.config["cart_width"], self.config["cart_height"]))
        cart_shape.friction = self.config["friction"]
        self.space.add(cart_body, cart_shape)
        self.cart = cart_body
        
        # Create the pole
        pole_body = pymunk.Body(self.config["pole_mass"], pymunk.moment_for_box(self.config["pole_mass"], (self.config["pole_width"], self.config["pole_length"])))
        pole_body.position = cart_body.position.x, cart_body.position.y - self.config["pole_length"] / 2 - self.config["cart_height"] / 2
        pole_shape = pymunk.Poly.create_box(pole_body, (self.config["pole_width"], self.config["pole_length"]))
        pole_shape.friction = self.config["friction"]
        pole_shape.color = (139, 69, 19, 255) # Set color
        self.space.add(pole_body, pole_shape)
        self.pole = pole_body
        
        # Connect the cart and pole with a PivotJoint
        self.pivot_joint = pymunk.PivotJoint(cart_body, pole_body, (cart_body.position.x, cart_body.position.y - self.config["cart_height"] / 2))
        self.space.add(self.pivot_joint)

        # Ensure no collision between the cart and the pole
        cart_shape.collision_type = 1
        pole_shape.collision_type = 2
        handler = self.space.add_collision_handler(1, 2)
        handler.begin = lambda *args, **kwargs: False

        impluse = self.config["initial_impluse"]
        self.cart.apply_impulse_at_local_point((impluse, 0), (0, 0))

    def _step_simulation(self, action):
        # Apply force to the cart
        force = (action[0], 0)
        self.cart.apply_force_at_local_point(force, (0, 0))
    
    def _get_state(self):
        return np.array([self.cart.position.x / 100, self.cart.velocity.x / 100, self.pole.angle, self.pole.angular_velocity])

    def _calculate_reward(self, state):
        return 1.0

    def _check_done(self, state):
        return abs(state[2]) > self.config["angle_threshold"] or state[0] < 0 or state[0] > self.config["x_size"]