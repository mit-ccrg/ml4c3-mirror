"""
Classic cart-pole system implemented by Rich Sutton et al.

Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

# Imports: standard library
import math
from typing import List, Tuple, Union, Optional

# Imports: third party
import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

# from gym.envs.classic_control import rendering

# pylint: disable=invalid-name,attribute-defined-outside-init


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num    Observation               Min             Max
        0    Cart Position             -4.8            4.8
        1    Cart Velocity             -Inf            Inf
        2    Pole Angle                -24 deg         24 deg
        3    Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num    Action
        0   Don't do anything
        1    Push cart to the left
        2    Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        """
        Init Gym env with cart-pole dynamics.
        """
        # Constant parameters for dynamics
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.time = 0

        # Others
        self.max_sim_time = 0
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Initialize
        self.seed()
        self.viewer = None
        self.observations = None
        self.steps_beyond_done = None

    def seed(self, seed=None) -> List:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple:
        """
        Simulate step forward for training RL algorithm.

        :param action: <int> Index with the action to take.
        :return: <Tuple> Array with observations, reward, done status and information.
        """
        # Compute the correspondent action
        if action == 0:
            force = 0
        elif action == 1:
            force = self.force_mag
        elif action == 2:
            force = -self.force_mag
        else:
            raise ValueError(
                f"Action out of scope (action > {self.action_space.n -1}), not allowed.",
            )

        # Get previous state
        x, x_dot, theta, theta_dot = self.observations

        # Math stuff
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # Update state and time
        self.time += self.tau
        self.observations = (x, x_dot, theta, theta_dot)

        done = bool(
            self.time >= self.max_sim_time * self.tau
            or x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians,
        )

        if not done:
            if (
                -self.theta_threshold_radians / 2
                < theta
                < self.theta_threshold_radians / 2
            ):
                reward = 1.0
            else:
                reward = 0.5
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior.",
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.observations), reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        :return: Reset observations.
        """
        self.time = 0
        self.observations = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.observations)

    def render(self, mode: str = "human") -> Optional[Union[bool, np.ndarray]]:
        """
        Render dynamics to visualize cart-pole.

        :param mode: <str> Rendering mode.
        :return: <Optional[Union[bool, np.ndarray]]> Viewer.
        """
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.observations is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.observations
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close visualization.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
