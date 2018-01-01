import numpy as np
import scipy.integrate as integrate
import threading
import time
import math
import gym
from gym.spaces import Box, Discrete
from numpy import pi, sin, cos

class RobotArmSimulator(threading.Thread):

    def __init__(self, 
            params,             # (M_P, L_P, L_1, L_2, b, g)
            init_state = [
                0,              # theta_P
                0,              # vtheta_P
                pi,             # theta_1
                0,              # vtheta_1
                pi,             # theta_2
                0               # vtheta_2
            ]
        ):
        super(RobotArmSimulator, self).__init__()

        # thread control stuff
        self.terminated = False

        # pendulum simulation stuff
        self.params = params
        self.state = init_state

        # pseudo P(ID)A control
        self.interval = 0.005
        self.step_counter = 0
        self.threshold = 0.001
        self.max_acceleration = 50.0
        self.kp = 10.0
        self.ka = 3.0
        self.__current_target = np.array([self.state[2], self.state[4]])

    def run(self):
        while not self.terminated:
            current_time = time.time()

            current_error = self.__current_target - [self.state[2], self.state[4]]
            new_velocity = self.kp * current_error
            control_signal = self.ka * self.kp * (new_velocity - [self.state[3], self.state[5]])

            # integrating to get the new state and "correcting" to remain within the range 0-2pi
            self.state = integrate.odeint(lambda y, t: self.__derivative(y, self.params, control_signal), self.state, [0, self.interval])[1]
            self.state[0] = self.state[0] if 0 <= self.state[0] < 2 * pi else (self.state[0] - math.floor(self.state[0] / (2 * pi)) * 2 * pi if 0 <= self.state[0] else (1 - math.floor(self.state[0] / (2 * pi))) * 2 * pi - self.state[0])
            self.state[2] = self.state[2] if 0 <= self.state[2] < 2 * pi else (self.state[2] - math.floor(self.state[2] / (2 * pi)) * 2 * pi if 0 <= self.state[2] else (1 - math.floor(self.state[2] / (2 * pi))) * 2 * pi - self.state[2])
            self.state[4] = self.state[4] if 0 <= self.state[4] < 2 * pi else (self.state[4] - math.floor(self.state[4] / (2 * pi)) * 2 * pi if 0 <= self.state[4] else (1 - math.floor(self.state[4] / (2 * pi))) * 2 * pi - self.state[4])

            #self.state[3] = new_velocity if abs(new_velocity) > self.threshold else 0.0
            #print("Velocity:", self.state[3])

            time_after_execution = time.time()
            time.sleep(0 if (time_after_execution - current_time) > self.interval else self.interval - (time_after_execution - current_time))
            self.step_counter = self.step_counter + 1


    @property
    def current_target(self):
        return self.__current_target

    @current_target.setter
    def current_target(self, new_target):
        self.__current_target[0] = new_target[0]
        self.__current_target[1] = new_target[1]

    def __derivative(self, state, params, u=[0, 0]):
        # unwrapping parameters
        (M_P, L_P, L_1, L_2, b, g) = params

        # computing intermediary results
        x_accel_term = L_1 * (u[0] * cos(state[2]) - state[3] ** 2 * sin(state[2])) + L_2 * ((u[0] + u[1]) * cos(state[2] + state[4] - pi) - (state[3] + state[5]) ** 2 * sin(state[2] + state[4] - pi))
        y_accel_term = L_1 * (u[0] * sin(state[2]) + state[3] ** 2 * cos(state[2])) + L_2 * ((u[0] + u[1]) * sin(state[2] + state[4] - pi) + (state[3] + state[5]) ** 2 * cos(state[2] + state[4] - pi))

        # returning resulting derivative vector
        return [state[1], -cos(state[0]) / L_P * x_accel_term - sin(state[0]) / L_P * y_accel_term - b / (M_P * L_P ** 2) * state[1] - g / L_P * sin(state[0]), state[3], u[0], state[5], u[1]]


class RobotArmEnvironment(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array', 'ansi'],
        'video.frames_per_second' : 60
    }

    def __init__(self,
            M_P = 0.004,        # pendulum mass
            L_P = 0.09,         # pendulum length
            L_1 = 0.12,         # lower segment length
            L_2 = 0.03,         # upper segment length
            b = 0.00005,        # damping on pendulum axis
            g = 9.81            # gravity    
        ):
        super(RobotArmEnvironment, self).__init__()

        # rendering stuff
        self.viewer = None
        self.segment_1_trans = None
        self.segment_2_trans = None
        self.pendulum_trans = None
        self.scaling_factor = 1000

        # pendulum simulation stuff
        self.params = (M_P, L_P, L_1, L_2, b, g)
        self.simulation = RobotArmSimulator(self.params)
        self.simulation.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.simulation.terminated = True
        self.simulation.join()

    action_space = DiscreteAction(49, -30, 31, 15)
    observation_space = Box(np.array([0, 256, 256]), np.array([1023, 768, 768]))
    center = np.array([512, 512, 512])

    ################################################################################
    # OpenAI Gym methods
    ################################################################################

    def _seed(self, seed=None):
        return super(RobotArmEnvironment, self)._seed(seed)

    def _step(self, action, take_action=True):
        self.simulation.current_target = self.__convert_action(action)
        observation = self.__convert_observation(self.simulation.state)
        return observation, 0, False, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        state = self.simulation.state
        if mode == 'ansi':
            print('theta_P: {:3f}, vtheta_P: {:3f}, 1theta_1: {:3f}, vtheta_1: {:3f}, theta_2: {:3f}, vtheta_2: {:3f}'.format(state[0], state[1], state[2], state[3], state[4], state[5]))
            return

        (M_P, L_P, L_1, L_2, b, g) = self.params

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            # creating the screen
            screen_width = int(2 * self.scaling_factor * (L_P + L_1 + L_2) + 100)
            screen_height = int(self.scaling_factor * (L_P + L_1 + L_2) + 100)
            self.viewer = rendering.Viewer(screen_width, screen_height)

            s1w = 8    # segment 1 width
            s2w = 8    # segment 2 width
            pw = 7     # pendulum width

            # creating the movable segments and pendulum
            segment_1 = rendering.FilledPolygon([(-s1w, s1w), (s1w, s1w), (s1w, -self.scaling_factor * L_1 - s1w), (-s1w, -self.scaling_factor * L_1 - s1w)])
            segment_2 = rendering.FilledPolygon([(-s2w, s2w), (s2w, s2w), (s2w, -self.scaling_factor * L_2 - s2w), (-s2w, -self.scaling_factor * L_2 - s2w)])
            pendulum = rendering.FilledPolygon([(-7, 7), (7, 7), (7, -self.scaling_factor * L_P - 7), (-7, -self.scaling_factor * L_P - 7)])

            # setting different colors
            segment_1.set_color(.8, .6, .4)
            segment_2.set_color(.2, .6, .4)
            pendulum.set_color(.8, .3, .8)

            # creating visible joints
            joint_1 = rendering.make_circle(s1w / 2)
            joint_2 = rendering.make_circle(s2w / 2)
            joint_p = rendering.make_circle(pw / 2)

            # setting all colors to black
            joint_1.set_color(.0, .0, .0)
            joint_2.set_color(.0, .0, .0)
            joint_p.set_color(.0, .0, .0)

            # defining initial transforms (everything upright for now, rotation might change depending on initial angle)
            self.segment_1_trans = rendering.Transform(translation=(300, 50))
            self.segment_2_trans = rendering.Transform(translation=(300, 50 + self.scaling_factor * L_1))
            self.pendulum_trans = rendering.Transform(translation=(300, 50 + self.scaling_factor * L_1 + self.scaling_factor * L_2))

            # adding transforms to the created shapes
            segment_1.add_attr(self.segment_1_trans)
            segment_2.add_attr(self.segment_2_trans)
            pendulum.add_attr(self.pendulum_trans)

            # adding the same transforms to the joints
            joint_1.add_attr(self.segment_1_trans)
            joint_2.add_attr(self.segment_2_trans)
            joint_p.add_attr(self.pendulum_trans)

            # adding a line (maybe rectangle later) to serve as reference ("table")
            base = rendering.FilledPolygon([(0, 0), (0, 50), (screen_width, 50), (screen_width, 0)])
            base.set_color(0.4, 0.4, 0.4)

            # add shapes to the display
            self.viewer.add_geom(base)
            self.viewer.add_geom(segment_1)
            self.viewer.add_geom(joint_1)
            self.viewer.add_geom(segment_2)
            self.viewer.add_geom(joint_2)
            self.viewer.add_geom(pendulum)
            self.viewer.add_geom(joint_p)

        # updating rotations and translations
        self.segment_1_trans.set_rotation(state[2])

        x_trans = 300 + self.scaling_factor * L_1 * sin(state[2])
        y_trans = 50 - self.scaling_factor * L_1 * cos(state[2])
        self.segment_2_trans.set_translation(x_trans, y_trans)
        self.segment_2_trans.set_rotation(state[4] + state[2] - pi)

        pend_x_trans = x_trans + self.scaling_factor * L_2 * sin(state[4] + state[2] - pi)
        pend_y_trans = y_trans - self.scaling_factor * L_2 * cos(state[4] + state[2] - pi)
        self.pendulum_trans.set_translation(pend_x_trans, pend_y_trans)
        self.pendulum_trans.set_rotation(state[0])

        return self.viewer.render(return_rgb_array = (mode == 'rgb_array'))

    def _reset(self):
        last_state = self.simulation.state
        # not a nice way of doing this, might want to change it
        self.simulation.terminated = True
        self.simulation.join()
        self.simulation = RobotArmSimulator(self.params)
        self.simulation.start()
        return last_state

    ################################################################################
    # Other methods
    ################################################################################

    def __reward(self, state):
        return 0

    def __convert_action(self, realworld_action):
        # under the assumption that the maximum range 
        # of the used potentiometers is 0-1023
        converted_actions = []
        for index, action in enumerate(realworld_action):
            if action == 0:
                converted_actions.append(0)
                continue

            converted_action = action * (2 * pi / 1024)
            converted_actions.append(converted_action)

    def __convert_observation(self, simulation_observation):
        return [int(obs * (1024 / (2 * pi))) for obs in simulation_observation]
