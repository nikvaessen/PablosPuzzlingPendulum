import sys

sys.path.append('../')

import numpy as np
import scipy.integrate as integrate
import threading
import time
import math
import gym
from gym.spaces import Box, Discrete
from numpy import pi, sin, cos
from ourgym import AbsoluteDiscreteActionMap


class RobotArmSimulatorParallel(threading.Thread):

    def __init__(self,
                 params,  # (M_P, L_P, L_1, L_2, b, g)
                 init_state=(
                         0,  # theta_P
                         0,  # vtheta_P
                         pi,  # theta_1
                         0,  # vtheta_1
                         pi,  # theta_2
                         0  # vtheta_2
                 ),
                 acceleration_control=False
                 ):
        super(RobotArmSimulatorParallel, self).__init__()

        # thread control stuff
        self.terminated = False

        # pendulum simulation stuff
        self.params = params
        self._state = init_state

        # pseudo P(ID)A control
        self.interval = 0.005
        self.step_counter = 0
        self.threshold = 0.001
        self.max_acceleration = 50.0
        self.kp = 20.0
        self.ka = 3.0
        self.__current_target = np.array([self._state[2], self._state[4]])
        self.control_signal = None

        # acceleration control stuff
        self.acceleration_control = acceleration_control
        self.acceleration_limit = pi
        self.current_acceleration = (0, 0)

        # noise parameters
        self.step_noise = 0
        self.observation_noise = 0

    def run(self):
        while not self.terminated:
            current_time = time.time()

            if not self.acceleration_control:
                current_error = self.__current_target - [self._state[2], self._state[4]]
                new_velocity = self.kp * current_error
                if abs(new_velocity[0]) > 4 * pi: new_velocity[0] = 4 * pi * np.sign(new_velocity[0])
                if abs(new_velocity[1]) > 4 * pi: new_velocity[1] = 4 * pi * np.sign(new_velocity[1])
                self.control_signal = self.ka * self.kp * (new_velocity - [self._state[3], self._state[5]])
            else:
                self.control_signal = self.current_acceleration

            # integrating to get the new state and "correcting" to remain within the range 0-2pi
            self._state = \
                integrate.odeint(lambda y, t: self.__derivative(y, self.params, self.control_signal), self._state,
                                 [0, self.interval])[1]
            self._state[0] = self._state[0] if 0 <= self._state[0] < 2 * pi else (
                self._state[0] - math.floor(self._state[0] / (2 * pi)) * 2 * pi if 0 <= self._state[0] else (
                                                                                                                    1 - math.floor(
                                                                                                                self._state[
                                                                                                                    0] / (
                                                                                                                        2 * pi))) * 2 * pi -
                                                                                                            self._state[
                                                                                                                0])
            self._state[2] = self._state[2] if 0 <= self._state[2] < 2 * pi else (
                self._state[2] - math.floor(self._state[2] / (2 * pi)) * 2 * pi if 0 <= self._state[2] else (
                                                                                                                    1 - math.floor(
                                                                                                                self._state[
                                                                                                                    2] / (
                                                                                                                        2 * pi))) * 2 * pi -
                                                                                                            self._state[
                                                                                                                2])
            self._state[4] = self._state[4] if 0 <= self._state[4] < 2 * pi else (
                self._state[4] - math.floor(self._state[4] / (2 * pi)) * 2 * pi if 0 <= self._state[4] else (
                                                                                                                    1 - math.floor(
                                                                                                                self._state[
                                                                                                                    4] / (
                                                                                                                        2 * pi))) * 2 * pi -
                                                                                                            self._state[
                                                                                                                4])

            if abs(self._state[3]) > 4 * pi:
                self._state[3] = 4 * pi * np.sign(self._state[3])
                print("LOWER JOINT CLIPPED, TARGET:", self.__current_target)
            if abs(self._state[5]) > 4 * pi:
                self._state[5] = 4 * pi * np.sign(self._state[5])
                print("UPPER JOINT CLIPPED, TARGET:", self.__current_target)

            # self.state[3] = new_velocity if abs(new_velocity) > self.threshold else 0.0
            # print("Velocity:", self.state[3])

            time_after_execution = time.time()
            # print("Time taken for parallel computations: {}s".format(time_after_execution - current_time))
            # time.sleep(0 if (time_after_execution - current_time) > self.interval else self.interval - (time_after_execution - current_time))
            time.sleep(0.00000001)
            self.step_counter += 1
            # print(self.step_counter)

    @property
    def state(self):
        return self._state + self.observation_noise * np.random.randn(len(self._state))

    @property
    def current_target(self):
        return self.__current_target

    @current_target.setter
    def current_target(self, new_target):
        self.step_counter = 0
        self.__current_target[0] = new_target[0]
        self.__current_target[1] = new_target[1]
        self.__current_target += self.step_noise * np.random.randn(1, 2)

    def get_counter(self):
        return self.step_counter

    @staticmethod
    def __derivative(state, params, u=(0, 0)):
        # unwrapping parameters
        (M_P, L_P, L_1, L_2, b, g) = params

        # computing intermediary results
        x_accel_term = L_1 * (u[0] * cos(state[2]) - state[3] ** 2 * sin(state[2])) + L_2 * (
                (u[0] + u[1]) * cos(state[2] + state[4] - pi) - (state[3] + state[5]) ** 2 * sin(
            state[2] + state[4] - pi))
        y_accel_term = L_1 * (u[0] * sin(state[2]) + state[3] ** 2 * cos(state[2])) + L_2 * (
                (u[0] + u[1]) * sin(state[2] + state[4] - pi) + (state[3] + state[5]) ** 2 * cos(
            state[2] + state[4] - pi))

        # returning resulting derivative vector
        return [state[1],
                -cos(state[0]) / L_P * x_accel_term - sin(state[0]) / L_P * y_accel_term - b / (M_P * L_P ** 2) * state[
                    1] - g / L_P * sin(state[0]), state[3], u[0], state[5], u[1]]


class RobotArmSimulatorSerial:

    def __init__(self,
                 params,  # (M_P, L_P, L_1, L_2, b, g)
                 init_state=(
                         pi + np.random.rand() * 0.01,  # theta_P
                         0,  # vtheta_P
                         pi,  # theta_1
                         0,  # vtheta_1
                         pi,  # theta_2
                         0  # vtheta_2
                 ),
                 acceleration_control=False
                 ):

        # pendulum simulation stuff
        self.params = params
        self._state = init_state

        # pseudo P(ID)A control
        self.interval = 0.005
        self.step_counter = 0
        self.threshold = 0.001
        self.max_acceleration = 50.0
        self.kp = 20.0
        self.ka = 3.0
        self.__current_target = np.array([self._state[2], self._state[4]])
        self.control_signal = None

        # torque control stuff
        self.acceleration_control = acceleration_control
        self.acceleration_limit = pi
        self.current_acceleration = (0, 0)

        # noise parameters
        self.step_noise = 0
        self.observation_noise = 0

    def advance(self, n):
        for _ in range(n):
            self.__step()

    def __step(self):
        if not self.acceleration_control:
            current_error = self.__current_target - [self._state[2], self._state[4]]
            new_velocity = self.kp * current_error
            if abs(new_velocity[0]) > 4 * pi: new_velocity[0] = 4 * pi * np.sign(new_velocity[0])
            if abs(new_velocity[1]) > 4 * pi: new_velocity[1] = 4 * pi * np.sign(new_velocity[1])
            self.control_signal = self.ka * self.kp * (new_velocity - [self._state[3], self._state[5]])
            # print("Target: {}, Position: {}, Error: {}, Control signal: {}".format(self.__current_target,
            #                                                                        self._state,
            #                                                                        current_error,
            #                                                                        self.control_signal))
        else:
            self.control_signal = self.current_acceleration

        # integrating to get the new state and "correcting" to remain within the range 0-2pi
        self._state = \
            integrate.odeint(lambda y, t: self.__derivative(y, self.params, self.control_signal), self._state,
                             [0, self.interval])[1]
        self._state[0] = self._state[0] if 0 <= self._state[0] < 2 * pi else (
            self._state[0] - math.floor(self._state[0] / (2 * pi)) * 2 * pi if 0 <= self._state[0] else (1 - math.floor(
                self._state[0] / (2 * pi))) * 2 * pi - self._state[0])
        self._state[2] = self._state[2] if 0 <= self._state[2] < 2 * pi else (
            self._state[2] - math.floor(self._state[2] / (2 * pi)) * 2 * pi if 0 <= self._state[2] else (1 - math.floor(
                self._state[2] / (2 * pi))) * 2 * pi - self._state[2])
        self._state[4] = self._state[4] if 0 <= self._state[4] < 2 * pi else (
            self._state[4] - math.floor(self._state[4] / (2 * pi)) * 2 * pi if 0 <= self._state[4] else (1 - math.floor(
                self._state[4] / (2 * pi))) * 2 * pi - self._state[4])

        if abs(self._state[3]) > 4 * pi:
            self._state[3] = 4 * pi * np.sign(self._state[3])
            print("LOWER JOINT CLIPPED, TARGET:", self.__current_target)
        if abs(self._state[5]) > 4 * pi:
            self._state[5] = 4 * pi * np.sign(self._state[5])
            print("UPPER JOINT CLIPPED, TARGET:", self.__current_target)

        self.step_counter += 1

    def start(self):
        pass

    def join(self):
        pass

    @property
    def state(self):
        return self._state + self.observation_noise * np.random.randn(len(self._state))

    @property
    def current_target(self):
        return self.__current_target

    @current_target.setter
    def current_target(self, new_target):
        self.step_counter = 0
        self.__current_target[0] = new_target[0]
        self.__current_target[1] = new_target[1]
        # print("Target on assignment: {}, {}".format(self.__current_target, new_target))
        self.__current_target += self.step_noise * np.random.randn(2)

    def get_counter(self):
        self.advance(1)
        return self.step_counter

    @staticmethod
    def __derivative(state, params, u=(0, 0)):
        # unwrapping parameters
        (M_P, L_P, L_1, L_2, b, g) = params

        # computing intermediary results
        x_accel_term = L_1 * (u[0] * cos(state[2]) - state[3] ** 2 * sin(state[2])) + L_2 * (
                (u[0] + u[1]) * cos(state[2] + state[4] - pi) - (state[3] + state[5]) ** 2 * sin(
            state[2] + state[4] - pi))
        y_accel_term = L_1 * (u[0] * sin(state[2]) + state[3] ** 2 * cos(state[2])) + L_2 * (
                (u[0] + u[1]) * sin(state[2] + state[4] - pi) + (state[3] + state[5]) ** 2 * cos(
            state[2] + state[4] - pi))

        # returning resulting derivative vector
        return [state[1],
                -cos(state[0]) / L_P * x_accel_term - sin(state[0]) / L_P * y_accel_term - b / (M_P * L_P ** 2) * state[
                    1] - g / L_P * sin(state[0]), state[3], u[0], state[5], u[1]]


class RobotArmEnvironment(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'ansi'],
        'video.frames_per_second': 60
    }

    def __init__(self,
                 M_P=0.04,  # pendulum mass
                 L_P=0.09,  # pendulum length
                 L_1=0.12,  # lower segment length
                 L_2=0.03,  # upper segment length
                 b=0.0005,  # damping on pendulum axis
                 g=9.81,  # gravity
                 sim_ticks_per_step=10,
                 reward_average=False,
                 reward_function_index=0,
                 reward_function_params=(1 / 6 * pi, 2 * pi, 1, 1),
                 from_json_object=None
                 ):
        super(RobotArmEnvironment, self).__init__()

        # rendering stuff
        self.viewer = None
        self.segment_1_trans = None
        self.segment_2_trans = None
        self.pendulum_trans = None
        self.scaling_factor = 1000

        if not from_json_object:
            # reward function stuff
            self.reward_average = reward_average
            self.reward_function_index = reward_function_index
            self.reward_function_params = reward_function_params
            self.reward_function = self.__get_reward_function(self.reward_function_index, self.reward_function_params)

            # pendulum simulation stuff
            self.params = (M_P, L_P, L_1, L_2, b, g)
            self.sim_ticks_per_step = sim_ticks_per_step
            self.simulation = RobotArmSimulatorSerial(self.params)
            self.simulation.start()
        else:
            # reward function stuff
            if 'average' in from_json_object['function'].keys():
                self.reward_average = from_json_object['reward_function']['average']
            else:
                self.reward_average = reward_average
            self.reward_function_index = from_json_object['reward_function']['index']
            self.reward_function_params = from_json_object['reward_function']['parameters']
            self.reward_function = self.__get_reward_function(self.reward_function_index, self.reward_function_params)

            # pendulum simulation stuff
            self.params = from_json_object['physical_parameters']
            self.sim_ticks_per_step = from_json_object['sim_ticks_per_step']
            self.simulation = RobotArmSimulatorSerial(self.params)
            self.simulation.interval = from_json_object['sim_interval']
            self.simulation.threshold = from_json_object['sim_threshold']
            self.simulation.max_acceleration = from_json_object['sim_max_acceleration']
            self.simulation.kp = from_json_object['sim_kp']
            self.simulation.ka = from_json_object['sim_ka']
            self.simulation.acceleration_control = from_json_object['sim_acceleration_control']
            self.simulation.acceleration_limit = from_json_object['sim_acceleration_limit']
            self.simulation.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.simulation.terminated = True
        self.simulation.join()

    # Map the disctete action space to a "real" action
    action_map = AbsoluteDiscreteActionMap(70, 110, 15)
    action_space = Discrete(len(action_map.actions))

    observation_space = Box(np.array([0, 256, 256, 0, 0, 0]), np.array([1023, 768, 768, 1000, 1000, 1000]))
    center = np.array([512, 512, 512])

    ################################################################################
    # OpenAI Gym methods
    ################################################################################

    def _seed(self, seed=None):
        return super(RobotArmEnvironment, self)._seed(seed)

    def _step(self, action, take_action=True):
        actual_action = self.__convert_action(self.action_map.get(action))
        # print(actual_action)
        state_before_action = self.simulation.state

        # if actual_action[0] + state_before_action[2] < 3 / 4 * pi:
        #     actual_action[0] = 0
        # elif actual_action[0] + state_before_action[2] > 5 / 4 * pi:
        #     actual_action[0] = 0
        #
        # if actual_action[1] + state_before_action[4] < 3 / 4 * pi:
        #     actual_action[1] = 0
        # elif actual_action[1] + state_before_action[4] > 5 / 4 * pi:
        #     actual_action[1] = 0

        self.simulation.current_target = np.array(actual_action)
        something_small = 0.0000001

        reward_weighted_sum = 0
        weight_steps = 1.0 / self.sim_ticks_per_step
        current_weight = weight_steps
        while True:
            time.sleep(something_small)
            if self.reward_average:
                reward_weighted_sum += current_weight * self.__reward(self.simulation.state)
                current_weight += weight_steps
            if self.simulation.get_counter() >= self.sim_ticks_per_step:
                break

        # TODO: replace this with advance(3)?
        # NOTE: for the serial simulation, the interval should be changed depending on how many
        #       states are being observed/how many steps the simulation is advanced; otherwise
        #       it may not be possible to do a swing-up at all


        state_after_action = self.simulation.state
        done = not ((5/8*pi) <= state_after_action[0] <= (11/8*pi))
        #done = False
        return self.__normalize_state(np.array(state_after_action)), \
               reward_weighted_sum if self.reward_average else self.__reward(state_after_action), \
               done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        state = self.simulation._state
        if mode == 'ansi':
            print(
                'theta_P: {:3f}, vtheta_P: {:3f}, 1theta_1: {:3f}, vtheta_1: {:3f}, theta_2: {:3f}, vtheta_2: {:3f}'.format(
                    state[0], state[1], state[2], state[3], state[4], state[5]))
            return

        (M_P, L_P, L_1, L_2, b, g) = self.params

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            # creating the screen
            screen_width = int(2 * self.scaling_factor * (L_P + L_1 + L_2) + 100)
            screen_height = int(self.scaling_factor * (L_P + L_1 + L_2) + 100)
            self.viewer = rendering.Viewer(screen_width, screen_height)

            s1w = 8  # segment 1 width
            s2w = 8  # segment 2 width
            pw = 7  # pendulum width

            # creating the movable segments and pendulum
            segment_1 = rendering.FilledPolygon([(-s1w, s1w), (s1w, s1w), (s1w, -self.scaling_factor * L_1 - s1w),
                                                 (-s1w, -self.scaling_factor * L_1 - s1w)])
            segment_2 = rendering.FilledPolygon([(-s2w, s2w), (s2w, s2w), (s2w, -self.scaling_factor * L_2 - s2w),
                                                 (-s2w, -self.scaling_factor * L_2 - s2w)])
            pendulum = rendering.FilledPolygon(
                [(-7, 7), (7, 7), (7, -self.scaling_factor * L_P - 7), (-7, -self.scaling_factor * L_P - 7)])

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
            self.pendulum_trans = rendering.Transform(
                translation=(300, 50 + self.scaling_factor * L_1 + self.scaling_factor * L_2))

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

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def _reset(self):
        last_state = self.simulation.state
        # not a nice way of doing this, might want to change it
        self.simulation.terminated = True
        self.simulation.join()
        self.simulation = RobotArmSimulatorSerial(self.params)
        self.simulation.start()
        return self.__normalize_state(np.array(last_state))

    ################################################################################
    # Other methods
    ################################################################################

    def __reward(self, state):
        return self.reward_function(state)
        # if abs(state[0] - pi) <= 1 / 6 * pi and abs(state[1]) <= 2 * pi:
        #     return np.e ** -abs(state[1]) * 10
        # else:
        #     return 0
        # return -((state[0]-np.pi)**2 + 0.001*abs(state[1]))

    @staticmethod
    def __get_reward_function(index, parameters):
        if index == 0:
            def reward_function(state):
                if abs(state[0] - pi) <= parameters[0] and abs(state[1]) <= parameters[1]:
                    return (np.e ** -(parameters[2] * abs(state[1]))) * parameters[3]
                else:
                    return 0

            return reward_function
        elif index == 1:
            def reward_function(state):
                # default parameters: (1/6 * np.pi, 2 * np.pi, 1, 10, 0.05, 0.1, 2, 0.001, 1)
                if abs(state[0] - pi) <= parameters[0] and abs(state[1]) <= parameters[1]:
                    return (np.e ** -(parameters[2] * abs(state[1]))) * parameters[3]
                else:
                    return -parameters[4] * (parameters[5] * (abs(state[0] - pi) ** parameters[6]) + parameters[7] * (
                            abs(state[1]) ** parameters[8]))

            return reward_function

    @staticmethod
    def __convert_action(realworld_action):
        ## the [0], [0]
        # under the assumption that the maximum range 
        # of the used potentiometers is 0-1023
        converted_actions = []
        for index, action in enumerate(realworld_action):
            if action == 0:
                converted_actions.append(0)
                continue

            converted_action = action * ((2 * pi) / 360) + (0.5 * pi)
            converted_actions.append(converted_action)
        return converted_actions

    @staticmethod
    def __convert_observation(simulation_observation):
        return [int(obs * (1024 / (2 * pi))) for obs in simulation_observation]

    @staticmethod
    def __normalize_state(state):
        # state is size 6,
        # index 0, 2 and 4 are positions with bounds
        # index 1, 3 and 5 are velocities with bounds -inf, inf
        state[0] = (state[0] - pi) / pi
        state[1] = np.sign(state[1]) * (1 + np.e ** (-0.1 * abs(state[1])))
        state[2] = (state[2] - pi) / pi
        state[3] = (state[3] - pi) / (4 * pi)
        state[4] = (state[4] - pi) / pi
        state[5] = (state[5] - pi) / (4 * pi)
        return state

    def to_json_object(self):
        obj = {}
        obj['description'] = "simulation"
        obj['physical_parameters'] = self.params
        obj['sim_ticks_per_step'] = self.sim_ticks_per_step
        obj['sim_interval'] = self.simulation.interval
        obj['sim_threshold'] = self.simulation.threshold
        obj['sim_max_acceleration'] = self.simulation.max_acceleration
        obj['sim_kp'] = self.simulation.kp
        obj['sim_ka'] = self.simulation.ka
        obj['sim_acceleration_control'] = self.simulation.acceleration_control
        obj['sim_acceleration_limit'] = self.simulation.acceleration_limit
        obj['reward_function'] = {
            'average': self.reward_average,
            'index': self.reward_function_index,
            'parameters': self.reward_function_params
        }
        return obj
