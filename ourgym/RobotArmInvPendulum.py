################################################################################
#
#
#
################################################################################
#
# Author(s): Pablo Soto
#            Nik Vaessen
#            Jose Velasquez
#
# This file is distributed under the MIT license. For a copy, check the LICENSE
# file in the root directory or check https://opensource.org/licenses/MIT.
#
################################################################################

import numpy as np

import gym
from gym.spaces import Box, Discrete
from communication.com import Communicator

from time import sleep, time

# disable numpy printing numbers in scientific notation
np.set_printoptions(suppress=True)

################################################################################
# discrete action space for robot env

class ActionMap:

    def __init__(self, possible_actions):
        idx = 0
        self.actions = [0 for _ in range(0, len(possible_actions)**2)]
        for i in range(len(possible_actions)):

            for j in range(len(possible_actions)):
                #print("init ({}, {}), with index {}".format(possible_actions[i], possible_actions[j], idx))
                self.actions[idx] = (possible_actions[i], possible_actions[j])
                idx += 1

        for a in self.actions:
            print(a)

    def get(self, index):
        return self.actions[index]

    def getIndex(self, action):
        idx = 0
        for a in self.actions:
            if action == a:
                return idx
            else:
                idx += 1

        raise ValueError("Could not get index of action")

class DiscreteAction(Discrete):

    def __init__(self, n, lower, upper, stepsize):
        super(DiscreteAction, self).__init__(n)
        self.actions = [0 for _ in range(0, n)]

        idx = 0
        for i in range(lower, upper, stepsize):
            for j in range(lower, upper, stepsize):
                #print("init ({}, {}), with index {}".format(i, j, idx))
                self.actions[idx] = (i, j)
                idx += 1
        print(self.actions)

    def get(self, index):
        return self.actions[index]

    def getIndex(self, action):
        idx = 0
        for a in self.actions:
            if action == a:
                return idx
            else:
                idx += 1

    def sample(self):
        return self.actions[super(DiscreteAction, self).sample()]


################################################################################
# The environment class doing the simulation work

class RobotArm(gym.Env):
    """
    The Environment class responsible for the simulation
    """
    def __init__(self, usb_port, max_step_count=100, time_step=10/1000):
        # The communicator object responsible for talking with the sensors and motors.
        self.com = Communicator(usb_port=usb_port)

        # The amount of time in ms to wait when a motor command was send.
        self.time_step = time_step

        # storage of robot state. These values are used for computing velocity and to determine
        # end of episodes
        self.prev_pendulum_pos = 0
        self.step_count = 0
        self.max_step_count = max_step_count
        self.swing_up = False

        self._reset()

################################################################################
# Properties which need to be set to create a valid Environment.

    # The action space defines all possible actions which can be taken during
    # one episode of the task.
    action_space_dim = 169
    action_map = DiscreteAction(action_space_dim, -30, 31, 5)
    action_space = Discrete(action_space_dim)

    # The observation space defines all possible states the environment can
    # take during one episode of a the task.
    # The first value is the position of the pendulum.
    # The second value is the position of the lower motor joint.
    # The third value is the position of the upper motor joint.
    # The fourth value is the difference in position of the pendulum
    # between the current and previous state. This value is capped at 1000.
    num_obs_per_step = 6
    num_states_per_obs = 4
    state_dim = num_states_per_obs * num_obs_per_step
    observation_space = Box(np.array([0, 400, 400, 0] * num_obs_per_step).reshape(state_dim),
                            np.array([1024, 600, 600, 1000] * num_obs_per_step).reshape(state_dim))

    # The positions when the pendulum is either pointing upwards (center_up)
    # or pointing down (center_down)
    center_up = np.array([625, 525, 519])
    center_down = np.array([60, 525, 519])

    # The motor joint's angles are bounded to the following positions:
    # lower motor max/left=600     min/right=400
    # upper motor max/left=600     min/right=400

################################################################################
# Abstract methods which need to be overridden to create a valid Environment.
# The methods deal with the reinforcement learning API.

    def _seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return super._seed(seed)

    def _step(self, action, take_action=True):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further
                            step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for
                         debugging, and sometimes learning)
        """
        no_action = False
        illegal_action = False
        if take_action:
            if type(action) != int and action < self.action_space_dim and action >= self.action_space_dim:
                raise ValueError("not a valid action")

            command = self.action_map.get(action)
            at = [self.prev_command[0] + command[0],  self.prev_command[1] + command[1]]

            if at[0] > 140:
                illegal_action = True
                at[0] = 140
            elif at[0] < 40:
                illegal_action = True
                at[0] = 40

            if at[1] > 140:
                illegal_action = True
                at[1] = 140
            elif at[1] < 40:
                illegal_action = True
                at[1] = 40

            #print(action, command, at, self.prev_command)
            if at == self.prev_command:
                print("no_action :)")
                no_action = True

            if np.random.rand() > 0.5:
                self.com.send_command(60, 60)
            else:
                self.com.send_command(120, 120)

            self.prev_command = at
            sleep(0.005)

        st = time()
        states = []
        for i in range(self.num_obs_per_step):
            state = self._get_current_state()
            corrected_state = self._get_current_state()
            states.append(corrected_state)

            if (self.center_up[0] - 100) <= corrected_state[0] <= (self.center_up[0] + 100):
                print("{}, {} is in center".format(state, corrected_state))
                self.swing_up = True

            sleep(0.005)

        reward = 0
        done = True if self.step_count >= self.max_step_count else False

        for st in states:
            r = self._reward(st, no_action, illegal_action)
            if self.swing_up and r < 0.4:
                done = True
                print("swong up and r < 0.4")
            reward += r


        #sleep(self.time_step) # wait x ms to observe what the effect of the action was on the state
        self.step_count += 1

        state = self._get_current_state()

        actual_reward = reward


        # if self.swing_up:
        #     if reward > 0.60:
        #         self.swing_up = False
        # else:
        #     if reward < 0.4:
        #         done = True
        #
        # if reward < 0.4:
        #     reward = -.1
        # elif reward < 0.8:
        #     reward = 0
        # else:
        #     reward *= 3

        return np.array(states).reshape(self.state_dim), reward, done, {'actual_reward' : actual_reward}

    def multi_step(self, action, steps):
        return_list = [None] * steps

        for i in range(0, steps):
            take_action = False
            start_time = time()
            end_time = start_time + 0.02

            if i == 0:
                take_action = True

            state, reward, done, info = self._step(action, take_action=take_action)
            return_list[i] = (state, reward, done, info)

            if done:
                break
            else:
                ct = time()
                print("step took " + str(ct-start_time))
                sleep_for = end_time - ct
                if sleep_for > 0:
                    sleep(sleep_for)
                else:
                    print("### WARNING step took longer than 10 ms")

        return return_list



    def _render(self, mode='human', close=False):
        """Renders the environment.
              The set of supported modes varies per environment. (And some
              environments do not support rendering at all.) By convention,
              if mode is:
              - human: render to the current display or terminal and
                return nothing. Usually for human consumption.
              - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
                representing RGB values for an x-by-y pixel image, suitable
                for turning into a video.
              - ansi: Return a string (str) or StringIO.StringIO containing a
                terminal-style text representation. The text can include
                newlines and ANSI escape sequences (e.g. for colors).
              Note:
                  Make sure that your class's metadata 'render.modes' key
                  includes the list of supported modes. It's recommended to call
                  super() in implementations to use the functionality of this
                  method.
              Args:
                  mode (str): the mode to render with
                  close (bool): close all open renderings
              Example:
              class MyEnv(Env):
                  metadata = {'render.modes': ['human', 'rgb_array']}
                  def render(self, mode='human'):
                      if mode == 'rgb_array':
                          return np.array(...) # return RGB frame suitable for
                          video
                      elif mode is 'human':
                          ... # pop up a window and render
                      else:
                          super(MyEnv, self).render(mode=mode) # just raise an
                          exception
              """
        pass

    def _reset(self):
        """Resets the state of the environment and returns an initial
           observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.com.send_command(90, 90)
        self.swing_up = False
        self.prev_command = (90, 90)

        sleep(7)

        self.state = self._get_current_state()
        self.state[3] = 0
        self.prev_pendulum_pos = 0
        self.step_count = 0
        return np.array([self.state for _ in range(self.num_obs_per_step)]).reshape(self.state_dim)

################################################################################
    def _update_joint(self, joint, new_pos):
        new_vel = (joint[0] - new_pos) / self.time_step
        return new_pos

    def _update_pendulum(self, new_position):
        new_vel = (new_position - self.prev_pendulum_pos) / self.time_step
        return new_position, new_vel

    def _reward(self, corrected_state, no_action, illegal_action):
        """

        :param state:
        :return:
        """
        # Check if pendulum in upper region
        if (self.center_up[0]) - 100 <= corrected_state[0] <= (self.center_up[0] + 100):
            # it is, thus we give a reward between 0 and 1, which is closer to one
            # the slower the velocity
            return np.e ** -abs(0.2* corrected_state[3])
        elif no_action and not illegal_action:
            return 0.1/self.num_obs_per_step
        else:
            return 0

        # highest_target = self.center_up[0]
        # lowest_target = self.center_down[0]
        #
        # max_dist = None
        # pos = corrected_state[0]
        #
        # option = 0
        # if pos >= lowest_target and pos <= highest_target:
        #     option = 1
        #     dist = highest_target - pos
        #     max_dist = highest_target - lowest_target + 1
        # elif pos > highest_target:
        #     option = 2
        #     dist = pos - highest_target
        #     max_dist = lowest_target + (1024 - highest_target) + 1
        # else:
        #     option = 3
        #     max_dist = lowest_target + (1024 - highest_target) + 1
        #     dist = pos + (1024 - highest_target)
        #
        # print(pos, lowest_target, highest_target, dist, max_dist, option)
        # return 1 - (dist / max_dist)


    def _get_current_state(self):
        # State in shape of (j1.pos, j2.pos, p.pos)
        state = self.com.observe_state()
        count = 0
        while not state or state[2] > 900:
            state = self.com.observe_state()
            count += 1
            if count > 10:
                print("cannot read state!!! {}".format(state))

        #print(state)
        pendulum_vel = (state[0] - self.prev_pendulum_pos) * .030
        #print(pendulum_vel)
        pendulum_vel = min(10000, max(-10000, pendulum_vel))
        self.prev_pendulum_pos = state[0]
        self.joint1 = state[1]
        self.joint2 = state[2]
        state = np.array([state[0], self.joint1, self.joint2, pendulum_vel])

        return state


    def pendulum_pos_correction_jose(self, t, c):
        d = 0
        if t + 512 > 1024:
            if c > t:
                d = c - t
            elif c >= t - 512:
                d = t - c
            else:
                d = 1024 - t + c
        else:
            if c < t:
                d = t - c
            elif c <= t + 512:
                d = c - t
            else:
                d = t + (1024 - c)

        return d

    def pendulum_pos_correction(self, state, center_j1=None, center_j2=None):
        """
        Correct the pendulum position in the state due to the potentiometer's center point
        being moved.

        e.g
        If joint 1 is angled 20 units to the right, the center point has to be moved 20 points to the left
        or
        if joint 1 is angled 20 units to the right and joint 2 40 units to the right,
        the center point needs to be pointed 20+40 = 60 units to the left

        :param state the state object, in the order [pendulum pos, joint 1 pos, joint 2 pos, pend velocity]
        :param center_j1 the position where joint 1 is angled 90 degrees upwards
        :param center_j2 the position where joint 2 is angled 90 degrees upwards

        :return a copy of the given state object where the pendulum position has been corrected
        """
        if center_j1 is None:
            center_j1 = self.center_up[2]
        if center_j2 is None:
            center_j2 = self.center_up[1]

        joint1_units_off_center = state[1] - center_j1
        joint2_units_off_center = state[2] - center_j2
        correction = joint1_units_off_center + joint2_units_off_center

        pend_state = (state[0] + correction) % 1024

        corrected_state = np.array(state)
        corrected_state[0] = pend_state

        #print(joint1_units_off_center, joint2_units_off_center, correction)
        return corrected_state

################################################################################

class RobotArmSwingUp(gym.Env):
    """
    The Environment class responsible for the simulation
    """
    def __init__(self, usb_port, max_step_count=100, time_step=10/1000):
        # The communicator object responsible for talking with the sensors and motors.
        self.com = Communicator(usb_port=usb_port)

        # The amount of time in ms to wait when a motor command was send.
        self.time_step = time_step

        # storage of robot state. These values are used for computing velocity and to determine
        # end of episodes
        self.prev_pendulum_pos = 0
        self.prev_pendulum_pos_time = time()
        self.step_count = 0
        self.max_step_count = max_step_count
        self.swing_up = False

        self._reset()

################################################################################
# Properties which need to be set to create a valid Environment.

    # The action space defines all possible actions which can be taken during
    # one episode of the task.
    action_space_dim = 25
    action_map = DiscreteAction(action_space_dim, 50, 131, 20)
    action_space = Discrete(action_space_dim)

    # The observation space defines all possible states the environment can
    # take during one episode of a the task.
    # The first value is the position of the pendulum.
    # The second value is the position of the lower motor joint.
    # The third value is the position of the upper motor joint.
    # The fourth value is the difference in position of the pendulum
    # between the current and previous state. This value is capped at 1000.
    num_obs_per_step = 6
    num_states_per_obs = 4
    state_dim = num_states_per_obs * num_obs_per_step
    observation_space = Box(np.array([0, 400, 400, 0] * num_obs_per_step).reshape(state_dim),
                            np.array([1024, 600, 600, 1000] * num_obs_per_step).reshape(state_dim))

    # The positions when the pendulum is either pointing upwards (center_up)
    # or pointing down (center_down)
    center_up = np.array([625, 525, 519])
    center_down = np.array([60, 525, 519])

    # The motor joint's angles are bounded to the following positions:
    # lower motor max/left=600     min/right=400
    # upper motor max/left=600     min/right=400

################################################################################
# Abstract methods which need to be overridden to create a valid Environment.
# The methods deal with the reinforcement learning API.

    def _seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return super._seed(seed)

    def _step(self, action, take_action=True):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further
                            step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for
                         debugging, and sometimes learning)
        """
        if type(action) != int and action < self.action_space_dim and action >= self.action_space_dim:
            raise ValueError("not a valid action")

        # take an action
        command = self.action_map.get(action)
        self.com.send_command(command[0], command[1])
        sleep(0.005)

        # observe the state a few times
        st = time()
        states = []
        for i in range(self.num_obs_per_step):
            state = self._get_current_state()
            corrected_state = self._get_current_state()
            states.append(corrected_state)

            if (self.center_up[0] - 100) <= corrected_state[0] <= (self.center_up[0] + 100):
                print("{}, {} is in center".format(state, corrected_state))
                self.swing_up = True

            sleep(0.005)

        # see if there was a swing up
        reward = 0
        done = True if self.step_count >= self.max_step_count else False

        for st in states:
            reward = self._reward(st)
            if reward > -1:
                done = True
                break

        # update the step counter and get the state
        self.step_count += 1
        state = self._get_current_state()

        return np.array(states).reshape(self.state_dim), reward, done, {}

    def _render(self, mode='human', close=False):
        pass

    def _reset(self):
        """Resets the state of the environment and returns an initial
           observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.com.send_command(90, 90)
        self.swing_up = False

        sleep(7)

        self.state = self._get_current_state()
        self.state[3] = 0
        self.prev_pendulum_pos = 0
        self.prev_pendulum_pos_time = time()
        self.step_count = 0

        return np.array([self.state for _ in range(self.num_obs_per_step)]).reshape(self.state_dim)

################################################################################

    def _reward(self, corrected_state):
        """

        :param state:
        :return:
        """
        # Check if pendulum in upper region
        if (self.center_up[0] - 30) <= corrected_state[0] <= (self.center_up[0] + 30):
            print("in region {}".format(corrected_state))
            if abs(corrected_state[3]) < 1:
                return 50

        return -1

    def _get_current_state(self):
        # State in shape of (j1.pos, j2.pos, p.pos)
        state = self.com.observe_state()
        count = 0
        while not state or state[2] > 900:
            state = self.com.observe_state()
            count += 1
            if count > 10:
                print("cannot read state!!! {}".format(state))

        dt = (time() - self.prev_pendulum_pos_time)
        pendulum_vel = (state[0] - self.prev_pendulum_pos) * dt
        pendulum_vel = min(10000, max(-10000, pendulum_vel))

        if pendulum_vel < -50:
            pendulum_vel = ((1023 - self.prev_pendulum_pos) + state[0]) * dt
        elif pendulum_vel > 50:
            pendulum_vel = (self.prev_pendulum_pos + (1023 - self.prev_pendulum_pos)) * dt


        self.prev_pendulum_pos = state[0]
        self.prev_pendulum_pos_time = time()
        self.joint1 = state[1]
        self.joint2 = state[2]
        state = np.array([state[0], self.joint1, self.joint2, pendulum_vel])

        return state

    def pendulum_pos_correction(self, state, center_j1=None, center_j2=None):
        """
        Correct the pendulum position in the state due to the potentiometer's center point
        being moved.

        e.g
        If joint 1 is angled 20 units to the right, the center point has to be moved 20 points to the left
        or
        if joint 1 is angled 20 units to the right and joint 2 40 units to the right,
        the center point needs to be pointed 20+40 = 60 units to the left

        :param state the state object, in the order [pendulum pos, joint 1 pos, joint 2 pos, pend velocity]
        :param center_j1 the position where joint 1 is angled 90 degrees upwards
        :param center_j2 the position where joint 2 is angled 90 degrees upwards

        :return a copy of the given state object where the pendulum position has been corrected
        """
        if center_j1 is None:
            center_j1 = self.center_up[2]
        if center_j2 is None:
            center_j2 = self.center_up[1]

        joint1_units_off_center = state[1] - center_j1
        joint2_units_off_center = state[2] - center_j2
        correction = joint1_units_off_center + joint2_units_off_center

        pend_state = (state[0] + correction) % 1024

        corrected_state = np.array(state)
        corrected_state[0] = pend_state

        #print(joint1_units_off_center, joint2_units_off_center, correction)
        return corrected_state




