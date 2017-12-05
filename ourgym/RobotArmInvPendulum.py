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

import sys, math
import numpy as np
import time
#import Box2D

import gym
from gym.spaces import Box, Discrete
from communication.com import Communicator

################################################################################
# discrete action space for robot env


class DiscreteAction(Discrete):

    def __init__(self, n, lower, upper, stepsize):
        super(DiscreteAction, self).__init__(n)
        self.actions = [0 for _ in range(0, n)]

        idx = 0
        for i in range(lower, upper, stepsize):
            for j in range(lower, upper, stepsize):
                self.actions[idx] = (i, j)
                idx += 1

    def get(self, index):
        return self.actions[index]

    def sample(self):
        return self.actions[super(DiscreteAction, self).sample()]


################################################################################
# The environment class doing the simulation work


class RobotArm(gym.Env):
    """
    The Environment class responsible for the simulation
    """
    # TODO rewrite to use with ourgym.spaces.Box

    def __init__(self, usb_port, time_step=15):
        # self.world = Box2D.b2World()
        # body = self.world.CreateBody(Box2D.b2BodyDef())
        # change to find right partition of space
        self.com = Communicator(usb_port=usb_port)
        self.time_step = time_step
        self.joint1 = (0, 0)
        self.joint2 = (0, 0)
        self.pendulum = (0, 0)
        self.max_distance = np.linalg.norm(self.center - self.observation_space.low)
        self.swing_up = True

################################################################################
# Properties which need to be set to create a valid Environment.

    # The action space defines all possible actions which can be taken during
    # one episode of the task
    action_space = DiscreteAction(256, 50, 130, 5)

    # The observation space defines all possible states the environment can
    # take during one episode of a the task

    observation_space = Box(np.array([0, 400, 400]), np.array([1020, 600, 600]))
    center = np.array([600, 530, 510])
    # lower motor max/left=600     min/right=400
    # upper motor max/left=600     min/right=400
    # center = [600, 530, 510]

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

    def _step(self, action):
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
        self.com.send_command(action[0], action[1])
        time.sleep(0.1)
        state = self._get_current_state()

        reward = self._reward(state)
        done = False

        if self.swing_up:
            if reward > 0.8:
                self.swing_up = False
        else:
            if reward < 0.4:
                done = True

        return state, reward, done, {}


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
        self.swing_up = True
        time.sleep(8)
        return self._get_current_state()

################################################################################
    def _update_joint(self, joint, new_pos):
        new_vel = (joint[0] - new_pos) / self.time_step
        return new_pos

    def _update_pendulum(self, new_position):
        new_vel = (new_position - self.pendulum[0]) / self.time_step
        return new_position, new_vel

    def _reward(self, state):
        j2 = state[2]
        if j2 > 510:
            j2 = 510 - abs(510 - j2)
        else:
            j2 = 510 + abs(510 - j2)

        target = 600 - (state[1] - 520) + (j2 - 510)
        current = state[0]
        dist = self.joses_madness(target, current)
        max_dist = 512

        if dist > max_dist:
            return 0
        else:
            return 1 - (dist / max_dist)


    def _get_current_state(self):
        # State in shape of (j1.pos, j2.pos, p.pos)
        state = self.com.observe_state()
        self.pendulum = state[0]
        self.joint1 = state[1]
        self.joint2 = state[2]
        state = np.array([self.pendulum, self.joint1, self.joint2])
        return state

    def joses_madness(self, t, c):
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

################################################################################


