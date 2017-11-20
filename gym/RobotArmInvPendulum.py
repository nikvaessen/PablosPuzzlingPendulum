################################################################################
#
#
#
################################################################################
#
# Author(s): Nik Vaessen
#
# This file is distributed under the MIT license. For a copy, check the LICENSE
# file in the root directory or check https://opensource.org/licenses/MIT.
#
################################################################################

import sys, math
import numpy as np

#import Box2D

import gym
from gym import spaces
from gym.utils import colorize, seeding

################################################################################
# The environment class doing the simulation work

import communcation

class RobotArm(gym.Env):
    """
    The Environment class responsible for the simulation
    """

    def __init__(self, time_step = 15, low = 0, high=1000, intervals =10 ):
        # self.world = Box2D.b2World()
        # body = self.world.CreateBody(Box2D.b2BodyDef())
        # change to find right partition of space
        self.time_step = time_step
        self.potentiometer = self.createMapForJoint(low, high, intervals)
        self.joint1 = (0, 0)
        self.joint2 = (0, 0)
        self.pendulum = (0, 0)

        pass



################################################################################
# Properties which need to be set to create a valid Environment.

    # The action space defines all possible actions which can be taken during
    # one episode of the task
    action_space = None

    # The observation space defines all possible states the environment can
    # take during one episode of a the task
    observation_space = None

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
        return super()._seed(seed)

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
        # State in shape of (j1.pos, j2.pos, p.pos)
        state = communcation.get_state()
        self.joint1 = self.updateJoint(self.joint1, state[0])
        self.joint2 = self.updateJoint(self.joint2, state[1])



        pass

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
        super()._render(mode, close)

    def _reset(self):
        """Resets the state of the environment and returns an initial
           observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        pass

################################################################################
    def createMapForJoint(self, low, high, intervals):
        def f(x):
            range1 = high - low
            x = x - low
            size = range1/intervals
            return x/size - x % size * 1/size

        return map(f, range(low, high))

    def updateJoint(self, joint, new_pos):
        new_pos = self.potentiometer[new_pos]
        new_vel = (joint[0] - new_pos) / self.time_step;
        return new_pos, new_vel

    def updatePendullumState(self, new_position):
        new_pos = self.potentiometer[new_position]
        new_acc = self.pendulum






################################################################################

if __name__ == '__main__':
    print("Hello World")