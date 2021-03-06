import sys, math
import numpy as np
import time

import gym
from gym.spaces import Box, Discrete
from communication.com import Communicator
from ourgym import DiscreteAction

################################################################################
# The environment class doing the simulation work

class RobotArmSimulation(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    """
    The Environment class responsible for the simulation
    """
    # TODO rewrite to use with ourgym.spaces.Box

    def __init__(self, usb_port, time_step=10):
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
    action_space = DiscreteAction(49, -30, 31, 15)

    # The observation space defines all possible states the environment can
    # take during one episode of a the task

    observation_space = Box(np.array([0, 256, 256]), np.array([1023, 768, 768]))
    center = np.array([512, 512, 512])
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
        if take_action:
            self.com.send_command(action[0], action[1])

        time.sleep(self.time_step) # wait x ms to observe what the effect of the action was on the state

        state = self._get_current_state()

        reward = self._reward(state)
        done = False

        if self.swing_up:
            if reward > 0.80:
                self.swing_up = False
        else:
            if reward < 0.4:
                done = True

        if reward < 0.8:
            reward = 0


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


