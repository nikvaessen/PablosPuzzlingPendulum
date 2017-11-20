################################################################################
#
#
#
################################################################################
#
# Author(s): Jose Velasquez
#
# This file is distributed under the MIT license. For a copy, check the LICENSE
# file in the root directory or check https://opensource.org/licenses/MIT.
#
################################################################################
import numpy as np
import random as r
import math

################################################################################
# A Q-learning class in tabular form


class Tabular(object):
    """
    A tabular implementation of Q-Learning
    """

    def __init__(self, environment, num_of_obs, state_bounds,
                 learning_rate=0.1, maximum_discount=0.99, exploration_rate=0.01):
        self.env = environment
        self.obs = num_of_obs
        self.bounds = state_bounds
        self.act = self.env.action_space
        self.lr = learning_rate
        self.gamma = maximum_discount
        self.er = exploration_rate
        self.Q = np.zeros(self.obs + (self.act.n,))

    def run_n_episodes(self, n, max_movements_in_episode):
        """
        Runs a simulation n times to update the Q-table
        :param n: number of episodes
        :param max_movements_in_episode: maximum length of episode
        :return: the updated Q-table
        """

        for episode in range(n):
            s = self.state_from_obs(self.env.reset())
            lr = self.get_learning_rate(episode)
            movements = 0

            for m in range(max_movements_in_episode):
                self.env.render()

                # an Action a
                a = self.select_action(s, episode)

                observation, reward, done, _ = self.env.step(a)

                s1 = self.state_from_obs(observation)

                # Update the Q based on the result
                best = np.amax(self.Q[s1])
                self.Q[s + (a,)] += lr * (reward + self.gamma * best - self.Q[s + (a,)])

                s = s1

                if done:
                    movements = m
                    break
            if movements is 0:
                movements = max_movements_in_episode
            print("Episode: {}\n \t finished after {} movements".format(episode+1, movements+1))
        return self.Q

    def state_from_obs(self, obs):
        """
        derives the discrete state from an observation of the environment
        :param obs: an observation from the environment
        :return: a discrete state
        """
        indice = []
        for i in range(len(obs)):
            if obs[i] <= self.bounds[i][0]:
                index = 0
            elif obs[i] >= self.bounds[i][1]:
                index = self.obs[i] - 1
            else:
                bound_width = self.bounds[i][0] - self.bounds[i][1]
                offset = (self.obs[i]-1) * self.bounds[i][0]/bound_width
                scaling = (self.obs[i]-1)/bound_width
                index = int(round(scaling * obs[i] - offset))
            indice.append(index)
        # print("observation: {} yields state: {}".format(obs, tuple(indice)))
        return tuple(indice)

    def select_action(self, state, episode):
        """
        selects an action depending on the episode
        we assume that new information is learned after every information
        :param state: current step
        :param episode: current episode
        :return: an action on the environment
        """
        if r.random() < self.get_explore_rate(episode):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def get_explore_rate(self, episode):
        """
        selects an exploration rate depending on the episode
        we assume that new information is learned after every information
        :param episode: the current episode
        :return: an exploration rate
        """
        return max(self.er, min(1.0, 1.0 - math.log10((episode+1)/25)))

    def get_learning_rate(self,episode):
        """
        selects a learning rate depending on the episode
        we assume that new information is learned after every information
        :param episode: the current episode
        :return: a learning rate
        """
        return max(self.lr, min(0.5, 1.0 - math.log10((episode+1)/25)))

################################################################################
#  A Q-learning class in network form


class Network(object):
    # TODO implement a neural network
    pass
