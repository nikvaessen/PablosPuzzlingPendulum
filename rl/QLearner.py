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
import tensorflow as tf
import time
import os

from gym import Env
from gym.spaces import Discrete, Box
from keras.models import Sequential
from keras.layers import Dense

################################################################################
# A Q-learning class in tabular form
################################################################################


class QLearner(object):

    def __init__(self, environment:Env, num_of_obs, state_bounds,
                 learning_rate=0.1, maximum_discount=0.99, exploration_rate=0.01):
        self.env = environment
        self.obs = num_of_obs
        self.bounds = state_bounds
        self.act = self.env.action_space
        self.lr = learning_rate
        self.gamma = maximum_discount
        self.er = exploration_rate
        self.start_time = time.time()
        self.exploring = True
        pass

    @classmethod
    def run_n_episodes(self, n, max_movements_in_episode, offline=False):
        pass

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
                bound_width = self.bounds[i][1] - self.bounds[i][0]
                offset = (self.obs[i]-1) * self.bounds[i][0]/bound_width
                scaling = (self.obs[i]-1)/bound_width
                index = int(round(scaling * obs[i] - offset))
            indice.append(index)
        # print("observation: {} yields state: {}".format(obs, tuple(indice)))
        return tuple(indice)

    def select_action(self, state, episode, offline):
        """
        selects an action depending on the episode
        we assume that new information is learned after every information
        :param state: current step
        :param episode: current episode
        :param offline:
        :return: an action on the environment
        """
        if offline:
            # TODO
            if time.time() - self.start_time > 10:
                self.exploring = False
                print("took argmax")
                return self.act.get(np.argmax(self.Q[state]))
            else:
                print("random sample")
                return self.act.sample()

        else:
            if r.random() < self.get_explore_rate(episode):
                print("random sample")
                return self.act.sample()
            else:
                print("took argmax")
                return self.act.get(np.argmax(self.Q[state]))

    def get_explore_rate(self, episode):
        """
        selects an exploration rate depending on the episode
        we assume that new information is learned after every information
        :param episode: the current episode
        :return: an exploration rate
        """
        return max(self.er, min(1.0, 1.0 - math.log10((episode+1)/25)))

    def get_learning_rate(self, episode):
        """
        selects a learning rate depending on the episode
        we assume that new information is learned after every information
        :param episode: the current episode
        :return: a learning rate
        """
        return max(self.lr, min(0.5, 1.0 - math.log10((episode+1)/25)))


class Tabular(QLearner):
    """
    A tabular implementation of Q-Learning
    """

    q_table_file_name = "qTable.npy"
    use_multi_step = True
    step_size = 5

    def __init__(self, environment:Env, num_of_obs, state_bounds,
                 learning_rate=0.1, maximum_discount=0.99, exploration_rate=0.01):
        super(Tabular, self).__init__(environment, num_of_obs, state_bounds,
                                      learning_rate, maximum_discount, exploration_rate)
        n = 0
        if isinstance(self.act, Discrete):
            n = self.act.n
        elif isinstance(self.act, Box):
           raise EnvironmentError("ah man we have to implement this shit")
        else:
            raise EnvironmentError("man your action space is fucked up")

        if os.path.exists(self.q_table_file_name):
            self.Q = np.load(self.q_table_file_name)
            print("Loaded q table from memory")
        else:
            self.Q = np.zeros(self.obs + (n,))
            print("reinitialized the q-table")

        print("shape of Q is {}".format(self.Q.shape))

    def run_n_episodes(self, n, max_movements_in_episode, offline=False):
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
            print("###### episode {} ######".format(episode))

            for m in range(max_movements_in_episode):
                print("# " + str(m) )
                #self.env.render()

                # an Action a
                a = self.select_action(s, episode, offline)
                print(a)

                done = None

                if self.use_multi_step:
                    rlist = self.env.multi_step(a, self.step_size)

                    for t in rlist:
                        observation, reward, done, _ = t

                        s1 = self.state_from_obs(observation)
                        print(s1, reward)

                        # Update the Q based on the result
                        best = np.amax(self.Q[s1])
                        self.Q[s + (a,)] += lr * (reward + self.gamma * best - self.Q[s + (a,)])

                        s = s1

                        if done:
                            break
                else:
                    observation, reward, done, _ = self.env.step(a)
                    print(observation, a, reward, done)
                    print()

                    s1 = self.state_from_obs(observation)
                    #print(s1, reward)

                    # Update the Q based on the result
                    best = np.amax(self.Q[s1])
                    self.Q[s + (a,)] += lr * (reward + self.gamma * best - self.Q[s + (a,)])

                    s = s1

                print()
                if done:
                    if self.exploring is False:
                        self.start_time = time.time()
                        self.exploring = True
                    movements = m
                    if episode % 10 is 0:
                        np.save(self.q_table_file_name, self.Q)
                    break

            if movements is 0:
                movements = max_movements_in_episode

            print("Episode: {}\n \t finished after {} movements".format(episode+1, movements+1))

        return self.Q


################################################################################
#  A Q-learning class in network form
################################################################################

class Network(QLearner):

    def __init__(self, environment, num_of_obs, state_bounds,
                 lr=0.1, maximum_discount=0.99, exploration_rate=0.01):
        super(Network, self).__init__(environment, num_of_obs, state_bounds, lr, maximum_discount, exploration_rate)
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights


    def run_n_episodes(self, n, max_movements_in_episode, offline=False):

            for episode in range(n):
                s = self.state_from_obs(self.env.reset())
                movements = 0

                for m in range(max_movements_in_episode):
                    self.env.render()

                    s1, reward, done, _ = self.env.step()
                    s = s1

                    if done:
                        movements = m
                        break
                if movements is 0:
                    movements = max_movements_in_episode
                print("Episode: {}\n \t finished after {} movements".format(episode+1, movements+1))


    def build_network(self):
        model = Sequential()
        model.add(Dense(self.act.n, activation='relu', input_shape=self.obs))
        s = tf.placeholder(dtype=tf.float32, shape=self.obs)
        q_values = model(s)

        return s, q_values, model
