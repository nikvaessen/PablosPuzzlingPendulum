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

################################################################################
# A Q-learning class in tabular form
################################################################################


class QLearner(object):

    def __init__(self, environment, num_of_obs, state_bounds,
                 learning_rate=0.1, maximum_discount=0.99, exploration_rate=0.01):
        self.env = environment
        self.obs = num_of_obs
        self.bounds = state_bounds
        self.act = self.env.action_space
        self.lr = learning_rate
        self.gamma = maximum_discount
        self.er = exploration_rate
        pass

    @classmethod
    def run_n_episodes(self, n, max_movements_in_episode):
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

    def __init__(self, environment, num_of_obs, state_bounds,
                 learning_rate=0.1, maximum_discount=0.99, exploration_rate=0.01):
        super(Tabular, self).__init__(environment, num_of_obs, state_bounds,
                                      learning_rate, maximum_discount, exploration_rate)
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


################################################################################
#  A Q-learning class in network form
################################################################################

class Network(QLearner):

    def __init__(self, environment, num_of_obs, state_bounds,
                 lr=0.1, maximum_discount=0.99, exploration_rate=0.01):
        super(Network, self).__init__(environment, num_of_obs, state_bounds, lr, maximum_discount, exploration_rate)

        tf.reset_default_graph()
        self.inputs1 = tf.placeholder(dtype=tf.float32, shape=[1, len(num_of_obs)])
        self.W = tf.Variable(tf.random_uniform([len(num_of_obs), self.act.n]))
        self.Qout = tf.matmul(self.inputs1, self.W)
        self.predict = tf.argmax(self.Qout, 1)
        self.nextQ = tf.placeholder(dtype=tf.float32, shape=[1, self.act.n])
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updateModel = self.trainer.minimize(self.loss)
        self.init = tf.initialize_all_variables()

    def run_n_episodes(self, n, max_movements_in_episode):

        with tf.Session() as sess:
            sess.run(self.init)
            for episode in range(n):
                s = self.state_from_obs(self.env.reset())
                movements = 0

                for m in range(max_movements_in_episode):
                    self.env.render()

                    a, allQ = sess.run([self.predict, self.Qout],
                                       feed_dict={self.inputs1:np.identity(len(self.obs))[s:s+1]})
                    a[0] = self.select_action(s,episode)
                    s1, reward, done, _ = self.env.step(a[0])

                    Q1 = sess.run(self.Qout, feed_dict={self.inputs1:np.identity(16)[s1:s1+1]})
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    targetQ[0, a[0]] = reward + self.gamma * maxQ1

                    _, W1 = sess.run([self.updateModel,self.W],
                                    feed_dict={self.inputs1:np.identity(16)[s:s+1], self.nextQ:targetQ})

                    s = s1

                    if done:
                        movements = m
                        break
                if movements is 0:
                    movements = max_movements_in_episode
                print("Episode: {}\n \t finished after {} movements".format(episode+1, movements+1))



