import numpy as np
import random as r
import math


class QLearner(object):

    def __init__(self, environment, num_of_obs, state_bounds,
                 learning_rate, maximum_discount, exploration_rate):
        self.env = environment
        self.obs = num_of_obs
        self.bounds = state_bounds
        self.act = self.env.action_space
        self.lr = learning_rate
        self.gamma = maximum_discount
        self.er = exploration_rate
        self.Q = np.zeros(self.obs + (self.act.n,))
        print(self.Q.shape)

    def run_n_episodes(self, n, max_movements_in_episode):

        for episode in range(n):
            s = self.state_from_obs(self.env.reset())
            lr = self.get_learning_rate(episode)

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
                    # print("Episode {} finished after {} movements".format(episode, m))
                    break
        print(self.Q)

    def state_from_obs(self, obs):
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
        if r.random() < self.get_explore_rate(episode):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    # get exploration rate depending on episode
    def get_explore_rate(self, episode):
        return max(self.er, min(1.0, 1.0 - math.log10((episode+1)/25)))

    # get learning rate depending on episode
    def get_learning_rate(self,episode):
        return max(self.lr, min(0.5, 1.0 - math.log10((episode+1)/25)))

