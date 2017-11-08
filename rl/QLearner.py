import numpy as np

class QLearner(object):

    def __init__(self, env, learning_rate, maximum_discount):
        self.env = env
        self.obs_s = env.observation_space
        self.act_s = env.action_space
        self.Q = np.zeros(self.obs_s.n, self.act_s.n)
        self.lr = learning_rate
        self.gamma = maximum_discount

    def run_n_episodes(self, n, initial_state, max_movements_in_episode):

        for i in range(n):
            s = initial_state
            r_all = 0
            d = False
            j = 0

            while j < max_movements_in_episode or d is True:
                j += 0
                # choose action by greedily (with noise) picking from Q
                a = np.argmax(self.Q[s, :] + np.random.randn(1, self.act_s.n) * (1./(i+1)))

                # Get new state and reward from environment
                s1, r, d, _ = self.env.step(a)

                self.Q[s, a] += self.lr * (r + self.gamma * np.max(self.Q[s1, :]) - self.Q[s, a])
                r_all += r
                s = s1

        self.rList.append(r_all)