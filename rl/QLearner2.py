from gym.spaces import Discrete
from gym import Env
import random
import numpy as np
import os

class DAction(Discrete):

    def __init__(self, n, actions):
        super(DAction, self).__init__(n)
        self.actions = actions

    def get(self, pos, action):
        return self.actions(pos, action)

    def sample(self, pos):
        return self.actions(pos, random.randint(0, self.n))


class DObservation(object):

    def __init__(self, observations, bounds):
        self.observations = observations
        self.bounds = bounds

    def state_from_readings(self, readings):
        """
        derives the discrete state from an observation of the environment
        :param readings: an observation from the environment
        :return: a discrete state
        """
        indice = []
        for i in range(len(readings)):
            if readings[i] <= self.bounds[i][0]:
                index = 0
            elif readings[i] >= self.bounds[i][1]:
                index = self.observations[i] - 1
            else:
                bound_width = self.bounds[i][1] - self.bounds[i][0]
                offset = (self.observations[i]-1) * self.bounds[i][0]/bound_width
                scaling = (self.observations[i]-1)/bound_width
                index = int(round(scaling * readings[i] - offset))
            indice.append(index)
        # print("observation: {} yields state: {}".format(obs, tuple(indice)))
        return tuple(indice)


class Learner(object):

    QFile = "Q.npy"

    def __init__(self, env: Env, actions: DAction, observations: DObservation):
        self.env = env
        self.actions = actions
        self.observations = observations
        if os.path.exists(self.QFile):
            self.Q = np.load(self.QFile)
            print("Loaded q table from memory")
        else:
            self.Q = np.zeros(observations.observations + (actions.n,))

    def run_epochs(self, epochs, max_movement, lr, gamma, eps):
        for epoch in range(epochs):
            s = self.observations.state_from_readings(self.env.reset())
            movements = 0
            past_action = (90, 90)

            for m in range(max_movement):
                if random.random() < eps:
                    action = self.actions.sample(past_action)
                else:
                    action = self.actions.get(past_action, np.argmax(self.Q[s]))

                past_action = action
                observation, reward, done, _ = self.env.step(action)

                s1 = self.observations.state_from_readings(observation)

                best = np.amax(self.Q[s1])
                self.Q[s + (action,)] += lr * (reward + gamma * best - self.Q[s + (action,)])

                s = s1

                if done:
                    movements = m
                    if m % 5 is 0:
                        np.save(self.QFile, self.Q)

            if movements is 0:
                movements = max_movement

            print("Episode: {}\n \t finished after {} movements".format(epoch+1, movements+1))


class Learner2(object):

    swingQ = "swingQ.npy"
    balanceQ = "balanceQ.npy"

    def __init__(self, env: Env,
                 actions1: DAction, observations1: DObservation,
                 actions2: DAction, observations2: DObservation):
        self.env = env
        self.actions1 = actions1
        self.actions2 = actions2
        self.observations1 = observations1
        self.observations2 = observations2
        self.swingingUp = True

        if os.path.exists(self.swingQ):
            self.Q1 = np.load(self.swingQ)
        else:
            self.Q1 = np.zeros(observations1.observations + (actions1.n,))

        if os.path.exists(self.balanceQ):
            self.Q2 = np.load(self.balanceQ)
        else:
            self.Q2 = np.zeros(observations2.observations + (actions2.n,))

    def run_epochs(self, epochs, max_movement, lr, gamma, eps, max_balance_vel=0.1):
        for epoch in range(epochs):
            s = self.observations.state_from_readings(self.env.reset())
            movements = 0
            past_action = (90, 90)
            for m in range(max_movement):
                if self.swingingUp:
                    self.Q = self.Q1
                else:
                    self.Q = self.Q2

                if random.random() < eps:
                    action = self.actions.sample(past_action)
                else:
                    action = self.actions.get(past_action, np.argmax(self.Q[s]))

                past_action = action
                observation, reward, done, _ = self.env.step(action)

                s1 = self.observations.state_from_readings(observation)

                best = np.amax(self.Q[s1])
                self.Q[s + (action,)] += lr * (reward + gamma * best - self.Q[s + (action,)])

                s = s1

                if done:
                    movements = m
                    if self.swingingUp:
                        self.Q1 = self.Q
                        if reward > 0.8 and observation[1] < max_balance_vel:
                            self.swingingUp = False
                    else:
                        self.Q2 = self.Q

                    if m % 5 is 0:
                        np.save(self.swingQ, self.Q1)
                        np.save(self.balanceQ, self.Q2)

            if movements is 0:
                movements = max_movement

            print("Episode: {}\n \t finished after {} movements".format(epoch+1, movements+1))


class Learner3(object):
    QFile = "myQ.npy"
    example = "example1.npy"

    def __init__(self, env: Env, n, observations: DObservation):
        self.env = env
        self.observations = observations
        if os.path.exists(self.QFile):
            #if os.path.exists(self.example):
            self.Q = np.load(self.QFile)
            #self.Q = np.load(self.example)
            print("Loaded q table from memory")
        else:
            self.Q = np.zeros(observations.observations + (n,))

    def run_epochs(self, epochs, max_movement, lr, gamma):
        for epoch in range(epochs):
            randoms = 0
            s = self.observations.state_from_readings(self.env.reset())
            movements = 0

            for m in range(max_movement):
                #self.env.render()

                if random.random() < 1 - epoch/epochs:
                    #if False:
                    randoms = randoms + 1
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[s])

                observation, reward, done, _ = self.env.step(action)

                s1 = self.observations.state_from_readings(observation)

                best = np.amax(self.Q[s1])
                self.Q[s + (action,)] += lr * (reward + gamma * best - self.Q[s + (action,)])

                s = s1

                if done:
                    movements = m
                    if m % 5 is 0:
                        np.save(self.QFile, self.Q)
                        print("saved table")

            if movements is 0:
                movements = max_movement

            print("Episode: {}\n \t finished after {} movements. With {} of moves random"
                  .format(epoch+1, movements+1, randoms/movements))
            np.save(self.QFile, self.Q)
