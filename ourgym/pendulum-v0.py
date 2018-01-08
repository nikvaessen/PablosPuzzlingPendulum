import gym
import time
import numpy as np
import random

from rl import DQNAgent

from ourgym import DiscreteAction

class ActionMap:

    def __init__(self, total_actions):
        self.total_actions = total_actions + 1
        self.possibleActions = np.linspace(-2, 2, self.total_actions)

    def get(self, index):
        return self.possibleActions[index]

    def getIndex(self, action):
        a = np.where(np.isclose(action, self.possibleActions) == True)

        if len(a[0]) > 0:
            return a[0][0]
        else:
            raise ValueError("action {} does not exist".format(action))

    def sample(self):
        return self.possibleActions[random.randint(0, self.total_actions - 2)]


def calculate_mean_reward(agent, env):
    episodes = 100
    total_reward = 0
    for i in range(episodes):
        print(i + 1)
        observation = env.reset()

        while True:
            action = agent.act(observation)
            new_observation, reward, done, info = env.step(env.action_space.sample())

            agent.remember(observation, action, reward, new_observation, done)

            total_reward += reward
            if done:
                agent.replay(100)
                break

    return total_reward

if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    observation = env.reset()

    print(observation, type(observation), observation.shape)
    print(env.action_space)
    print(env.action_space.sample)

    dim_action = 40
    dim_state = 3

    am = ActionMap(dim_action)
    agent = DQNAgent(dim_state, dim_action, am)

    high = np.array([1., 1., 8.])
    low = -high

    #print("mean 100 episode reward before learning: {}".format(calculate_mean_reward(agent, env)))

    episodes = 1000
    for i in range(episodes):
        print(i)
        observation = env.reset()

        while True:
            env.render(mode="human")
            action = agent.act(observation)
            new_observation, reward, done, info = env.step(env.action_space.sample())

            agent.remember(observation, action, reward, new_observation, done)

            if done:
                agent.replay(100)
                break

    print("mean 100 episode reward after learning: {}".format(calculate_mean_reward(agent, env)))
