import gym
import time

from rl import DQNAgent

from ourgym import DiscreteAction

class ActionMap(DiscreteAction):

    def __init__(self):
        pass

    def get(self, index):
        return index

    def getIndex(self, action):
        return action

if __name__ == '__main__':

    env = gym.make('BipedalWalker-v2')
    observation = env.reset()

    print(observation, type(observation), observation.shape)
    print(env.action_space)
    print(env.action_space.sample)

    agent = DQNAgent(24, 4, ActionMap())

    while True:
        env.render(mode="human", close=False)

        action = env.action_space.sample()
        observation, reward, done, info = env.step(env.action_space.sample())

        print(action, reward, done)

        if done:
            break

        time.sleep(1/60)

