import gym
import time

if __name__ == '__main__':

    env = gym.make('BipedalWalker-v2')
    observation = env.reset()

    print(observation)
    print(env.action_space)
    print(env.action_space.sample)

    while True:
        env.render(mode="human", close=False)

        observation, reward, done, info = env.step(env.action_space.sample())
        time.sleep(1)

