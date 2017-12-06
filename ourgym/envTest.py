import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../communication'))

from communication.com import Communicator
from time import sleep
from ourgym import RobotArm
import math
import rl.QLearner as ql
from rl.DQNAgent import DQNAgent


port = "/dev/cu.usbserial-A6003X31" #on mac for jose, pablo and nik
if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'

def learn():
    env1 = RobotArm(usb_port=port)
    json = env1.observation_space.to_jsonable(env1.action_space.sample())
    print("an action: " + str(json))

    from ourgym import DiscreteAction
    a = DiscreteAction(256, 50, 130, 5)

    # number areas per space
    # (pendulum, lower motor)
    obs1 = (100, 20, 20)
    print(env1.action_space.n)

    # create bounds for for each observation parameter
    bounds1 = list(zip(env1.observation_space.low, env1.observation_space.high))

    # check size of default bounds
    print("bounds" + str(bounds1))

    learner = ql.Tabular(env1, obs1, bounds1)
    learner.run_n_episodes(100, 1000)

def learn_dqn():
    import numpy as np
    from ourgym import DiscreteAction

    state_size = 3
    action_size = 256
    episodes = 500
    max_episode_length = 1000

    # initialize gym environment and the agent
    env = RobotArm(port)
    agent = DQNAgent(state_size, action_size)
    action_map = DiscreteAction(256, 50, 130, 5)

    # Iterate the environment
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        total_r = 0
        for _ in range(max_episode_length):
            # turn this on if you want to render
            # env.render()

            # Decide action
            action = action_map.get(agent.act(state))

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            total_r += reward
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, total_r))
                break

        # train the agent with the experience of the episode
        agent.replay(32)




def move_test():
    port = "/dev/cu.usbserial-A6003X31"
    if sys.platform == 'linux' or sys.platform == 'linux2':
        port = '/dev/ttyUSB0'
    elif sys.platform == 'win32':
        port = 'COM4'

    #com = Communicator(usb_port=port, baudrate=9600)
    change = 30

    robot = RobotArm(usb_port=port)

    n = 0
    while n < 100:
        n += 1
        #change = -change
        #obot.com.send_command(50, 50)
        #sleep(5)
        print(robot.com.observe_state())
        sleep(5)

    robot.reset()

def reward_test():
    robot = RobotArm(usb_port=port)

    while True:
        s = robot._get_current_state()
        r = robot._reward(s)

        j2 = s[2]
        if j2 > 510:
            j2 = 510 - abs(510 - j2)
        else:
            j2 = 510 + abs(510 - j2)

        target = 600 - (s[1] - 520) + (j2 - 510)
        current = s[0]
        dist = joses_madness(target, current)
        max_dist = 512

        if dist > max_dist:
            r = 0
        else:
            r = 1 - (dist/max_dist)

        print("s:{}, r:{}, c:{}, t:{}, d:{}".format(s, round(r, 5), current, target, dist))
        sleep(1)


def joses_madness(t, c):
    d = 0
    if t + 512 > 1024:
        if c > t:
            d = c - t
        elif c >= t - 512:
            d = t - c
        else:
            d = 1024 - t + c
    else:
        if c < t:
            d = t - c
        elif c <= t + 512:
            d = c -t
        else:
            d = t + (1024 - c)

    return d

def debug_reward():
    robot = RobotArm(usb_port=port)

    while True:
        state = robot._get_current_state()
        reward = robot._reward(state)
        print(state, reward)
        sleep(2)


if __name__ == '__main__':
    learn_dqn()