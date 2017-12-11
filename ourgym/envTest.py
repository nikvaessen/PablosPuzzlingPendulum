import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../communication'))

from communication.com import Communicator
from time import sleep, time
from ourgym import RobotArm
import math
import rl.QLearner as ql
from rl.Agent import DQNAgent


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
    action_size = 49
    episodes = 500
    max_episode_length = 1000  # 1000 / (1 / 0.025) = 25 secs
    iteration_length = 0.030
    safe_every = 5

    past_action = [90, 90]

    weight_file = "" # set manually each time

    # initialize gym environment and the agent
    env = RobotArm(port, time_step=0.0015)
    print("created robot")
    action_map = DiscreteAction(49, -30, 31, 15)
    agent = DQNAgent(state_size, action_size, action_map)

    if weight_file is not "":
        agent.load(weight_file)
        print("loaded " + str(weight_file))

    # Iterate the environment
    for e in range(1, episodes + 1):
        # reset state in the beginning of each episode
        state = env.reset()

        # time_r represents the sum of the reward over the episode
        total_r = 0
        for moves in range(max_episode_length):
            # decide when this iteration should be over
            start_time = time()
            desired_end_time = start_time + iteration_length

            # Decide action
            action = agent.act(state)

            # Advance the environment to the next frame based on the action.
            # Reward is bases on the angle of the pendulum
            real_action = add_action_to_position(action, past_action)
            next_state, reward, done, _ = env.step(real_action)

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # sleep for the remaining time left, or warn when time-limit was exceeded
            total_r += reward
            #print("move {}: {}, {}, {}, {}, {}".format(moves + 1, state, action, reward, next_state, done))
            ct = time()
            past_action = real_action

            if ct < desired_end_time:
                sleep(desired_end_time - ct)
            else:
                print("### warning took to long !!!! off by: {}".format(ct - desired_end_time))

            # done becomes True when the pendulum was swung up but fell down
            if done or moves+1 == max_episode_length:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, moves: {}"
                      .format(e, episodes, total_r, moves + 1))
                break

        # train the agent with the experience of the episode
        if len(agent.memory) > 1000:
            agent.replay(1000)

        if e % safe_every == 0:
            agent.safe()


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
        change = -change
        #robot.com.send_command(90 + change, 90 + change)
        #sleep(5)
        print(robot.com.observe_state())
        sleep(0.2)

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

def add_action_to_position(action, past_action):
    # print("action: {}, past actions: {}".format(action, past_action))
    if action is 0:
        action = (0, 0)

    if past_action[0] + action[0] <= 50:
        a1 = 50
    elif past_action[0] + action[0] >= 130:
        a1 = 130
    else:
        a1 = past_action[0] + action[0]

    if past_action[1] + action[1] <= 50:
        a2 = 50
    elif past_action[1] + action[1] >= 130:
        a2 = 130
    else:
        a2 = past_action[1] + action[1]

    return a1, a2

def debug_reward():
    robot = RobotArm(usb_port=port)

    while True:
        state = robot._get_current_state()
        reward = robot._reward(state)
        print(state, reward)
        sleep(0.1)


if __name__ == '__main__':
    learn_dqn()
