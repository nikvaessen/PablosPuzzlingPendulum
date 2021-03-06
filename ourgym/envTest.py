import sys
import os.path
sys.path.append('../')

from communication.com import Communicator
from time import sleep, time
from ourgym import RobotArm, RobotArmSwingUp
import math
import numpy as np
import rl.QLearner as ql
from rl.Agent import DQNAgent

if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'
else:
    port = "/dev/cu.usbserial-A6003X31"

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

    num_episodes = 5000
    num_steps = 200
    memory_size = 10000
    batch_size = 64
    e_start = 1.0
    e_finish = 0.05
    e_decay_steps = 4500
    dr = 0.995
    lr = 0.0001
    layers = 2
    nodes = (20, 20)
    frequency_updates = 0

    iteration_length = 0.030

    weight_file = "" # set manually each time
    weight_path = "backup/" + weight_file

    # initialize gym environment and the agent
    env = RobotArm(port, time_step=iteration_length)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("created robot")
    agent = DQNAgent(env,
                     state_dim,
                     action_dim,
                     memory_size,
                     e_start,
                     e_finish,
                     e_decay_steps,
                     dr,
                     lr,
                     layers,
                     nodes,
                     frequency_updates)

    if weight_file is not "":
        agent.load(weight_path)
        print("loaded " + str(weight_path))

    # Iterate the environment
    for e in range(1, num_episodes + 1):
        print("Starting episode {}".format(e))
        # reset state in the beginning of each episode
        state = norm_state(env.reset())

        # time_r represents the sum of the reward over the episode
        total_r = 0
        for moves in range(num_steps):
            # decide when this iteration should be over
            start_time = time()
            desired_end_time = start_time + iteration_length

            # Decide action
            action = env.action_space.action_map.get(agent.act(state))

            # Advance the environment to the next frame based on the action.
            # Reward is bases on the angle of the pendulum
            next_state, reward, done, _ = env.step(action)
            #print(next_state)
            next_state = norm_state(next_state)
            #print(next_state)

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # sleep for the remaining time left, or warn when time-limit was exceeded
            total_r += reward
            #print("move {}: {}, {}, {}, {}, {}".format(moves + 1, state, action, reward, next_state, done))
            ct = time()

            if ct < desired_end_time:
                sleep_time = desired_end_time - ct
                if sleep_time < 0:
                    print("sleeping for: " + str(sleep_time))
                sleep(sleep_time)
            else:
                print("### warning took to long !!!! off by: {}".format(ct - desired_end_time))

            # done becomes True when the pendulum was swung up but fell down
            if done or moves + 1 == num_steps:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, moves: {}, lr: {}"
                      .format(e, e, total_r, moves + 1, agent.epsilon))
                break

        # train the agent with the experience of the episode
        if len(agent.memory) >= 50:
            agent.replay(50)

        # if e % safe_every == 0:
        #     agent.safe()


def agent_non_random(env, agent, max_episode_length, iteration_length, past_action):
    # reset state in the beginning of each episode
    state = norm_state(env.reset())

    # time_r represents the sum of the reward over the episode
    total_r = 0
    for moves in range(max_episode_length):
        # decide when this iteration should be over
        start_time = time()
        desired_end_time = start_time + iteration_length

        # Decide action
        action = agent.act(state, use_random_chance=False)
        print(moves, action)

        # Advance the environment to the next frame based on the action.
        # Reward is bases on the angle of the pendulum
        real_action = add_action_to_position(action, past_action)
        next_state, reward, done, _ = env.step(real_action)
        next_state = norm_state(next_state)

        # Remember the previous state, action, reward, and done
        #agent.remember(state, action, reward, next_state, done)

        # make next_state the new current state for the next frame.
        state = next_state

        # sleep for the remaining time left, or warn when time-limit was exceeded
        total_r += reward
        # print("move {}: {}, {}, {}, {}, {}".format(moves + 1, state, action, reward, next_state, done))
        ct = time()
        past_action = real_action

        if ct < desired_end_time:
            print("sleeping for " + str(desired_end_time - ct))
            sleep(desired_end_time - ct)
        else:
            print("### warning took to long !!!! off by: {}".format(ct - desired_end_time))

        # done becomes True when the pendulum was swung up but fell down
        if done or moves + 1 == max_episode_length:
            # print the score and break out of the loop
            print("Non-random episode: score: {}, moves: {}, lr: {}"
                  .format(total_r, moves + 1, agent.epsilon))
            break


def norm_state(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def move_test():
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
    robot = RobotArmSwingUp(usb_port=port)

    while True:
        state = robot._get_current_state()
        corrected_state = robot.pendulum_pos_correction(state)
        reward = robot._reward(state)
        print(state, corrected_state, reward)
        print()
        sleep(0.05)

def debug_state_trail(robot):
    import random

    random.seed(1337)

    action1 = 90
    action2 = 90
    change = -30
    iteration_length = 0.2

    total_r = 0
    state = robot.reset()

    states = list()
    states.append(state)

    for moves in range(20):
        # decide when this iteration should be over
        start_time = time()
        desired_end_time = start_time + iteration_length

        # Decide action
        change = random.randint(-50, 50)
        action = (action1 + change, action2 + change)

        # Advance the environment to the next frame based on the action.
        # Reward is bases on the angle of the pendulum
        #real_action = add_action_to_position(action, past_action)
        next_state, reward, done, _ = robot.step(action)
        # print(next_state)
        #next_state = norm_state(next_state)
        # print(next_state)

        states.append((action, next_state))

        # Remember the previous state, action, reward, and done
        # agent.remember(state, action, reward, next_state, done)

        # make next_state the new current state for the next frame.
        state = next_state

        # sleep for the remaining time left, or warn when time-limit was exceeded
        total_r += reward
        # print("move {}: {}, {}, {}, {}, {}".format(moves + 1, state, action, reward, next_state, done))
        ct = time()

        if ct < desired_end_time:
            sleep_time = desired_end_time - ct
            if sleep_time < 0:
                print("sleeping for: " + str(sleep_time))
            sleep(sleep_time)
        else:
            print("### warning took to long !!!! off by: {}".format(ct - desired_end_time))

        # done becomes True when the pendulum was swung up but fell down
        if done or moves + 1 == 100:
            # print the score and break out of the loop
            print("score: {}, moves: {}"
                  .format( total_r, moves + 1))
            break

    print("run results:\n")
    for s in states:
        print(s)

    robot.reset()

if __name__ == '__main__':
    debug_reward()