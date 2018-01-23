import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from gym.spaces import Discrete

from ourgym import AbsoluteDiscreteActionMap
from rl import DQNAgent
from simulation import RobotArmEnvironment

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import gym
import time
import json
import random
import uuid
import errno
import multiprocessing
import numpy as np

from datetime import datetime


def run(env: RobotArmEnvironment,
        agent: DQNAgent,
        num_episodes: int,
        max_num_steps: int,
        batch_size: int,
        directory_path: str):

    reward_history_file_name = directory_path + "reward.csv"
    action_history_file_name = directory_path + "action.csv"
    max_q_history_file_name = directory_path + "max-q.csv"
    state_history_file_name = directory_path + "state.csv"

    # Parse these files with:
    # with open(file_name, "r") as f:
    #     reader = csv.reader(f, delimiter=" ")
    #     for row in reader:
    #         for col in row:
    #             col = ast.literal_eval(col) # (nan values have to be checked for)

    previous = time.time()
    for episode_idx in range(num_episodes):
        state = env.reset()

        for step_idx in range(max_num_steps):
            env.render()
            time.sleep(1 / 60)

            with open(state_history_file_name, "a") as f:
                f.write(("("+("{}," * 6)+") ").format(*env.simulation.state))

            # take an action
            max_q, action, prediction = agent.act(state)

            with open(max_q_history_file_name, "a") as f:
                f.write("{} ".format(max_q))
            with open(action_history_file_name, "a") as f:
                f.write("({},{}) ".format(env.action_map.get(int(action))[0], env.action_map.get(int(action))[1]))

            # observe effect of action and remember
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            with open(reward_history_file_name, "a") as f:
                f.write("{} ".format(float(reward)))

            # store new state
            state = new_state

            if done:
                break

        agent.replay(batch_size)

        # new line in all data files
        with open(action_history_file_name, "a") as f:
            f.write("\n")
        with open(reward_history_file_name, "a") as f:
            f.write("\n")
        with open(max_q_history_file_name, "a") as f:
            f.write("\n")
        with open(state_history_file_name, "a") as f:
            f.write(("(" + ("{}," * 6) + ") \n").format(*env.simulation.state))

        if episode_idx % 50 == 0:
            agent.save(directory_path + "weights-ep-{}".format(episode_idx))

        current = time.time()
        print("{}: episode {:3}/{:3} completed in {:4}s".format(os.getpid(), episode_idx, num_episodes, current - previous))
        previous = current
        # agent._update_epsilon()
        # check if last 20 episodes have had a reward of 0
        # s = sum(reward_history_per_episode[episode_idx])
        # rewards.append(s)
        # if len(rewards) == 20 and sum(rewards) == 0:
        #     break


def run_experiments():
    # changes reward and done function
    task_index = 2

    # common parameters
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

    while True:
        # create directory if it does not exist
        directory_path = "../experiments_{}/{}_{}/".format(task_index, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), uuid.uuid4())
        if not os.path.exists(os.path.dirname(directory_path)):
            try:
                os.makedirs(os.path.dirname(directory_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        try:
            nr_actions_per_motor = 9
            lower_bound = 45
            upper_bound = 135
            simulation_init_state = (0, 0, np.pi, 0, np.pi, 0)
            reset_with_noise = False
            if task_index == 1:
                nr_actions_per_motor = 9
                lower_bound = 70
                upper_bound = 110
                simulation_init_state = (np.pi, 0, np.pi, 0, np.pi, 0)
                reset_with_noise = True
            elif task_index == 2:
                nr_actions_per_motor = 5
                lower_bound = 45
                upper_bound = 135
                simulation_init_state = (0, 0, np.pi, 0, np.pi, 0)
                reset_with_noise = False

            env = RobotArmEnvironment(reward_function_index=task_index, done_function_index=task_index,
                                      simulation_init_state=simulation_init_state, reset_with_noise=reset_with_noise)
            env.action_space = Discrete(nr_actions_per_motor ** 2)
            env.action_map = AbsoluteDiscreteActionMap(lower_bound, upper_bound, nr_actions_per_motor)

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

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

            run(env, agent, num_episodes, num_steps, batch_size, directory_path)
        except KeyboardInterrupt as e:
            # for f in os.listdir(os.path.dirname(directory_path)):
            #     if re.search(file_path, f):
            #         os.remove(os.path.join(directory_path, f))
            break


if __name__ == '__main__':
    # for i in range(multiprocessing.cpu_count()):
    #     p = multiprocessing.Process(target=run_experiments)
    #     print("Starting process {} with PID {}.".format(i, os.getpid()))
    #     p.start()
    # print("Process {} quit.".format(os.getpid()))
    run_experiments()