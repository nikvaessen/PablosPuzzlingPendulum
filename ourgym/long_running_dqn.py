from simulation.robot_arm_simulation import RobotArmEnvironment
from rl import DQNAgent
from time import time

import numpy as np

import multiprocessing
import os
import errno
import uuid
import time
import json

number_of_episodes = 50000
max_iterations_per_episode = 200


def save_info(current_episode,
              reward_index,
              parameters_json,
              action_map_json,
              reward_history_list,
              action_history_list,
              env_json):

    filename = "../experiments/DQN-episode_{}-reward_type_{}-{}.json".format(
        current_episode, reward_index, time.time())

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, 'w') as fp:
        object = {}

        object['parameters'] = parameters_json
        object['action_map'] = action_map_json
        object['rewards'] = reward_history_list
        object['actions'] = action_history_list
        object['environment'] = env_json
        json.dump(object, fp)
        print("{}: written json file to {}".format(os.getpid(), fp.name))


def run_experiments(reward_index):
    if reward_index < 0 or reward_index > 1:
        raise ValueError()

    num_episodes = number_of_episodes
    num_steps = max_iterations_per_episode
    batchsize = 32
    state_size = 6
    action_size = 81
    memory_size = 100000
    epsilon_start = 1
    epsilon_min = 0.1
    epsilon_decay_per_step = 10000
    lr = 0.00001
    dr = 0.99
    amount_layers = 2
    amount_nodes_layer = 40

    parameters = {}
    parameters['num_episodes'] = num_episodes
    parameters['num_steps'] = num_steps
    parameters['batchsize'] = batchsize
    parameters['state_size'] = state_size
    parameters['action_size'] = action_size
    parameters['memory_size'] = memory_size
    parameters['epsilon_start'] = epsilon_start
    parameters['epsilon_min'] = epsilon_min
    parameters['epsilon_decay_episodes_required'] = epsilon_decay_per_step
    parameters['learning_rate'] = lr
    parameters['discount_rate'] = dr
    parameters['amount_layers'] = amount_layers
    parameters['amount_nodes_layer'] = amount_nodes_layer

    agent = DQNAgent(6, 81,
                     200,
                     epsilon_start,
                     epsilon_min,
                     epsilon_decay_per_step,
                     dr,
                     lr,
                     amount_layers,
                     (amount_nodes_layer, amount_nodes_layer))

    with RobotArmEnvironment(reward_function_index=reward_index,
                             reward_function_params=(1 / 6 * np.pi, 2 * np.pi, 1, 10, 0.05, 0.1, 2, 0.001, 1)) as env:

        ah = list()
        rh = list()

        for episode_idx in range(number_of_episodes):
            state = env.reset()
            tr = 0
            ct = time.time()

            ah.append(list())
            rh.append(list())

            for i in range(max_iterations_per_episode):
                action = agent.act(state)
                ah[episode_idx].append(env.action_map.get(int(action)))

                next_state, reward, done, _ = env.step(action)
                rh[episode_idx].append(float(reward))

                agent.remember(state, action, reward, next_state, done)

                state = next_state
                tr += reward

                if done:
                    break

                agent.replay(32)

            print("episode {}/{}, average reward {}, epsilon {}, time taken {}s".format(
                episode_idx + 1, number_of_episodes, tr, agent.get_epsilon(), time.time() - ct))

            agent._update_epsilon()

            if episode_idx % 1000 == 0 and episode_idx != 0:
                agent.safe()
                save_info(episode_idx, reward_index,
                          parameters,
                          env.action_map.to_json_object(),
                          rh,
                          ah,
                          env.to_json_object())

if __name__ == '__main__':
    for i in [0, 1]:
        p = multiprocessing.Process(target=run_experiments, args=(i,))
        print("starting process {} with pid {}".format(i, os.getpid()))
        p.start()
    print("process {} quited".format(os.getpid()))
