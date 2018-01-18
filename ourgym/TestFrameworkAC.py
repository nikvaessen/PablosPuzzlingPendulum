import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import gym
import time
import json
import random
import uuid
import errno
import multiprocessing
import numpy as pi

from queue import deque


class TestVariableCreator:

    def __init__(self,
                 num_episodes_options: list,
                 num_steps_options: list,
                 state_size: int,
                 action_size: int,
                 epsilon_start_options: list,
                 epsilon_finish_options: list,
                 epsilon_decay_steps_options: list,
                 discount_rate_options: list,
                 learning_rate_options: list,
                 amount_nodes_layer_options: list):

        # running the agent
        self.num_episodes = num_episodes_options
        self.num_steps = num_steps_options

        # model size
        self.state_size = state_size
        self.action_size = action_size

        self.amount_nodes_layer = amount_nodes_layer_options

        # memory for updating

        # exploration
        self.epsilon_start = epsilon_start_options
        self.epsilon_min = epsilon_finish_options
        self.epsilon_decay_per_step = epsilon_decay_steps_options

        # parameters for updating model
        self.lr = learning_rate_options
        self.dr = discount_rate_options

    def poll(self):
        num_episodes = self.get_random_item(self.num_episodes)
        e_decay = self.get_random_item(self.epsilon_decay_per_step)

        return TestVariables(
            num_episodes,
            self.get_random_item(self.num_steps),
            self.state_size,
            self.action_size,
            self.get_random_item(self.epsilon_start),
            self.get_random_item(self.epsilon_min),
            int(e_decay * num_episodes),
            self.get_random_item(self.dr),
            self.get_random_item(self.lr),
            self.get_random_item(self.amount_nodes_layer)
        )

    @staticmethod
    def get_random_item(items):
        return items[random.randint(0, len(items) - 1)]


class TestVariables:

    def __init__(self,
                 num_episodes: int,
                 num_steps: int,
                 state_size: int,
                 action_size: int,
                 epsilon_start: float,
                 epsilon_finish: float,
                 epsilon_decay_steps: int,
                 discount_rate: float,
                 learning_rate: float,
                 amount_nodes_layer: int):
        # running the agent
        self.num_episodes = num_episodes
        self.num_steps = num_steps

        # model size
        self.state_size = state_size
        self.action_size = action_size

        self.amount_nodes_layer = amount_nodes_layer

        # exploration
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_finish
        self.epsilon_decay_per_step = epsilon_decay_steps

        # parameters for updating model
        self.lr = learning_rate
        self.dr = discount_rate

    def create_agent(self, agent_constructor):
        return agent_constructor(self.state_size,
                                 self.action_size,
                                 self.epsilon_start,
                                 self.epsilon_min,
                                 self.epsilon_decay_per_step,
                                 self.lr,
                                 self.dr,
                                 self.amount_nodes_layer)

    def run_experiment(self, env, agent_constructor):
        rh , ah = run(env, self.create_agent(agent_constructor),
                      self.num_episodes, self.num_steps)

        save_info(self.to_json_object(), env.action_map.to_json_object(), rh, ah, env.to_json_object())

    def to_json_object(self):
        obj = {}

        obj['num_episodes'] = self.num_episodes
        obj['num_steps'] = self.num_steps
        obj['state_size'] = self.state_size
        obj['action_size'] = self.action_size
        obj['epsilon_start'] = self.epsilon_start
        obj['epsilon_min'] = self.epsilon_min
        obj['epsilon_decay_episodes_required'] = self.epsilon_decay_per_step
        obj['learning_rate'] = self.lr
        obj['discount_rate'] = self.dr
        obj['amount_nodes_layer'] = self.amount_nodes_layer

        return obj


def run(env: gym.Env,
        agent,
        num_episodes: int,
        max_num_steps: int):

    reward_history_per_episode = []
    action_history_per_episode = []
    rewards = deque(maxlen=20)

    for episode_idx in range(num_episodes):
        print("{}: episode {:3}/{:3}".format(os.getpid(), episode_idx, num_episodes))
        state = env.reset()
        reward_history_per_episode.append(list())
        action_history_per_episode.append(list())

        for step_idx in range(max_num_steps):
            #env.render()
            #time.sleep(1/20)
            # take an action
            action = agent.act(state)
            # print(action, env.action_map.get(action),agent.epsilon)
            action_history_per_episode[episode_idx].append(env.action_map.get(int(action)))

            # observe effect of action and remember
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            reward_history_per_episode[episode_idx].append(float(reward))
            # store new state
            state = new_state

            agent.replay(None)

        # check if last 20 episodes have had a reward of 0
        s = sum(reward_history_per_episode[episode_idx])
        rewards.append(s)
        if len(rewards) == 20 and sum(rewards) == 0:
            break

    return reward_history_per_episode, action_history_per_episode


def save_info(parameters_json, action_map_json, reward_history_list, action_history_list, env_json):
    filename = "../experiments/{}-{}.json".format(time.time(), uuid.uuid4())

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


def run_experiments():
    from rl import ACAgent
    from simulation import RobotArmEnvironment

    env = RobotArmEnvironment()
    agent_constructor = ACAgent

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = [500, 1000, 2000, 4000]
    num_steps = [50, 100, 200]
    e_start = [1, 0.5]
    e_finish = [0.05, 0.01]
    e_decay = [0.1, 0.5, 0.9]
    dr = [0.9999, 0.999, 0.99, 0.9]
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    nodes = [10, 20, 50]

    creator = TestVariableCreator(
        num_episodes,
        num_steps,
        state_dim,
        action_dim,
        e_start,
        e_finish,
        e_decay,
        dr,
        lr,
        nodes
    )

    while True:
        try:
            creator.poll().run_experiment(env, agent_constructor)
        except KeyboardInterrupt as e:
            break


if __name__ == '__main__':
    for i in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=run_experiments)
        print("starting process {} with pid {}".format(i, os.getpid()))
        p.start()
    print("process {} quited".format(os.getpid()))