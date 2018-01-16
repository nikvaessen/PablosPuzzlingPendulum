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

class TestVariableCreator:

    def __init__(self,
                 num_episodes_options: list,
                 num_steps_options: list,
                 batchsize_options: list,
                 state_size: int,
                 action_size: int,
                 memory_size_options: list,
                 epsilon_start_options: list,
                 epsilon_finish_options: list,
                 epsilon_decay_steps_options: list,
                 discount_rate_options: list,
                 learning_rate_options: list,
                 amount_layers_options: list,
                 amount_nodes_layer_options: list):

        # running the agent
        self.num_episodes = num_episodes_options
        self.num_steps = num_steps_options
        self.batchsize_options = batchsize_options

        # model size
        self.state_size = state_size
        self.action_size = action_size

        self.amount_layers = amount_layers_options
        self.amount_nodes_layer = amount_nodes_layer_options

        # memory for updating
        self.memory_size = memory_size_options

        # exploration
        self.epsilon_start = epsilon_start_options
        self.epsilon_min = epsilon_finish_options
        self.epsilon_decay_per_step = epsilon_decay_steps_options

        # parameters for updating model
        self.lr = learning_rate_options
        self.dr = discount_rate_options

    def poll(self):
        amount_of_layers = self.get_random_item(self.amount_layers)
        amount_of_nodes_list = [self.get_random_item(self.amount_nodes_layer)
                                for _ in range(amount_of_layers)]

        num_episodes = self.get_random_item(self.num_episodes)
        memory_size = self.get_random_item(self.memory_size)

        e_decay = self.get_random_item(self.epsilon_decay_per_step)
        print(e_decay, num_episodes, int(e_decay*num_episodes))

        return TestVariables(
            num_episodes,
            self.get_random_item(self.num_steps),
            int(self.get_random_item(self.batchsize_options) * memory_size),
            self.state_size,
            self.action_size,
            memory_size,
            self.get_random_item(self.epsilon_start),
            self.get_random_item(self.epsilon_min),
            int(e_decay * num_episodes),
            self.get_random_item(self.dr),
            self.get_random_item(self.lr),
            amount_of_layers,
            amount_of_nodes_list
        )

    @staticmethod
    def get_random_item(items):
        return items[random.randint(0, len(items) - 1)]

class TestVariables:

    def __init__(self,
                 num_episodes: int,
                 num_steps: int,
                 batchsize: int,
                 state_size: int,
                 action_size: int,
                 memory_size: int,
                 epsilon_start: float,
                 epsilon_finish: float,
                 epsilon_decay_steps: int,
                 discount_rate: float,
                 learning_rate: float,
                 amount_layers: int,
                 amount_nodes_layer: list):
        # running the agent
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.batchsize = batchsize

        # model size
        self.state_size = state_size
        self.action_size = action_size

        if len(amount_nodes_layer) != amount_layers:
            raise ValueError("length of tuple of amount of nodes per layer "
                             "should be equal to "
                             "the amount of required layers")

        self.amount_layers = amount_layers
        self.amount_nodes_layer = amount_nodes_layer

        # memory for updating
        self.memory_size = memory_size

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
                                 self.memory_size,
                                 self.epsilon_start,
                                 self.epsilon_min,
                                 self.epsilon_decay_per_step,
                                 self.lr,
                                 self.dr,
                                 self.amount_layers,
                                 self.amount_nodes_layer)

    def run_experiment(self, env, agent_constructor):
        rh , ah = run(env, self.create_agent(agent_constructor),
                      self.num_episodes, self.num_steps, self.batchsize)

        save_info(self.to_json_object(), env.action_map.to_json_object(), rh, ah, env.to_json_object())

    def to_json_object(self):
        obj = {}

        obj['num_episodes'] = self.num_episodes
        obj['num_steps'] = self.num_steps
        obj['batchsize'] = self.batchsize
        obj['state_size'] = self.state_size
        obj['action_size'] = self.action_size
        obj['memory_size'] = self.memory_size
        obj['epsilon_start'] = self.epsilon_start
        obj['epsilon_min'] = self.epsilon_min
        obj['epsilon_decay_episodes_required'] = self.epsilon_decay_per_step
        obj['learning_rate'] = self.lr
        obj['discount_rate'] = self.dr
        obj['amount_layers'] = self.amount_layers
        obj['amount_nodes_layer'] = self.amount_nodes_layer

        return obj


def run(env: gym.Env,
        agent,
        num_episodes: int,
        max_num_steps: int,
        batchsize: int):

    reward_history_per_episode = []
    action_history_per_episode = []

    for episode_idx in range(num_episodes):
        print("episode {:3}/{:3}, reward ".format(episode_idx, num_episodes), end = "", flush=True)
        state = env.reset()
        reward_history_per_episode.append(list())
        action_history_per_episode.append(list())

        ct = time.time()
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
            reward_history_per_episode[episode_idx].append(int(reward))

            # store new state
            state = new_state

        agent.replay(batchsize)

        average_reward = 0
        count = 0
        for r in reward_history_per_episode[episode_idx]:
            average_reward += r
            count += 1

        print("{:5f}, epsilon {:5f}, ms {}".format(float(average_reward/count), agent.epsilon,
                                                   time.time() - ct))

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
        print("written json file to {}".format(fp.name))


def run_experiments():
    from rl import DQNAgent
    from simulation import RobotArmEnvironment

    env = RobotArmEnvironment()
    agent_constructor = DQNAgent

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = [500, 1000, 2000, 4000]
    num_steps = [50, 100, 200]
    memory_size = [100, 200, 400, 600, 800, 1000]
    batch_size = [0.05, 0.10, 0.25, 0.50, 1]
    e_start = [1, 0.5]
    e_finish = [0.05, 0.01]
    e_decay = [0.1, 0.5, 0.9]
    dr = [0.9999, 0.999, 0.99, 0.9]
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    layers = [1, 2, 3]
    nodes = [10, 20, 50, 100, 200]

    creator = TestVariableCreator(
        num_episodes,
        num_steps,
        batch_size,
        state_dim,
        action_dim,
        memory_size,
        e_start,
        e_finish,
        e_decay,
        dr,
        lr,
        layers,
        nodes
    )

    while True:
        creator.poll().run_experiment(env, agent_constructor)


if __name__ == '__main__':
    for i in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=run_experiments)
        print("starting process {}".format(i))
        p.start()