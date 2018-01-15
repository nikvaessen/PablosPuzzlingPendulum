import sys
import os

# print(sys.path)
# sys.path.append(os.path.pardir(__file__))
# print(sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import gym
import time
import json


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
                 amount_nodes_layer: tuple):
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

        save_info(self.to_json_object(), env.action_map.to_json_object(), rh, ah)

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
        obj['epsilon_decay_per_step'] = self.epsilon_decay_per_step
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


def save_info(parameters_json, action_map_json, reward_history_list, action_history_list):
    with open("{}.json".format(time.time()), 'w') as fp:
        object = {}

        object['parameters'] = parameters_json
        object['action_map'] = action_map_json
        object['rewards'] = reward_history_list
        object['actions'] = action_history_list
        json.dump(object, fp)



if __name__ == '__main__':
    from rl import DQNAgent
    from simulation import RobotArmEnvironment

    env = RobotArmEnvironment()
    agent_constructor = DQNAgent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    tv = TestVariables(510, 209, 10, state_dim, action_dim,
                       1000, 0.9, 0.1, 100, 0.99, 0.0001, 2, (1000, 2000))
    # agent = tv.create_agent(DQNAgent)
    tv.run_experiment(env, agent_constructor)

    # rh, ah = run(env, agent, 10, 200, 200)


