import gym
import time

import numpy as np

from queue import deque

from rl import DQNAgent


class QLearner():

    def __init__(self,
                 state_dim,
                 low,
                 high,
                 delta,
                 action_dim,
                 init_e,
                 min_e,
                 e_log_decay,
                 lr,
                 dr):

        #  Create Q-table based on given state space
        if len(low) != state_dim or len(high) != state_dim or len(delta) != state_dim:
            raise ValueError("length of low, high and delta arrays do not match state dimension")

        self.state_dim = state_dim
        self.low = low
        self.high = high
        self.possible_actions = np.array([a for a in range(0, action_dim)], dtype=np.int32)

        self.Q = self._create_Q(state_dim, action_dim,
                                low, high, delta)

        # exploration
        self.epsilon = init_e
        self.min_epsilon = min_e
        self.epsilon_log_decay = e_log_decay

        # learning parameters
        self.lr = lr
        self.dr = dr

    def _create_Q(self, state_dim, action_dim, low, high, delta):
        if len(low) != state_dim or len(high) != state_dim or len(delta) != state_dim:
            raise ValueError("length of low, high and delta arrays do not match state dimension")

        max_indexes = np.zeros((len(delta) + 1), dtype=np.int32)
        max_indexes[:-1] = delta
        max_indexes[-1] = action_dim

        num_entries = np.array(max_indexes).prod()
        Q = np.zeros((num_entries, state_dim + 2))  # one for q values, one for action

        options = [list() for _ in low]
        for idx, l in enumerate(options):
            for bound in np.linspace(low[idx], high[idx], delta[idx]):
                l.append(np.round(bound, 2))

        options.append(self.possible_actions.tolist())

        indexes = [0 for _ in range(0, state_dim + 1)]
        row = 0
        while row < num_entries:
            Q[row, 0:-1] = np.array([options[i][j] for i, j in enumerate(indexes)])

            row += 1

            idx_to_update = state_dim
            while idx_to_update > -1:
                indexes[idx_to_update] += 1

                if indexes[idx_to_update] >= max_indexes[idx_to_update]:
                    indexes[idx_to_update] = 0
                    idx_to_update -= 1
                else:
                    break

        return Q

    def act(self, state: np.ndarray,
            use_internal_epsilon: bool=True,
            use_external_epsilon: bool=False,
            external_epsilon: float=0,
            return_q_value: bool=False,
            update_epsilon=True):
        self._verify_state(state)

        if (use_external_epsilon and np.random.rand() <= external_epsilon) or \
                (use_internal_epsilon and np.random.rand() <= self.epsilon):

                if use_internal_epsilon and update_epsilon:
                    self._update_epsilon()

                if return_q_value:
                    return np.random.choice(self.possible_actions), None
                else:
                    return np.random.choice(self.possible_actions)

        possible_rows = self._get_possible_row_indexes(state)
        subQ = self.Q[possible_rows, :]

        # column -1 is q values, column -2 is actions
        action = subQ[np.argmax(subQ[:, -1]), -2]
        value =  subQ[np.argmax(subQ[:, -1]), -1]

        # verify action is inside possible actions :)
        in_possible_actions = (action == self.possible_actions).any()

        if use_internal_epsilon and update_epsilon:
           self._update_epsilon()

        if in_possible_actions:
            action = int(action)
            if return_q_value:
                return action, value
            else:
                return action
        else:
            raise ValueError("unable to select a valid action")

    def _get_possible_row_indexes(self, state: np.ndarray):
        self._verify_state(state)

        # check which indexes in the table are smaller than or equal to the
        # given state
        valid = (self.Q[:, 0:self.state_dim] <= state).all(axis=1)

        # select the highest (thus last) options as the valid bound
        row_indexes = np.where(valid == True)[0][-action_dim:]

        if len(row_indexes) != action_dim:
            raise ValueError("Could not find the correct amount of rows")

        # verify the indexes all map to the same state and all possible actions
        equal_rows = (self.Q[row_indexes, 0:-2] == self.Q[row_indexes[0], 0:-2]).all()
        all_actions = (self.Q[row_indexes, -2] == self.possible_actions).all()

        if equal_rows and all_actions:
            return row_indexes
        else:
            raise ValueError("Calculated rows are not correct")

    def _get_row_index(self, state, action):
        if action in self.possible_actions:
            possible_rows = self._get_possible_row_indexes(state)
            # column -2 is the action column
            return possible_rows[np.where(action == self.Q[possible_rows, -2])[0]]
        else:
            raise ValueError("given action does not exist")

    def _verify_state(self, state: np.ndarray):
        if len(state) != self.state_dim \
                or (state < self.low).any() \
                or (state > self.high).any():
            raise ValueError("given state {} is outside of bounds".format(state))


    def update_online(self, state, action, reward, next_state):
        # first get the next action and the expected value
        next_action, next_value = self.act(next_state, use_internal_epsilon=False, return_q_value=True)

        # then update the value of Q(state, action) by doing:
        # Q(s,a) += learning_rate * (reward + (discount_rate * Q(s', a') - Q(s, a))
        row_index = self._get_row_index(state, action)
        self._update(row_index, reward, next_value)


    def update_offline(self):
        pass

    def _update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_log_decay

    def _update(self, row_index, reward, predicted_future_reward):
        current_value = self.Q[row_index, -1]
        new_value = current_value + (self.lr * (reward + (self.dr * predicted_future_reward) - current_value))
        self.Q[row_index, -1] = new_value


if __name__ == '__main__':
    num_tries = 10
    env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")

    print(env.action_space.n)
    print(env.observation_space.shape[0])
    print(env.observation_space.low)
    print(env.observation_space.high)

    # environment parameters
    num_episodes = 1000
    num_steps = 200
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # exploration parameters
    epsilon_start = 1
    epsilon_finish = 0.01
    epsilon_annealing_steps = 800
    epsilon_log_decay = 0.995

    # learning parameters
    lr = 0.01
    lr_decay = 0.01
    dr = 1
    batchsize = 64
    memory_size = 100000
    fixation_frequency = 1  # every 5 episodes

    # architecture parameters
    layers = 2
    nodes = (24, 48)
    activation = "tanh"

    # offline initialisation of memory
    num_steps_to_initialise = 200000 # 200.000

    # create the agent
    # agent = DQNAgent(env, state_dim, action_dim,
    #                  memory_size,
    #                  epsilon_start, epsilon_finish, epsilon_annealing_steps,
    #                  dr, lr,
    #                  layers, nodes,
    #                  fixation_frequency,
    #                  learning_rate_decay=lr_decay, activation=activation,
    #                  use_regularisation=False)

    # agent = QLearner(env.observation_space.shape[0],
    #                  env.observation_space.low,
    #                  env.observation_space.high,
    #                  np.array([10 for _ in env.observation_space.low]),
    #                  action_dim,
    #                  epsilon_start,
    #                  epsilon_finish,
    #                  epsilon_log_decay,
    #                  lr,
    #                  dr)

    # #  experience some initial information about the environment randonly
    # state = env.reset()
    # sum = 0
    # accomplished_first_reward = False
    # for step_idx in range(num_steps_to_initialise):
    #     action = agent.act(state, randomly=True)
    #     new_state, reward, done, _ = env.step(action)
    #     agent.remember(state, action, reward, new_state, done)
    #     state = new_state
    #     sum += reward
    #     if done:
    #         print("{:6d}/{:6d}: accomplished reward {}, updating agent".format(step_idx, num_steps_to_initialise, sum))
    #         state = env.reset()
    #         sum = 0
    #         accomplished_first_reward = True
    #         for i in range(1000):
    #             agent.replay(batchsize, update_epsilon=False, epochs=5)
    #     if (step_idx) % 1000 == 0:
    #         print("{:6d}/{:6d}".format(step_idx, num_steps_to_initialise))
    #     # if accomplished_first_reward and step_idx % 100 == 0:
    #     #     print("{:6d}/{:6d} replaying agent".format(step_idx, num_steps_to_initialise))
    #     #     agent.replay(batchsize, update_epsilon=False)

    for a_try in range(num_tries):
        print("Try {:2d}/{:2d}".format(a_try + 1, num_tries))

        agent = DQNAgent(env, state_dim, action_dim,
                         memory_size,
                         epsilon_start, epsilon_finish, epsilon_annealing_steps,
                         dr, lr,
                         layers, nodes,
                         fixation_frequency,
                         learning_rate_decay=lr_decay, activation=activation,
                         use_regularisation=False)

        #  start learning by acting based on what was learned by random acts
        mean_reward_calculator = deque(maxlen=100)
        for episode_idx in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, (1, 4))
            tr = 0
            ac = {0:0, 1: 0, 2:0}
            while True:
                if (episode_idx) % 50 == 0 and True:
                    time.sleep(1/60)
                    env.render()
                    q_values = []
                    #action = agent.act(state, q_values=q_values)
                    #print(q_values)
                    action = agent.act(state)
                else:
                    action = agent.act(state)

                new_state, reward, done, _ = env.step(action)
                new_state = np.reshape(new_state, (1, 4))
                agent.remember(state, action, reward, new_state, done)

                state = new_state
                tr += reward
                ac[action] += 1

                if done:
                    break


            mean_reward_calculator.append(tr)

            if episode_idx % 100 == 0:
                mean = np.array(mean_reward_calculator).mean()
                print("Episode {:4d}/{:4d}: mean reward over last 100 episodes was {:4f}. "
                      "Currently exploring with epsilon {}".
                      format(episode_idx, num_episodes, mean, agent.epsilon))

            agent.replay(batchsize, update_epsilon=True)
            #agent.anneal_exploration()
            # print("{:4d}/{:4d}: r={}, ah={}, e={}".format(episode_idx, num_episodes, tr, ac, agent.get_epsilon()))


