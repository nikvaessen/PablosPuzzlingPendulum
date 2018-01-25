import random
import gym
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf

# from .advanced.AcAgent import PolicyGradientActorCritic

from collections import deque

from gym.spaces import Discrete
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l1

from ourgym import AbsoluteDiscreteActionMap
from simulation import RobotArmEnvironment


class Agent:

    def __init__(self,
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

        # model size
        self.state_size = state_size
        self.action_size = action_size

        if len(amount_nodes_layer) != amount_layers:
            raise ValueError("length of tuple of amount of nodes per layer "
                             "should be equal to "
                             "the amount of required layers")

        self.num_layers = amount_layers
        self.num_nodes = amount_nodes_layer

        # memory for updating
        self.memory = deque(maxlen=memory_size)

        # exploration
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_finish
        self.epsilon_decay_per_step = (epsilon_start - epsilon_finish) / epsilon_decay_steps

        # parameters for updating model
        self.lr = learning_rate
        self.dr = discount_rate

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_per_step

    def remember(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, use_random_chance=True):
        raise NotImplementedError()

    def replay(self, batch_size):
        raise NotImplementedError()


# Deep Q-learning Agent
class DQNAgent(Agent):

    def __init__(self,
                 env,
                 state_size: int,
                 action_size: int,
                 memory_size: int,
                 epsilon_start: float,
                 epsilon_finish: float,
                 epsilon_decay_steps: int,
                 discount_rate: float,
                 learning_rate: float,
                 amount_layers: int,
                 amount_nodes_layer: tuple,
                 fixate_model_frequency: int,
                 learning_rate_decay : float=0,
                 activation: str ='tanh',
                 use_regularisation: bool = False,
                 regularisation_factor: int=1
                 ):

        super().__init__(state_size,
                         action_size,
                         memory_size,
                         epsilon_start,
                         epsilon_finish,
                         epsilon_decay_steps,
                         discount_rate,
                         learning_rate,
                         amount_layers,
                         amount_nodes_layer,
                         )

        self.env = env

        self.lr_decay = learning_rate_decay
        self.activation = activation
        self.use_regularisation = use_regularisation
        self.reg_factor = regularisation_factor

        self.model = self._build_model()
        self.fixed_model = self._build_model()
        self.acts = 0
        self.fix_frequency = fixate_model_frequency
        self.should_fixate = fixate_model_frequency > 1

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.num_nodes[0], input_dim=self.state_size,
                        activation=self.activation,
                        kernel_regularizer=l1(self.reg_factor) if self.use_regularisation else None))

        for i in range(1, self.num_layers):
            model.add(Dense(self.num_nodes[i], activation=self.activation,
                            kernel_regularizer=l1(self.reg_factor) if self.use_regularisation else None))

        model.add(Dense(self.action_size, activation='linear'))

        # model.add(Dense(24, input_dim=4, activation='tanh'))
        # model.add(Dense(48, activation='tanh'))
        # model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr, decay=self.lr_decay))

        return model

    def fix_weights(self):
        self.fixed_model.set_weights(self.model.get_weights())

    def save(self, path=None):
        if not path:
            if not os.path.isdir("backup"):
                os.makedirs("backup")
            self.model.save_weights("backup/weights_" + str(time.time()))
        else:
            self.model.save_weights(path)

    def plot_weights(self):
        f, axarr = plt.subplots(len(self.model.layers))
        for l, layer in enumerate(self.model.layers):
            weights = layer.get_weights()
            temp = axarr[l].imshow(weights[0], cmap=plt.cm.Blues)
            # plt.colorbar(temp)
        plt.show()

    def load(self, path):
        if not os.path.exists(path):
            raise ValueError("{} does not exist".format(path))

        self.model.load_weights(path)

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state).reshape([1, self.state_size])
        next_state = np.array(next_state).reshape([1, self.state_size])
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_random_chance=True, randomly=False, q_values=[]):
        # self.acts += 0
        # if self.should_fixate and self.acts % self.fix_frequency == 0:
        #     print("fixating")
        #     self.fix_weights()
        #
        # if randomly or (use_random_chance and np.random.rand() <= self.epsilon):
        #     action = random.randrange(self.action_size)
        #     # print("Act randomly: {}".format(action))
        #     return action
        #
        # act_values = self.model.predict(state.reshape(1, self.state_size))
        #
        # for q in act_values[0]:
        #     q_values.append(q)
        #
        # action = np.argmax(act_values) # returns action
        # # print("Act non-randomly: {}".format(action))
        # return action

        if np.random.random() <= self.epsilon:
            return np.nan, self.env.action_space.sample(), None
        else:
            prediction = self.model.predict(np.array(state).reshape([1, self.state_size]))
            return np.max(prediction), np.argmax(prediction), prediction

    def replay(self, batch_size, update_epsilon=True, epochs=1):
        # if len(self.memory) < batch_size:
        #     return
        #
        # mini_batch = random.sample(self.memory, batch_size)
        #
        # states = np.zeros((batch_size, self.state_size))
        # next_states = np.zeros((batch_size, self.state_size))
        # Y = np.zeros((batch_size, self.action_size))
        #
        # # Create X and Y matrices for update
        # for idx, (state, action, reward, next_state, done) in enumerate(mini_batch):
        #     target = reward
        #     states[idx, :] = state.reshape(1, self.state_size)
        #     next_states[idx, :] = next_state.reshape(1, self.state_size)
        #
        # # calculate the expected reward
        # P = self.fixed_model.predict(next_states) if self.should_fixate else \
        #     self.model.predict(next_states)
        #
        # for idx, (state, action, reward, next_state, done) in enumerate(mini_batch):
        #     target = P[idx]
        #     if done:
        #         target[action] = reward
        #     else:
        #         target[action] = reward + self.dr * np.amax(P[idx])
        #
        #     Y[idx] = target
        #
        # self.model.fit(states, Y, epochs=epochs, verbose=0)
        #
        # if update_epsilon:
        #     self._update_epsilon()

        x_batch, y_batch = [], []
        mini_batch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in mini_batch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.dr * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        if update_epsilon:
            self._update_epsilon()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon = self.epsilon * 0.995

    def anneal_exploration(self):
        self._update_epsilon()

    def get_epsilon(self):
        return self.epsilon


class ACAgent(Agent):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 epsilon_start: float,
                 epsilon_finish: float,
                 epsilon_decay_steps: int,
                 discount_rate: float,
                 learning_rate: float,
                 amount_nodes_layer: int):

        self.session = tf.Session()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.state_dim = state_size
        self.action_size = action_size
        self.num_actions = action_size

        self.amount_nodes_layer = amount_nodes_layer

        self.model = PolicyGradientActorCritic(self.session,
                                               self.optimizer,
                                               self._get_actor_network_function(),
                                               self._get_critic_network_function(),
                                               state_size,
                                               action_size,
                                               epsilon_start, epsilon_finish,
                                               epsilon_decay_steps,
                                               discount_rate,
                                               )

    def reset(self):
        tf.reset_default_graph()

    def _get_actor_network_function(self):
        state_dim = self.state_dim
        action_size = self.action_size
        amount_nodes = self.amount_nodes_layer

        def actor(states):
            # define policy neural network
            W1 = tf.get_variable("W1", [state_dim, amount_nodes],
                                 initializer=tf.random_normal_initializer())
            b1 = tf.get_variable("b1", [amount_nodes],
                                 initializer=tf.constant_initializer(0))
            h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)

            W2 = tf.get_variable("W2", [amount_nodes, action_size],
                                 initializer=tf.random_normal_initializer(stddev=0.1))
            b2 = tf.get_variable("b2", [action_size],
                                 initializer=tf.constant_initializer(0))
            p = tf.matmul(h1, W2) + b2
            return p

        return actor

    def _get_critic_network_function(self):
        state_dim = self.state_dim

        def critic(states):
            # define policy neural network
            W1 = tf.get_variable("W1", [state_dim, 20],
                                 initializer=tf.random_normal_initializer())
            b1 = tf.get_variable("b1", [20],
                                 initializer=tf.constant_initializer(0))
            h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
            W2 = tf.get_variable("W2", [20, 1],
                                 initializer=tf.random_normal_initializer())
            b2 = tf.get_variable("b2", [1],
                                 initializer=tf.constant_initializer(0))
            v = tf.matmul(h1, W2) + b2
            return v

        return critic

    def get_epsilon(self):
        return self.model.exploration

    def remember(self, state, action, reward, next_state, done):
        self.model.storeRollout(state, action, reward)

    def act(self, state, use_random_chance=True):
        return self.model.sampleAction(state[np.newaxis, :])

    def replay(self, batch_size):
        self.model.updateModel()
        pass

    def safe(self):
        saver = tf.train.Saver()
        saver.save(self.session, "tf/backup/{}.ckpt".format(time.time()))
        pass


def test():
    state_size = 4
    action_size = 40

    state_low = np.array([-1, -1, -1, -1])
    #state_high = np.array([5, 10 , 1, 70])
    state_high = np.array([1, 1 , 1, 1])


    action_low = 0
    action_high = action_size - 1

    agent = DQNAgent(state_size, action_size)

    c1 = count_predict_distribution(agent, state_size, state_low, state_high, action_low, action_high)
    print(c1)

    train_bandit_problem(agent, state_size, state_low, state_high, action_size)

    c2 = count_predict_distribution(agent, state_size, state_low, state_high, action_low, action_high)
    print(c2)


def train_randomly(agent: DQNAgent, state_size, state_low, state_high):
    for i in range(0, 1000):
        print(i)

        for j in range(0, 100):
            state = state_low + ((state_high - state_low) * np.random.rand(state_size))
            new_state = state_low + ((state_high - state_low) * np.random.rand(state_size))
            action = random.randrange()

            reward = np.random.random_sample()
            done = True if np.random.random_sample() > 0.99 else False

            agent.remember(state, action, reward, new_state, done)

            if done:
                break

        agent.replay(100)


def train_bandit_problem(agent: DQNAgent, state_size, state_low, state_high, action_size=40):
    rewards = []
    r = -1
    for _ in range(action_size):
        rewards.append(r)
        r += -0.2


    print(rewards)

    for i in range(0, 1000):
        print(i, agent.epsilon)

        for j in range(0, 100):
            state = state_low + ((state_high - state_low) * np.random.rand(state_size))
            new_state = state_low + ((state_high - state_low) * np.random.rand(state_size))
            action = agent.act(state)

            reward = rewards[action]
            done = True if action == 1 else False

            agent.remember(state, action, reward, new_state, done)

            if done:
                break

        agent.replay(50)


def count_predict_distribution(agent, state_size, state_low, state_high, action_low, action_high):
    # Test if a randomly initialised network returns every possible action
    action_count_map = {n: 0 for n in range(action_low, action_high + 1)}

    for i in range(0, 10000):
        state = state_low + ((state_high - state_low) * np.random.rand(state_size))
        action = agent.act(state)
        action_count_map[action] += 1

    return action_count_map


def cartpole_test():
    env = gym.make("CartPole-v0")
    agent = DQNAgent(env,
                     env.observation_space.shape[0],
                     env.action_space.n,
                     10000,
                     1.0,
                     0.05,
                     600,
                     0.995,
                     0.01,
                     2,
                     (20, 20),
                     0,
                     )

    for e in range(1000):
        state = env.reset()
        done = False
        tr = 0
        i = 0
        while not done and i < 200:
            if e % 100 == 0 or False:
                env.render()
                # time.sleep(1)

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            tr += reward
            i += 1

        agent.replay(64)
        # if e % 100 == 0:
        print("Episode {}, reward = {}, epsilon = {}".format(e, tr, agent.epsilon))


def run_without_training_balancing(path):
    env = RobotArmEnvironment(reward_function_index=1,
                              done_function_index=1,
                              simulation_init_state=(np.pi, 0, np.pi, 0, np.pi, 0),
                              reset_with_noise=True)
    env.action_space = Discrete(25)
    env.action_map = AbsoluteDiscreteActionMap(70, 110, 5)

    agent = DQNAgent(env,
                     env.observation_space.shape[0],
                     env.action_space.n,
                     10000,
                     1.0,
                     0.05,
                     600,
                     0.995,
                     0.01,
                     2,
                     (20, 20),
                     0)


    agent.load(path)
    agent.epsilon = 0.0

    for e in range(10000):
        print("Starting episode {}.".format(e))
        state = env.reset()
        done = False
        i = 0

        while i <= 200:
            env.render()
            time.sleep(1 / 30)

            # action = np.argmax(agent.model.predict(np.array(state).reshape([1, agent.state_size])))
            _, action, _ = agent.act(state)
            print(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            i += 1

        print("Ending episode {} after {} steps.\n".format(e, i))


if __name__ == '__main__':
    run_without_training_balancing("/home/simon/University/Project/SimonsSignificantStatistics/normal-index-2/22-01-2018_23-49-16_808fe23e-786a-437c-a765-6b66200341a6/weights-ep-1500")
