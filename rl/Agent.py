import random
import gym
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf

from .advanced.AcAgent import PolicyGradientActorCritic

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l1



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
                 fixate_model_frequency: int):

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

        self.model = self._build_model()
        self.fixed_model = self._build_model()
        self.acts = 0
        self.fix_frequency = fixate_model_frequency
        self.should_fixate = fixate_model_frequency > 1

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.num_nodes[0], input_dim=self.state_size,
                        activation='relu', kernel_regularizer=l1(1)))

        for i in range(1, self.num_layers):
            model.add(Dense(self.num_nodes[i], activation='relu',
                            kernel_regularizer=l1(1)))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model

    def fix_weights(self):
        self.fixed_model.set_weights(self.model.get_weights())

    def safe(self):
        if not os.path.isdir("backup"):
            os.makedirs("backup")
        self.model.save_weights("backup/weights_" + str(time.time()))

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
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_random_chance=True):
        self.acts += 0
        if self.should_fixate and self.acts % self.fix_frequency == 0:
            self.fix_weights()

        if use_random_chance and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            # print("Act randomly: {}".format(action))
            return action

        act_values = self.model.predict(state.reshape(1, self.state_size))
        action = np.argmax(act_values[0]) # returns action
        # print("Act non-randomly: {}".format(action))
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))

        # Create X and Y matrices for update
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            states[idx, :] = state.reshape(1, 6)
            next_states[idx, :] = next_state.reshape(1, 6)

        # calculate the expected reward
        P = self.should_fixate and self.fixed_model.predict(next_states) or \
            self.model.predict(next_states)

        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = P[idx]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.dr * np.amax(P[idx])

            Y[idx] = target

        self.model.fit(states, Y, epochs=1, verbose=0)

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


if __name__ == '__main__':
    test()
