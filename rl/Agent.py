import random
import gym
import numpy as np
import time
import os

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l1


# Deep Q-learning Agent
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        #self.long_memory = deque(maxlen=2000)

        self.gamma = 0.99    # discount rate
        self.epsilon = 1 # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(320, input_dim=self.state_size, activation='relu', kernel_regularizer=l1(1)))
        model.add(Dense(320, activation='relu', kernel_regularizer=l1(1)))
        model.add(Dense(320, activation='relu', kernel_regularizer=l1(1)))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def safe(self):
        if not os.path.isdir("backup"):
            os.makedirs("backup")
        self.model.save_weights("backup/weights_" + str(time.time()))

        for layer in self.model.layers:
            weights = layer.get_weights()
            print(weights[0].shape)
            print(layer.get_weights())

    def load(self, path):
        if not os.path.exists(path):
            raise ValueError("{} does not exist".format(path))

        self.model.load_weights(path)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #self.long_memory.append((state, action, reward, next_state, done))

    def act(self, state, use_random_chance=True):
        if use_random_chance and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state.reshape(1, self.state_size))
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size) # + random.sample(self.long_memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state.reshape(1, self.state_size)))

            target_f = self.model.predict(state.reshape(1, self.state_size))

            target_f[0][action] = target
            self.model.fit(state.reshape(1, self.state_size), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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
