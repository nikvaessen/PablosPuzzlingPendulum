import random
import gym
import numpy as np
import time
import os

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from ourgym import DiscreteAction


# Deep Q-learning Agent
class DQNAgent:

    def __init__(self, state_size, action_size, action_map: DiscreteAction):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.action_map = action_map

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.975
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def safe(self):
        if not os.path.isdir("backup"):
            os.makedirs("backup")
        self.model.save_weights("backup/weights_" + str(time.time()))

    def load(self, path):
        if not os.path.exists(path):
            raise ValueError("{} does not exist".format(path))

        self.model.load_weights(path)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_map.get(random.randrange(self.action_size))

        act_values = self.model.predict(state.reshape(1, 3))
        return self.action_map.get(np.argmax(act_values[0]))  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state.reshape(1, 3)))
            target_f = self.model.predict(state.reshape(1, 3))

            target_f[0][self.action_map.getIndex(action)] = target
            self.model.fit(state.reshape(1, 3), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


