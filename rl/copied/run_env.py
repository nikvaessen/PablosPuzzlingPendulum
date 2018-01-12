from __future__ import print_function
from collections import deque
from simulation import RobotArmEnvironment

from AcAgent import PolicyGradientActorCritic
from ourgym import RobotArm, RobotArmSwingUp

import tensorflow as tf
import numpy as np
import sys

if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'
else:
    port = "/dev/cu.usbserial-A6003X31"


env_name = "simulation"
#env = RobotArmEnvironment()
env = RobotArmSwingUp(usb_port=port, max_step_count=200, time_step=50/1000)

sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

print(env.observation_space.shape)
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

print(state_dim, num_actions)

layer1_nodes = 10
layer2_nodes = 10


def actor_network(states):
  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, layer1_nodes],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("b1", [layer1_nodes],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
  W2 = tf.get_variable("W2", [layer2_nodes, num_actions],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b2 = tf.get_variable("b2", [num_actions],
                       initializer=tf.constant_initializer(0))
  p = tf.matmul(h1, W2) + b2
  return p

def critic_network(states):
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

pg_reinforce = PolicyGradientActorCritic(sess,
                                         optimizer,
                                         actor_network,
                                         critic_network,
                                         state_dim,
                                         num_actions,
                                         init_exp=0.9,
                                         final_exp=0.05,
                                         anneal_steps=50,
                                         summary_writer=writer)

MAX_EPISODES = 70

no_reward_since = 0

episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

  # initialize
  state = env.reset()
  total_rewards = 0

  for t in range(env.max_step_count):
    #env.render()
    action = pg_reinforce.sampleAction(state[np.newaxis,:])
    next_state, reward, done, _ = env.step(action)

    total_rewards += reward
    #print(reward, t)
    # reward = 5.0 if done else -0.1
    pg_reinforce.storeRollout(state, action, reward)

    state = next_state
    if done: break

  # if we don't see rewards in consecutive episodes
  # it's likely that the model gets stuck in bad local optima
  # we simply reset the model and try again
  if total_rewards <= -500:
    no_reward_since += 1
    if no_reward_since >= 5:
      # create and initialize variables
      print('Resetting model... start anew!')
      pg_reinforce.resetModel()
      no_reward_since = 0
      continue
  else:
    no_reward_since = 0

  pg_reinforce.updateModel()

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  print("Episode {}, with exploration {}".format(i_episode, pg_reinforce.exploration))
  print("Finished after {} timesteps".format(t+1))
  print("Reward for this episode: {}".format(total_rewards))
  print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
