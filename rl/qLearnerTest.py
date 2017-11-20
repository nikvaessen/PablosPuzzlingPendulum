from rl import QLearner as ql
import gym
import math

env = gym.make('CartPole-v0')

# number areas per space
# (pos, vel, angle, angular_vel)
obs = (1, 1, 6, 3)

# create bounds for for each observation parameter
bounds = list(zip(env.observation_space.low, env.observation_space.high))

# check size of default bounds
# print(bounds)
# if "infinite" create your own bounds
bounds[1] = [-0.5, 0.5]
bounds[3] = [-math.radians(50), math.radians(50)]
# print(bounds)

learner = ql.QLearner(env, obs, bounds,
                      learning_rate=0.1,
                      maximum_discount=0.99,
                      exploration_rate=0.01)

learner.run_n_episodes(1000, 10000)
