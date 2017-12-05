import rl.QLearner as ql
import gym
import math

################################################################################
# env 1 CartPole-v0

env1 = gym.make('CartPole-v0')
json = env1.observation_space.to_jsonable(env1.action_space.sample())
print(json)
# number areas per space
# (pos, vel, angle, angular_vel)
obs1 = (1, 1, 6, 3)
# create bounds for for each observation parameter
bounds1 = list(zip(env1.observation_space.low, env1.observation_space.high))
# check size of default bounds
# print(bounds1)
# if "infinite" create your own bounds
bounds1[1] = [-0.5, 0.5]
bounds1[3] = [-math.radians(50), math.radians(50)]
# print(bounds1)
learner1 = ql.Tabular(env1, obs1, bounds1)
learner1.run_n_episodes(2000, 10000, offline=True)

################################################################################
# env 2 MountainCar-v0
"""
env2 = ourgym.make('MountainCar-v0')
obs2 = (18, 14)
bounds2 = list(zip(env2.observation_space.low, env2.observation_space.high))
# print(bounds2)
learner2 = ql.Tabular(env2, obs2, bounds2)
learner2.run_n_episodes(200, 500)
"""

