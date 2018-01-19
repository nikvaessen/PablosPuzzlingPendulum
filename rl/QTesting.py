import rl.QLearner2 as ql
import simulation.robot_arm_simulation as sim
from numpy import pi, sin, cos, save

init_state = (
    pi,  # theta_P
    0,  # vtheta_P
    pi,  # theta_1
    0,  # vtheta_1
    pi,  # theta_2
    0  # vtheta_2
)
env = sim.RobotArmEnvironment(init_state=init_state)
bounds = list(zip(env.observation_space.low, env.observation_space.high))
lr = [0.1, 0.01, 0.001]
obs = {
    (20, 5, 9, 1, 9, 1),
    (40, 3, 9, 1, 9, 1),
    (10, 3, 9, 1, 9, 1)
}

for rate in lr:
    for observation in obs:
        o = ql.DObservation(observation, bounds)
        learner = ql.Learner3(env, env.action_space.n, o)
        save("Qtable{}-{}".format(rate*1000, o[0]),
             learner.run_epochs(500, 500, rate, 0.999))