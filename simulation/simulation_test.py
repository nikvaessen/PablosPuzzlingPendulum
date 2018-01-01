import sys
sys.path.append('../')

import time
from numpy import pi
from simulation import RobotArmEnvironment

if __name__ == '__main__':
    with RobotArmEnvironment() as env:
            env.reset()
            try:
                interval = 1.0
                target = 3/4*pi
                other_target = 5/4*pi
                last_time = time.time()
                flag = False
                counter = 0
                target = [0, 0]
                while True:
                    current_time = time.time()
                    if False and current_time - last_time > interval:
                        flag = True
                        last_time = current_time
                        #target = 1.5 * pi
                        env.simulation.current_target = [target, target] if env.simulation.current_target[0] == other_target else [other_target, other_target]
                        #pass
                    env.render()
                    state, reward, done, _ = env.step(env.action_space.sample())
                    #print("vel[1] =", state[3], "vel[2] =", state[5], "control_signal =", env.simulation.control_signal)
                    counter = counter + 1
            except KeyboardInterrupt:
                pass