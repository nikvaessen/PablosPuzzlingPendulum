import numpy as np
from simulation import RobotArmEnvironment
from numpy import pi

if __name__ == "__main__":
    num_repetitions = 100

    with RobotArmEnvironment() as env:
        # initialisation
        env.reset()

        variances = [[0.0 for _ in range(0, env.action_space.n)] for _ in range(0, env.action_space.n)]
        max_variance = -1

        # go through all 81 possible start states
        for start_state_idx in range(0, env.action_space.n):
            # reset environment and set to correct start state
            env.reset()
            start_state = env.action_map.get(start_state_idx)
            end_states = [[] for _ in range(0, env.action_space.n)]

            print("Starting state index {}, position {}".format(start_state_idx, start_state))

            # repeat for the specified number of iterations and calculate variance
            for idx in range(0, num_repetitions):
                # reset states
                env.simulation.state = [0, 0, (start_state[0] * (pi / 180) + pi / 2), 0,
                                        (start_state[1] * (pi / 180) + pi / 2), 0]
                print(("\r" * 14) + "Repetition {:3d}".format(idx), end="")

                # go to all other 80 states
                for action_idx in range(0, env.action_space.n):
                    if action_idx is not start_state_idx:
                        action = env.action_map.get(action_idx)
                        new_state, reward, done, _ = env.step(action_idx)
                        end_states[action_idx].append(new_state)

                        # print(("\r" * 35) + "Action index {:3d}, action ({:3d}, {:3d})"
                        #       .format(action_idx, action[0], action[1]), end="")

            for action_idx, action_state_results in enumerate(end_states):
                variances[start_state_idx][action_idx] = np.var(action_state_results, 0)
                local_max = np.max(variances[start_state_idx][action_idx])
                if local_max > max_variance:
                    max_variance = local_max

            print("\n")

        print("Max variance: {}".format(max_variance))

