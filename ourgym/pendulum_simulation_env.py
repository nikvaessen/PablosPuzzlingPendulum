from simulation.robot_arm_simulation import RobotArmEnvironment
from rl import DQNAgent
from time import sleep, time
import numpy as np

number_of_episodes = 500
max_iterations_per_episode = 300

if __name__ == '__main__':

    agent = DQNAgent(6, 81, 200, 1.0, 0.05, 100, 0.99, 0.01, 2, (10, 10))
    # agent.epsilon = 0.05
    # agent.load('backup/weights_1515613961.468759')

    with RobotArmEnvironment(reward_function_index=1, reward_function_params=(1/6 * np.pi, 2 * np.pi, 1, 10, 0.05, 0.1, 2, 0.001, 1)) as env:

        for episode_idx in range(number_of_episodes):
            state = env.reset()
            tr = 0

            ct = time()

            total_time_acting = 0
            total_time_stepping = 0
            total_time_remembering = 0
            total_overhead = time()

            ct_act, ct_step, ct_rem = 0, 0, 0
            for i in range(max_iterations_per_episode):
                # if (episode_idx+1) % 10 == 0 or episode_idx == 0:
                env.render()
                #sleep(1 / 60)

                ct_act = time()
                action = agent.act(state)
                total_time_acting += time() - ct_act

                ct_step = time()
                next_state, reward, done, _ = env.step(action)
                #print("action {}, state {}".format(env.action_map.get(action), next_state))
                total_time_stepping += time() - ct_step

                ct_rem = time()
                agent.remember(state, action, reward, next_state, done)
                total_time_remembering += time() - ct_rem

                state = next_state
                tr += reward

                # print("Action took {} seconds performing {} simulation steps".format(previous - current, env.simulation.step_counter))
                # previous = current
                if done:
                    break

            total_overhead = time() - total_overhead
            step_per = total_time_stepping / total_overhead
            act_per = total_time_acting / total_overhead
            rem_per = total_time_remembering / total_overhead
            over_per = 1 - step_per - rem_per - act_per

            # print("act:{}, step:{}, remember:{}, overhead:{}".
            #       format(act_per, step_per, rem_per, over_per))

            agent.replay(int(10))
            print("episode {}/{}, average reward {}, epsilon {}, time taken {}s".format(
                episode_idx + 1, number_of_episodes, tr, agent.epsilon, time() - ct))

            if episode_idx % 100 == 0:
                agent.safe()

            # if (episode_idx + 1) % 10 == 0:
            #    agent.plot_weights()

            # print("done with episode, sleeping.... zzzzz")
            # ct = time()
            # while True:
            #     env.render()
            #     sleep(1/60)
            #
            #     if time() - ct > 10:
            #         break
