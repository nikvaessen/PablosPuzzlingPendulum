from gym.spaces import Discrete, Box

from ourgym import RelativeDiscreteActionMap
from ourgym.RobotArmInvPendulum import SingleMotorActionMap
from simulation.robot_arm_simulation import RobotArmEnvironment
from rl import DQNAgent, ACAgent
from time import sleep, time
import numpy as np

number_of_episodes = 10000
max_iterations_per_episode = 500

if __name__ == '__main__':

    agent = DQNAgent(6, 9, 10000, 1.0, 0.05, 9000, 0.99, 0.00001, 2, (10, 10), 1000)
    # agent.epsilon = 0.05
    # agent.load('backup/weights_1515613961.468759')

    with RobotArmEnvironment(sim_ticks_per_step=15) as env:
        # FOR ACCELERATION CONTROL
        # env.action_space = Discrete(9)
        # env.action_map = RelativeDiscreteActionMap(9, -100, 101, 100)
        # env.observation_space = Box(np.array([0, -1, 0, -1, 0, -1]), np.array([1, 1, 1, 1, 1, 1]))

        # FOR SINGLE MOTOR CONTROL
        env.action_space = Discrete(9)
        env.action_map = SingleMotorActionMap(9, 45, 135)
        env.observation_space = Box(np.array([0, -1, 0, -1, 0, -1]), np.array([1, 1, 1, 1, 1, 1]))

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
                if (episode_idx+1) % 50 == 0 or episode_idx == 0:
                    env.render()
                # sleep(1 / 2)

                ct_act = time()
                action = agent.act(state)
                total_time_acting += time() - ct_act

                ct_step = time()
                next_state, reward, done, _ = env.step(action)
                # print("action {}, state {}".format(env.action_map.get(action), next_state))
                total_time_stepping += time() - ct_step

                ct_rem = time()
                agent.remember(state, action, reward, next_state, done)
                total_time_remembering += time() - ct_rem

                state = next_state
                tr += reward

                agent.replay(32)
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

            agent._update_epsilon()
            print("episode {}/{}, average reward {}, epsilon {}, time taken {}s".format(
                episode_idx + 1, number_of_episodes, tr, agent.get_epsilon(), time() - ct))

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
