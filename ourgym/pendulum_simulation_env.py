from simulation.robot_arm_simulation import RobotArmEnvironment
from rl import DQNAgent
from time import sleep, time

number_of_episodes = 10000
max_iterations_per_episode = 200


if __name__ == '__main__':

    agent = DQNAgent(6, 25)

    with RobotArmEnvironment() as env:

        for episode_idx in range(number_of_episodes):
            state = env.reset()
            tr = 0

            for i in range(max_iterations_per_episode):
                env.render()
                action = agent.act(state)

                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)

                state = next_state
                tr += reward

                if done:
                    break

            agent.replay(int(max_iterations_per_episode*0.25))
            print("episode {}/{}, average reward {}, epsilon {}".format(
                episode_idx + 1, number_of_episodes, tr, agent.epsilon))

            if episode_idx % 100 == 0:
                agent.safe()

            # print("done with episode, sleeping.... zzzzz")
            # ct = time()
            # while True:
            #     env.render()
            #     sleep(1/60)
            #
            #     if time() - ct > 10:
            #         break
