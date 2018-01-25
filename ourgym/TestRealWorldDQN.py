import sys
from ourgym import RobotArm, RobotArmSwingUp
from rl import DQNAgent

if __name__ == '__main__':

    if sys.platform == 'linux' or sys.platform == 'linux2':
        port = '/dev/ttyUSB0'
    elif sys.platform == 'win32':
        port = 'COM4'
    else:
        port = "/dev/cu.usbserial-A6003X31"

    env = RobotArmSwingUp(port)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = 500
    num_steps = 200
    memory_size = 10000
    batch_size = 64
    e_start = 1
    e_finish = 0.05
    e_decay = 400
    dr = 0.99
    lr = 0.00001
    layers =  2
    nodes = 20
    frequency_updates = 0

    agent = DQNAgent(
        env,
        state_dim,
        action_dim,
        memory_size,
        e_start,
        e_finish,
        e_decay,
        dr,
        lr,
        layers,
        (nodes, nodes),
        frequency_updates,
    )


    for episode in range(num_episodes):
        state = env.reset()
        tr = 0

        for step in range(num_steps):
            action = agent.act(state)[1]

            print(step, flush=True, end=" ")
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            tr += reward
            state = next_state

            agent.replay(batch_size, update_epsilon=False)

            if done:
                break

        print("{}/{}: r={}, e={}".format(episode, num_episodes, tr, agent.get_epsilon()))
        agent._update_epsilon()

