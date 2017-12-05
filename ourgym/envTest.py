import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../communication'))

from communication.com import Communicator
from time import sleep
from ourgym import RobotArm
import math
import rl.QLearner as ql


port = "/dev/cu.usbserial-A6003X31" #on mac for jose, pablo and nik
if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'

def learn():
    env1 = RobotArm(usb_port=port)
    json = env1.observation_space.to_jsonable(env1.action_space.sample())
    print("an action: " + str(json))

    from ourgym import DiscreteAction
    a = DiscreteAction(256, 50, 130, 5)

    # number areas per space
    # (pendulum, lower motor)
    obs1 = (100, 20, 20)
    print(env1.action_space.n)

    # create bounds for for each observation parameter
    bounds1 = list(zip(env1.observation_space.low, env1.observation_space.high))


    # check size of default bounds
    print("bounds" + str(bounds1))

    learner = ql.Tabular(env1, obs1, bounds1)
    learner.run_n_episodes(100, 10000)

def move_test():
    port = "/dev/cu.usbserial-A6003X31"
    if sys.platform == 'linux' or sys.platform == 'linux2':
        port = '/dev/ttyUSB0'
    elif sys.platform == 'win32':
        port = 'COM4'

    #com = Communicator(usb_port=port, baudrate=9600)
    change = 30

    robot = RobotArm(usb_port=port)

    n = 0
    while n < 100:
        n += 1
        #change = -change
        #obot.com.send_command(50, 50)
        #sleep(5)
        print(robot.com.observe_state())
        sleep(5)

    robot.reset()

def reward_test():
    robot = RobotArm(usb_port=port)

    while True:
        s = robot._get_current_state()
        r = robot._reward(s)

        j2 = s[2]
        if j2 > 510:
            j2 = 510 - abs(510 - j2)
        else:
            j2 = 510 + abs(510 - j2)

        target = 600 - (s[1] - 520) + (j2 - 510)
        current = s[0]
        dist = joses_madness(target, current)
        max_dist = 512

        if dist > max_dist:
            r = 0
        else:
            r = 1 - (dist/max_dist)

        print("s:{}, r:{}, c:{}, t:{}, d:{}".format(s, round(r, 5), current, target, dist))
        sleep(1)


def joses_madness(t, c):
    d = 0
    if t + 512 > 1024:
        if c > t:
            d = c - t
        elif c >= t - 512:
            d = t - c
        else:
            d = 1024 - t + c
    else:
        if c < t:
            d = t - c
        elif c <= t + 512:
            d = c -t
        else:
            d = t + (1024 - c)

    return d

if __name__ == '__main__':
    learn()