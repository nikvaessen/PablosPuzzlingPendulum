import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../communication'))

from communication.com import Communicator
from time import sleep
#from ourgym import RobotArm
import math
import rl.QLearner as ql


port = "/dev/cu.usbserial-A6003X31" #on mac for jose, pablo and nik
if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'

def robotEnv():
    env1 = RobotArm(usb_port=port)
    json = env1.observation_space.to_jsonable(env1.action_space.sample())
    print(json)
    # number areas per space
    # (pos, vel, angle, angular_vel)
    obs1 = (12, 3, 6, 3, 6, 3)
    # create bounds for for each observation parameter
    bounds1 = list(zip(env1.observation_space.low, env1.observation_space.high))
    # check size of default bounds
    # print(bounds1)
    # if "infinite" create your own bounds
    bounds1[1] = [-0.5, 0.5]
    #bounds1[3] = [-math.radians(50), math.radians(50)]
    # print(bounds1)
    learner1 = ql.Tabular(env1, obs1, bounds1)
    learner1.run_n_episodes(2000, 10000)

def move_test():
    port = "/dev/cu.usbserial-A6003X31"
    if sys.platform == 'linux' or sys.platform == 'linux2':
        port = '/dev/ttyUSB0'
    elif sys.platform == 'win32':
        port = 'COM4'

    com = Communicator(usb_port=port, baudrate=9600)
    change = 15

    while True:
        change = -change
        sleep(0.3)
        com.send_command(90 + change, 90 + change)
        print(com.observe_state())

if __name__ == '__main__':
    #robotEnv()
    move_test()