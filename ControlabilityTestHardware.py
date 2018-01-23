import sys
from time import sleep, time
from ourgym import RobotArm
import numpy as np

from simulation import RobotArmEnvironment

if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB1'
elif sys.platform == 'win32':
    port = 'COM4'
else:
    port = "/dev/cu.usbserial-A6003X31"


def calc(pendulum, joint1, joint2, pendulum_velocity):
    current_state = (arm._get_current_state())
    return current_state[0], current_state[1], current_state[2], current_state[3]


if __name__ == '__main__':

    arm = RobotArm(usb_port=port)

    num_exp = 150
    num_commands = 10

    pendulum = []
    joint1 = []
    joint2 = []
    pendulum_velocity = []

    actions = []
    list_for_command_n = [[] for u in range(0, num_commands)]

    env = RobotArmEnvironment()

    for i in range(0, num_commands):
        print(env.action_map.get(env.action_space.sample()))
        actions.append(env.action_map.get(env.action_space.sample()))

    for i in range(num_exp):
        arm.reset()
        for j in range(0, num_commands):
            arm.com.send_command(actions[j][0], actions[j][1])
            sleep(0.2)
            pendulum, joint1, joint2, pendulum_velocity = arm._get_current_state()
            list_for_command_n[j].append(calc(pendulum, joint1, joint2, pendulum_velocity))
        sleep(1)

    with open("readings_after_%s_experiments.txt" % num_exp, "w") as file:
        for i in range(0, num_commands):
            file.write("Readings after %s commands: \n" % i)
            for obs in list_for_command_n[i]:
                file.write(str(obs))
                file.write("\n")
            extracted_values = [[] for z in range(0, 4)]

            for y in range(0, 4):
                for x in range(0, num_exp):
                    extracted_values[y].append(list_for_command_n[i][x][y])
            file.write("Variance:\t")
            for y in range(0, 4):
                file.write(str(np.var(extracted_values[y])))
                file.write("\t")
            file.write("\n")

        file.flush()
