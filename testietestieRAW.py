import sys
from time import sleep, time
from ourgym import RobotArm
import numpy as np

if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'
else:
    port = "/dev/cu.usbserial-A6003X31"

if __name__ == '__main__':

    arm = RobotArm(usb_port=port)

    rainsch = 5

    pendulum = []
    joint1 = []
    joint2 = []
    pendulumVelocity = []

    variances = [[] for i in range(0, rainsch)]


    for i in range(rainsch):
        arm.reset()

        arm.com.send_command(110, 95)
        sleep(0.3)
        arm.com.send_command(50, 120)
        sleep(0.3)
        arm.com.send_command(120, 50)
        sleep(1)

        currentState = (arm._get_current_state())
        pendulum.append(currentState[0])
        joint1.append(currentState[1])
        joint2.append(currentState[2])
        pendulumVelocity.append(currentState[3])
        variances[i].append([np.var(pendulum), np.var(joint1), np.var(joint2), np.var(pendulumVelocity)])

        print(currentState)
        #print(np.var(currentState))
        print(variances[i])
        print("\n")

        with open("variances.txt", "w") as file:
            file.write("##This file contains the variances after each run. Last entry is significant##\n")
            file.write("##Pendulum/Joint1/Joint2/Pendulum_velocity##\n")
            for obs in variances:
                file.write(str(obs))
                file.write("\n")
            file.flush()

        with open("raw_readings.txt", "w") as file:
            file.write("##This file contains RAW potentiometer readings##\n")
            file.write("##Pendulum/Joint1/Joint2/Pendulum_velocity##\n")
            for obs in currentState:
                file.write(str(obs))
                file.write("\n")
            file.flush()


