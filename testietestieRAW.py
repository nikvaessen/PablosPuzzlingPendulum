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

    rainsch = 3

    pendulum = []
    joint1 = []
    joint2 = []
    pendulumVelocity = []

    variances = [[] for i in range(0, rainsch)]

    with open("readings_for_%sRuns.txt" % rainsch, "w") as file:
        file.write("## This file contains RAW potentiometer readings and the Variances of the Run\n")
        file.write("## Structure: Pendulum / Joint1 / Joint2 / Pendulum_velocity\n\n")

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

        #print("current run: ", currentState)
        print("Readings run %s: %s" % (i+1, currentState))
        #print(np.var(currentState))

        with open("readings_for_%sRuns.txt" % rainsch, "a") as file:
            for obs in currentState:
                file.write(str(obs))
                file.write("\t")
            file.write("\n")

    print("Variances: ", variances[len(variances)-1])

    with open("readings_for_%sRuns.txt" % rainsch, "a") as file:
        file.write("No. runs: %s. Variances: " % rainsch)
        for obs in variances[len(variances)-1]:
            file.write(str(obs))
            file.write("\t")
        file.flush()
