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

    rainsch = 50

    pendulumVelocity, pendulum, joint1, joint2 = []

    variances = [][]

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

        variances[i].append(np.var(pendulum))
        variances[i].append(np.var(joint1))
        variances[i].append(np.var(joint2))
        variances[i].append(np.var(pendulumVelocity))

        print(variances)

