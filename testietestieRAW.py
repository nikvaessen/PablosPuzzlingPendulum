import sys
from time import sleep, time
from ourgym import RobotArm

if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'
else:
    port = "/dev/cu.usbserial-A6003X31"

if __name__ == '__main__':

    arm = RobotArm(usb_port="/dev/cu.usbserial-A6003X31")

    all_states = []

    for i in range(50):
        arm.reset()
        #arm.com.send_command(80, 110) # always sleep a bit after sending a command
        #sleep(0.1)

        arm.com.send_command(110, 95)
        #arm.step(52)
        sleep(0.3)
        arm.com.send_command(50, 120)
        sleep(0.3)
        arm.com.send_command(120, 50)
        sleep(1)
        print(arm._get_current_state())

        sleep(3)


