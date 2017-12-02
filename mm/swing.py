import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../communication'))

from communication.com import Communicator
import time
import sys


class SwingController:
    def __init__(self, com):
        self.com = com

    def step(self):
        state = self.com.observe_state()
        if state is None:
            return

        (pendulum, motor_bot, motor_top) = state
        sys.stdout.write('\rPot: {:d}, Bot: {:d}, Top: {:d}'.format(pendulum, motor_bot, motor_top))
        sys.stdout.flush()
        #time.sleep(0.005)

    def controllable_location(self):
        return False

    def controllable_speed(self):
        pass


if __name__ == '__main__':
    # determining the default port to use for serial communication
    port = '/dev/cu.usbserial-A6003X31'
    if sys.platform == 'linux' or sys.platform == 'linux2':
        port = '/dev/ttyUSB0'
    elif sys.platform == 'win32':
        port = 'COM4'

    com = Communicator(port)

    swing = SwingController(com)
    while not swing.controllable_location():
        swing.step()
