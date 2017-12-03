import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../communication'))

from communication.com import Communicator, Converter
import numpy as np
import time


class SwingController:
    def __init__(self, com):
        self.com = com
        self.change = 15
        self.previous_pos = None
        self.previous_vel = None
        self.converter = Converter()

    def step(self):
        state = self.com.observe_state()
        if state is None or len(state) < 3:
            return

        (pendulum_pos, motor_bot, motor_top) = self.converter.convert_vals(state)

        if self.previous_pos is None:
            self.previous_pos = pendulum_pos
            return

        if self.previous_vel is None:
            self.previous_vel = self.previous_pos - pendulum_pos
            return

        current_vel = pendulum_pos - self.previous_pos
        #sys.stdout.write('\rPrevious: {:d}, Current: {:d}, Velocity: {:d}'.format(self.previous_pos, pendulum_pos, current_vel))
        #sys.stdout.flush()
        print('Previous: {:d}, Current: {:d}, Velocity: {:d}'.format(self.previous_pos, pendulum_pos, current_vel))
        if abs(current_vel) > 1 and np.sign(current_vel) != np.sign(self.previous_vel):
            self.previous_vel = current_vel
            print('Change in direction')
            #ser.write('mov '.encode() + str(current_pos).encode() + '\n'.encode())
        self.previous_pos = pendulum_pos

        #self.com.send_command(90 + self.change, 90 + self.change)
        self.change = -self.change

        time.sleep(0.03)

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
