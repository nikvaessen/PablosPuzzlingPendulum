import serial  # python -m pip install pyserial OR apt-get install python3-serial
import re
import threading
import random
import time
import sys


def send_human_data_to_controller():
    while True:
        x = input("Command: ")
        if re.match('^[0-9]*,[0-9]*$', x):
            numbers = x.split(",")
            x = int(numbers[0])
            y = int(numbers[1])
            print(x, y)
            write("{},{}\n".format(x, y))
        else:
            print("Input needs to be in the form 'digit,digit'")


def write(string):
    serial.write(str(string).encode("utf-8"))


def write_int(x, y):
    serial.write(int.to_bytes(x, length=8, byteorder=sys.byteorder))
    serial.write(32)  # space
    serial.write(int.to_bytes(y, length=8, byteorder=sys.byteorder))
    serial.write(10)  # new line


def send_random_data_to_controller(t=1):
    # Valid commands are numbers between 0 and 180
    while True:
        x = random.randint(0, 180)
        y = random.randint(0, 180)
        print(x, y)
        write_int(x, y)
        time.sleep(t)


def send_oscillating_data_to_controller(amp=30, center=90, f=1):
    while True:
        x = 90  # keep the bottom actuator stable
        y = center + amp
        print(x, y)
        write_int(x, y)
        amp = -amp
        time.sleep(f/2)


def read_data_from_controller():
    # Buffer for line
    line = []

    # infinite loop checking for input from micro-controller
    while True:
        for c in serial.read():
            line.append(c)
            if c == 10:
                print("Line: " + str(line))
                line = []


class Communicator:

    request_data_token = "READ\n".encode()
    write_motor_token = "WRITE\n".encode()
    failure_token = "FAILURE".encode()

    lower_joint_offset = -10
    upper_joint_offset = -15

    def __init__(self, usb_port, baudrate=9600):
        # open serial connection
        self.ser = serial.Serial(
            # port='/dev/ttyUSB0', # Nik
            # port='/dev/cu.usbmodem1411',  # Jose
            port=usb_port,
            baudrate=baudrate,  # this needs to be set on micro-controller by doing Serial.begin(9600)
            parity=serial.PARITY_NONE,  # check parity of UC32, maybe it's even/odd
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1)  # set time-out higher if we want to wait for input

        time.sleep(5)
        print("Serial connection established over port " + self.ser.port)

    def observe_state(self):
        self.ser.write(Communicator.request_data_token)

        try:
            line = self.ser.readline()
            numbers = line.strip().decode('ASCII').split(' ')
            return [int(x) for x in numbers]
        except (KeyboardInterrupt, ValueError, UnicodeDecodeError) as e:
            print('FAILED TO READ DATA, INSTEAD GOT:')
            try:
                if line:
                    print(line.strip().decode(), end='\n\n')
            except Exception as e:
                print(e)
                pass

    def send_command(self, motor1, motor2):
        self.ser.write(Communicator.write_motor_token)
        self.ser.write(str(motor1 + Communicator.lower_joint_offset).encode())
        self.ser.write(" ".encode())
        self.ser.write(str(motor2 + Communicator.upper_joint_offset).encode())


class Converter:

    def __init__(self,
                 offset_pend=50,
                 offset_bot=720,
                 offset_top=495,
                 range=[0, 180]):
        self.offsets = [offset_pend, offset_bot, offset_top]
        self.range = range

    def convert_vals(self, vals):
        return vals[0] - 620 if vals[0] > 620 else 1023 + (vals[0] - 620), vals[1], vals[2]


if __name__ == '__main__':
    # Create two threads, 1 sending data, 1 receiving data
    sender = threading.Thread(target=send_human_data_to_controller)
    receiver = threading.Thread(target=read_data_from_controller)

    sender.start()
    receiver.start()
