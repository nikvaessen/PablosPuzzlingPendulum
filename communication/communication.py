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
    ser.write(str(string).encode("utf-8"))


def write_int(x, y):
    ser.write(int.to_bytes(x, length=8, byteorder=sys.byteorder))
    ser.write(32)  # space
    ser.write(int.to_bytes(y, length=8, byteorder=sys.byteorder))
    ser.write(10)  # new line


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
        for c in ser.read():
            line.append(c)
            if c == 10:
                print("Line: " + str(line))
                line = []


def get_state():
    """Here we want to get a tuple that represents the state from the micro controller"""
    return ()


if __name__ == '__main__':
    # open serial connection
    ser = serial.Serial(
        # port='/dev/ttyUSB0', # Nik
        port='/dev/cu.usbmodem1411', # Jose
        baudrate=9600,  # this needs to be set on micro-controller by doing Serial.begin(9600)
        parity=serial.PARITY_NONE,  # check parity of UC32, maybe it's even/odd
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0)  # set time-out higher if we want to wait for input

    print("connected to: " + ser.port)

    # Create two threads, 1 sending data, 1 receiving data
    sender = threading.Thread(target=send_human_data_to_controller)
    receiver = threading.Thread(target=read_data_from_controller)

    sender.start()
    receiver.start()
