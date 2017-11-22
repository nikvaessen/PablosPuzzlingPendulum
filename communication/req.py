# miscellaneous imports
import numpy as np
import serial
import argparse
import time
from collections import deque

# parsing arguments
buf = 3
interval = 20
nr_outputs = 3
port = '/dev/ttyUSB0'
baud_rate = 9600

# initialising serial connection
ser = serial.Serial(port, baud_rate)

# stuff
x_vals = np.linspace(0, (buf - 1), num=buf)
y_vals = [deque([0.0] * buf) for _ in range(0, nr_outputs)]

previous = time.time()
previous_vel = 0
interval = 0.1

while True:
    ser.write('req\n'.encode())
    #while ser.in_waiting == 0:
    #    pass
    line = ''
    while ser.in_waiting > 0:
        try:
            line = ser.readline()
            numbers = line.strip().decode('ASCII').split(' ')
            val = [float(n) for n in numbers]

            for v in val:
                print(v, end=' ')
            print('')
            
            for i, y_val in enumerate(y_vals):
                # update y data
                if len(y_val) < buf:
                    y_val.appendleft(val[i])
                else:
                    y_val.pop()
                    y_val.appendleft(val[i])
            

        except (KeyboardInterrupt, ValueError, UnicodeDecodeError) as e:
            print('FAILED TO READ DATA, INSTEAD GOT:')
            try:
                if line:
                    print(line.strip().decode(), end='\n\n')
            except Exception as e:
                pass

    current = time.time()
    if current - previous > interval:
        previous = current
        vel = y_vals[2][0] - y_vals[2][1]
        if vel != 0 and vel != 1 and np.sign(vel) != np.sign(previous_vel):
            previous_vel = vel
            print('Change in direction')

    time.sleep(interval)