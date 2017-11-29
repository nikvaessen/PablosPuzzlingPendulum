# miscellaneous imports
import numpy as np
import serial
import argparse
import time
from collections import deque
from deadzone import Deadzone

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
y_vals = [deque([0] * buf) for _ in range(0, nr_outputs)]

previous = time.time()
previous_vel = 0
interval = interval * 0.001

thing = 1

dz = Deadzone()

while True:
    ser.write('req\n'.encode())
    #while ser.in_waiting == 0:
    #    pass
    line = ''
    while ser.in_waiting > 0:
        try:
            line = ser.readline()
            numbers = line.strip().decode('ASCII').split(' ')
            val = [int(n) for n in numbers]

            for i, y_val in enumerate(y_vals):
                # update y data
                if len(y_val) < buf:
                    y_val.appendleft(val[i])
                else:
                    y_val.pop()
                    y_val.appendleft(val[i])
            
            dz.activate()
        except (KeyboardInterrupt, ValueError, UnicodeDecodeError) as e:
            print('FAILED TO READ DATA, INSTEAD GOT:')
            try:
                if line:
                    print(line.strip().decode(), end='\n\n')
            except Exception as e:
                pass

    print(dz.clean_val(y_vals[thing][0]))

    '''
    if y_vals[thing][0] > 600:
        result = y_vals[thing][0] - 600
    else:
        diff = 600 - y_vals[thing][0]
        result = 1023 - diff
    '''

    #print(result)
    #print(-(y_vals[0][0] - 660), y_vals[1][0] - 540, result)

    time.sleep(interval)