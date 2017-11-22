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
y_vals = [deque([0] * buf) for _ in range(0, nr_outputs)]

previous = time.time()
previous_vel = 0
interval = interval * 0.001
critical_val = 10
flag = False
flag2 = False

lower_range = range(0, critical_val)
upper_range = range(1023 - critical_val + 1, critical_val + 1)
wrong_range = range(critical_val * 3, 1023 - critical_val * 3)

thing = 2

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
            
            flag2 = True
        except (KeyboardInterrupt, ValueError, UnicodeDecodeError) as e:
            print('FAILED TO READ DATA, INSTEAD GOT:')
            try:
                if line:
                    print(line.strip().decode(), end='\n\n')
            except Exception as e:
                pass

    if flag2:
        # need to check whether dead zone was entered
        if not flag and y_vals[thing][0] < critical_val or abs(y_vals[thing][0] - 1023) < critical_val:
            #print('entered critical range')
            flag = True

        if flag and not y_vals[thing][0] in wrong_range and y_vals[thing][0] > critical_val and abs(y_vals[thing][0] - 1023) > critical_val:
            #print('exited critical range')
            flag = False

        if flag and y_vals[thing][0] in wrong_range:
            y_vals[thing][0] = 0

        if y_vals[thing][0] > 600:
            result = y_vals[thing][0] - 600
        else:
            diff = 600 - y_vals[thing][0]
            result = 1023 - diff

        #print(result)
        print(-(y_vals[0][0] - 660), y_vals[1][0] - 540, result)

    time.sleep(interval)