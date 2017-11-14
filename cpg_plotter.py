'''
This script requires the following modules to be installed:
- numpy
- matplotlib (should also install numpy)
- pyserial
'''

# miscellaneous imports
import numpy as np
import serial
import argparse
import time
from collections import deque

# matplotlib for plotting the servo positions
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
matplotlib.use("TkAgg")
style.use('fivethirtyeight')

# tkinter for UI components
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    from Tkinter import *
else:
    import tkinter as tk
    from tkinter import *

default_port = ''
if sys.platform == 'linux' or sys.platform == 'linux2':
    default_port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    default_port = 'COM4'

# creating parser and adding arguments
parser = argparse.ArgumentParser(description='Plots serial data from a microcontroller and can be used to send commands for a CPG implementation given that the correct input parsing is implemented on the controller')
parser.add_argument('-b', '--buf', dest='buf', default=100, help='Number of buffered (displayed) values. The default is 100.')
parser.add_argument('-i', '--interval', dest='interval', default=1, help='Interval at which data is sent from the microcontroller (in ms, should be known from the implementation on the controller). Only used to update the figures\' x-axis and store the interval if the data is saved to a file.')
parser.add_argument('-n', '--nr-outputs', dest='nr_outputs', default=3, help='Number of outputs to plot. The default is 3.')
parser.add_argument('-p', '--port', dest='port', default=default_port, help=('Serial port to use for communication with the microcontroller.' + (' The default is ' + default_port + '.' if default_port else '')))
parser.add_argument('-r', '--baud-rate', dest='baud_rate', default=9600, help='Baud rate for the serial connection. The default is 9600.')
parser.add_argument('-s', '--save', dest='save', default=False, nargs='?', help='Filepath to save all measured values to (only y-values). NOTE: this may take up a lot of memory when the program is running for a long time.')

# parsing arguments
args = parser.parse_args()
buf = int(args.buf)
interval = int(args.interval)
nr_outputs = int(args.nr_outputs)
port = args.port
baud_rate = int(args.baud_rate)
save = args.save

# initialising serial connection
ser = serial.Serial(port, baud_rate)

# initialising the figure and range of the axes
fig = plt.figure()
axis = plt.axes(xlim=(0, buf), ylim=(-10, 190))
axis.set_xticklabels(range(0, buf * interval + 1, int(buf * interval / 5)))
axis.tick_params(labelsize=14)

# initialising the x- and y-values of all plots
x_vals = np.linspace(0, (buf - 1), num=buf)
y_vals = [deque([0.0] * buf) for _ in range(0, nr_outputs)]

# creating a list of lines to display (one for each output)
output_lines = [axis.plot([], [], lw=4)[0] for _ in range(0, nr_outputs)]
for i, line in enumerate(output_lines):
    line.set_xdata(x_vals)
    line.set_ydata(y_vals[i])

# to be used for storing all y data, which can then be written to a file
full_y_data = []

# tkinter GUI setup
root = tk.Tk()
root.wm_title('CPG Plotter')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# quit methods for tkinter window
def quit():
    root.quit()
    root.destroy()

def quit_by_key(v):
    root.quit()

root.protocol("WM_DELETE_WINDOW", quit)
root.bind('<Control-w>', quit_by_key)

frame = tk.Frame(root)
frame.pack(side=tk.BOTTOM)

# creating the command input window
command_label = tk.Label(frame, text="Command: ")
command_label.grid(row=0, column=0)
command_label.config(font=("Arial", 15))
command_entry = tk.Entry(frame)
command_entry.grid(row=0, column=1)
command_entry.config(font=("Arial", 15))
def send_command(v):
    cmd_val = command_entry.get()
    if 'PI' in cmd_val:
        index = cmd_val.index('PI') + 3
        multiplier = float(cmd_val[index:])
        if len(cmd_val) >= index:
            new_value = np.pi * multiplier
        else:
            new_value = np.pi
        print("Value: ", new_value)
        ser.write(cmd_val[0:index-3].encode() + str(new_value).encode() + '\n'.encode())
    else:
        ser.write(cmd_val.encode() + '\n'.encode())
command_entry.bind('<Return>', send_command)

# animation method called by FuncAnimation, updated every couple milliseconds
def animate(v):
    line = ''
    while ser.in_waiting > 0:
        try:
            line = ser.readline()
            numbers = line.strip().decode('ASCII').split(' ')
            val = [float(n) for n in numbers]

            for i, y_val in enumerate(y_vals):
                # update y data
                if len(y_val) < buf:
                    y_val.appendleft(val[i])
                else:
                    y_val.pop()
                    y_val.appendleft(val[i])

                # update lines shown in figure
                output_lines[i].set_ydata(y_val)

            # if the values should be saved add them to the full_y_data list
            if save or save == None:
                full_y_data.append(val)
        except (KeyboardInterrupt, ValueError, UnicodeDecodeError) as e:
            print('FAILED TO READ DATA, INSTEAD GOT:')
            try:
                if line:
                    print(line.strip().decode(), end='\n\n')
            except Exception as e:
                pass

# initialising the function animation
ani = animation.FuncAnimation(fig, animate, interval=25)

# displaying the GUI
try:
    plt.gca().invert_xaxis()
    root.mainloop()
except:
    print('A display error occured.')

# closing serial line after program has been closed
ser.flush()
ser.close()

# save the file if the -s flag is used
if save or save == None:
    filename = ''
    if isinstance(save, str):
        filename = save
    else:
        filename = 'cpg_data__'
        filename = filename + time.strftime("%Y-%m-%d__%H-%M-%S")
    with open(filename, 'w') as file:
        if interval:
            file.write('i: ' + str(interval) + '\n')
        for y_vals in full_y_data:
            for i, y in enumerate(y_vals):
                file.write(str(y) + ('' if (i == len(y_vals) - 1) else ','))
            file.write('\n')
        print('Saved to file \"' + filename + '\".')