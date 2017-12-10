# imports
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import time
import math

# "from" imports
from numpy import sin, cos, pi
from scipy.signal import square
from matplotlib.widgets import Slider, Button

# derivative function
def dy(y, M_p, R, b, g, u):
    return [y[1], -cos(y[0]) / R * u - b / (M_p * R * R) * y[1] - g / R * sin(y[0]), y[3], u]

# parameters
m_p = 0.01
b = 0.001
R = 0.3
L_1 = 0.5
L_2 = 0.5
g = 9.81

# time step
dt = 1.0 / 100.0

# set up figure and animation
fig = plt.figure()
fig.canvas.set_window_title('Inverted Pendulum Simulation')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
plt.subplots_adjust(bottom=0.25)
ax.grid()

ax_input = plt.axes([0.25, 0.1, 0.5, 0.03])
u_in = Slider(ax_input, 'Input', -5.0, 5.0, valinit=0.0)

# pausing animation on click
paused = False
def onclick(event):
    global paused
    if paused:
        paused = False
    else:
        paused = True
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# setting up drawables
pendulum_line, = ax.plot([], [], 'o-')
pendulum_line.set_linewidth(4.0)
pendulum_line.set_color('#01c155')
pendulum_bob = plt.Circle((0, 0), 0.03, color='b')
ax.add_artist(pendulum_bob)

# initialising state
state = [3/4 * pi, 0, 0, 0]
t = 0

previous = time.time()

def animate(i):
    global paused, m_p, R, b, g, state, t, previous
    if not paused:
        interval = time.time() - previous
        state = integrate.odeint(lambda y, t: dy(y, m_p, R, b, g, u_in.val), state, [0, interval])[1]
        previous = time.time()

        pendulum_line.set_data([0, R * sin(state[0])], [0, -R * cos(state[0])])
        pendulum_bob.center = (pendulum_line.get_data()[0][1], pendulum_line.get_data()[1][1])

        t = t + interval

    return pendulum_line, pendulum_bob

# run animation
t0 = time.time()
animate(0)
t1 = time.time()
interval = 1000 * dt - (t1 - t0)

previous = time.time()
ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, blit=True)
plt.show()
time.sleep(1)