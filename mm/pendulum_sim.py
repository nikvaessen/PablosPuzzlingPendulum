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
def full_dy(y, m_p, l_p, l_1, l_2, g, b, u):
	x_accel_term = l_1 * (u[0] * cos(y[2]) - y[3] ** 2 * sin(y[2])) + l_2 * ((u[0] + u[1]) * cos(y[2] + y[4] - pi) - (y[3] + y[5]) ** 2 * sin(y[2] + y[4] - pi))
	#y_accel_term = l_1 * (-u[0] * sin(y[2]) - y[3] ** 2 * cos(y[2])) - l_2 * ((u[0] + u[1]) * sin(y[2] + y[4] - pi) - (y[3] + y[5]) ** 2 * cos(y[2] + y[4] - pi))
	y_accel_term = l_1 * (u[0] * sin(y[2]) + y[3] ** 2 * cos(y[2])) + l_2 * ((u[0] + u[1]) * sin(y[2] + y[4] - pi) + (y[3] + y[5]) ** 2 * cos(y[2] + y[4] - pi))
	return [y[1], -cos(y[0]) / l_p * x_accel_term - sin(y[0]) / l_p * y_accel_term - b / (m_p * l_p ** 2) * y[1] - g / l_p * sin(y[0]), y[3], u[0], y[5], u[1]]

# parameters (estimated from physical setup)
m_p = 0.004		# pendulum mass
l_p = 0.09		# pendulum length
l_1 = 0.12		# lower segment length
l_2 = 0.03		# upper segment length
g = 9.81		# gravity
b = 0.00001		# damping on pendulum axis

# time step
dt = 1.0 / 100.0

# set up figure and animation
fig = plt.figure()
fig.canvas.set_window_title('Inverted Pendulum Simulation')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-0.25, 0.25), ylim=(-0.1, 0.35))
plt.subplots_adjust(bottom=0.25)
ax.grid()

# setting up drawables
pendulum_line, = ax.plot([], [], 'o-')
pendulum_line.set_linewidth(4.0)
pendulum_line.set_color('#01c155')
pendulum_bob = plt.Circle((0, 0), 0.01, color='b')
ax.add_artist(pendulum_bob)

lower_joint, = ax.plot([], [], 'o-', lw=4.0, color='#01c155')
upper_joint, = ax.plot([], [], 'o-', lw=4.0, color='#01c155')

# initialising state and time-keeping
speed = 2.4 * pi
state = [0, 0, pi, speed / 2, pi, speed]
t = 0
previous = 0

paused = False
def onclick(event):
    global paused, previous
    if paused:
        previous = time.time()
        paused = False
    else:
        paused = True
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# animation update function
def animate(i):
	global m_p, l_p, l_1, l_2, g, b, state, t, previous, speed
	if not paused:
		# computing next state
		diff = time.time() - previous
		if diff > 10:
			diff = 0
		state = integrate.odeint(lambda y, t: full_dy(y, m_p, l_p, l_1, l_2, g, b, [0, 0]), state, [0, diff])[1]
		previous = time.time()

		if state[3] == 0 or (state[2] > 5/4 * pi and state[3] > 0) or (state[2] < 3/4 * pi and state[3] < 0):
			#print("state[4]:", state[2], "state[5]", state[2])
			state[3] = -state[3]
		if state[5] == 0 or (state[4] > 3/2 * pi and state[5] > 0) or (state[4] < 1/2 * pi and state[5] < 0):
			#print("state[4]:", state[4], "state[5]", state[4])
			state[5] = -state[5]
		

		# computing positions for rendering
		lower_joint_end = [l_1 * sin(state[2]), -l_1 * cos(state[2])]
		upper_joint_end = [lower_joint_end[0] + l_2 * sin(state[4] + state[2] - pi), lower_joint_end[1] - l_2 * cos(state[4] + state[2] - pi)]
		pendulum_end = [upper_joint_end[0] + l_p * sin(state[0]), upper_joint_end[1] - l_p * cos(state[0])]

		# updating graphics
		lower_joint.set_data([0, lower_joint_end[0]], [0, lower_joint_end[1]])
		upper_joint.set_data([lower_joint_end[0], upper_joint_end[0]], [lower_joint_end[1], upper_joint_end[1]])
		pendulum_line.set_data([upper_joint_end[0], pendulum_end[0]], [upper_joint_end[1], pendulum_end[1]])
		pendulum_bob.center = (pendulum_line.get_data()[0][1], pendulum_line.get_data()[1][1])

		t = t + diff
	return lower_joint, upper_joint, pendulum_line, pendulum_bob

# run animation
t0 = time.time()
animate(0)
paused = True
t1 = time.time()
interval = 1000 * dt - (t1 - t0)

previous = time.time()
ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, blit=True)
plt.show()
time.sleep(1)