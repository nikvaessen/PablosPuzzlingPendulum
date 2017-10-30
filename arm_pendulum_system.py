import numpy as np
from numpy import sin, cos
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import time
import math

class SinglePendulum:
    """Double Pendulum Class

    init_state is [theta1, omega1, theta2, omega2] in degrees,
    where theta1, omega1 is the angular position and velocity of the first
    pendulum arm, and theta2, omega2 is that of the second pendulum arm
    """
    def __init__(self,
                 init_state = [0, 0],
                 L=0.2,  # length of pendulum 1 in m
                 M=0.04,  # mass of pendulum 1 in kg
                 G=9.81,  # acceleration due to gravity, in m/s^2
                 B=0.0005,  # damping coefficient
                 origin=(0, 0)): 
        self.init_state = np.asarray(init_state, dtype='float')
        self.params = (L, M, G, B)
        self.origin = origin
        self.time_elapsed = 0

        self.state = self.init_state * np.pi / 180.

        self.acceleration = (0, 0)
    
    def position(self):
        (L, M, G, B) = self.params

        x = np.cumsum([self.origin[0],
                       L * sin(self.state[0])])
        y = np.cumsum([self.origin[1],
                       -L * cos(self.state[0])])
        return (x, y)

    def position_vector(self):
        (x, y) = self.position()
        return (x[1], y[1])

    def dstate_dt(self, state, t):
        (M, L, G, B) = self.params

        dydx = np.zeros_like(state)
        dydx[0] = state[1]
        dydx[1] = - (cos(state[0]) / L) * self.acceleration[0] - (sin(state[0]) / L) * self.acceleration[1] - (B / (M * (L ** 2))) * state[1] - (G / L) * sin(state[0])

        return dydx

    def step(self, dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt

    def offset(self, x, y):
        self.origin = (x, y)

class Motor():
    def __init__(self, init_position=90, init_target=90, origin=(0, 0), speed=500.0, length=1.0, attached_to=None, test=True):
        # target and position in degrees
        self.position = init_position
        self.target = init_target

        # speed in degrees per second
        self.speed = speed

        # graphic stuff
        self.origin = origin
        self.length = length

        # attached to motor
        self.attached_to = attached_to

        self.current_phase = 0
        self.test = test
        self.rate_of_change = 2 * np.pi * 3

    def step(self, dt):
        self.rate_of_change = self.rate_of_change * 0.9999
        self.current_phase = self.current_phase + dt * self.rate_of_change
        self.position = self.position + np.sign(self.target - self.position) * (self.speed * dt if self.speed * dt < abs(self.target - self.position) else abs(self.target - self.position))
        if self.test:
            self.position = self.position + dt * 250

    def target_reached(self):
        return self.position == self.target

    def segment(self):
        if self.attached_to:
            x = np.cumsum([self.attached_to.origin[0] + self.attached_to.length * sin(np.radians(self.attached_to.position - 90)), self.length * sin(np.radians(self.position + self.attached_to.position - 180))])
            y = np.cumsum([self.attached_to.origin[1] + self.attached_to.length * cos(np.radians(self.attached_to.position - 90)), self.length * cos(np.radians(self.position + self.attached_to.position - 180))])
        else:
            x = np.cumsum([self.origin[0], self.length * sin(np.radians(self.position - 90))])
            y = np.cumsum([self.origin[1], self.length * cos(np.radians(self.position - 90))])
        return (x, y)


# start of code
dt = 1.0 / 60.0

m1 = Motor(90, 180, (0, -0.2), 100.0, 0.2, test=False)
m2 = Motor(90, 120, speed=00.0, length=0.2, attached_to=m1, test=False)

pendulum = SinglePendulum([160.0, 0.0])

# set up figure and animation
fig = plt.figure()
fig.canvas.set_window_title('Inverted Pendulum Simulation')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
ax.grid()

paused = True

def onclick(event):
    global paused
    if paused:
        paused = False
    else:
        paused = True

cid = fig.canvas.mpl_connect('button_press_event', onclick)

line1, = ax.plot([], [], 'o-')
line2, = ax.plot([], [], 'o-')
pendulum_line, = ax.plot([], [], 'o-')

line1.set_linewidth(6.0)
line2.set_linewidth(6.0)
line1.set_color('#5990ff')
line2.set_color('#5990ff')
pendulum_line.set_linewidth(4.0)
pendulum_line.set_color('#01c155')

test_line1, = ax.plot([], [], 'o-', lw=6.0, color='black')
test_line2, = ax.plot([], [], 'o-', lw=6.0, color='black')
test_line3, = ax.plot([], [], 'o-', lw=4.0, color='black')
weight_line, = ax.plot([], [], 'o-', lw=20.0, color='black')

queue = []
queue_length = 3
    
derivative_order = 2
spacing = [i for i in range(-1, -queue_length-1, -1)]
print(spacing)
n = len(spacing)
A = np.matrix([[s ** (i - 1) for s in spacing] for i in range(1, n + 1, 1)])
b = np.matrix([[0] for i in range(0, n)])
b[derivative_order] = math.factorial(derivative_order)
c = np.matrix(la.solve(A, b)).transpose()
print(c)


def animate(i):
    global paused
    if not paused:
        # positioning and drawing motors
        if m1.target_reached():
            m1.target = 0 if m1.target > 0 else 180
        if m2.target_reached():
            m2.target = 60 if m2.target > 60 else 120

        m1.step(dt)
        m2.step(dt)

        line1.set_data(*m1.segment())
        line2.set_data(*m2.segment())
        unwrapped = np.unwrap(m2.segment())

        test_line1.set_data(line1.get_data()[0][0], line1.get_data()[1][0])
        test_line2.set_data(line2.get_data()[0][0], line2.get_data()[1][0])
        test_line3.set_data(line2.get_data()[0][1], line2.get_data()[1][1])

        # figuring out acceleration
        if len(queue) == queue_length:
            # calculate acceleration over last three steps
            
            velocity1 = ((queue[1][0] - queue[0][0]) / dt, (queue[1][1] - queue[0][1]) / dt)
            velocity2 = ((queue[2][0] - queue[1][0]) / dt, (queue[2][1] - queue[1][1]) / dt)
            acceleration = ((velocity2[0] - velocity1[0]) / dt, (velocity2[1] - velocity1[1]) / dt)
            pendulum.acceleration = acceleration
            #print("1: ", acceleration)
            #print("Vel1: ", velocity1, "Vel2: ", velocity2, "Acc: ", acceleration)
            
            #print(queue)
            #acceleration = (c * np.flip(np.matrix(queue), 0)) #* (1.0 / (dt ** derivative_order))
            #print("2: ", (acceleration.item(0, 0), acceleration.item(0, 1)))
            #pendulum.acceleration = (-acceleration.item(0, 0), -acceleration.item(0, 1))

        pendulum.step(dt)
        pendulum.offset(unwrapped[0][1], unwrapped[1][1])

        if len(queue) == queue_length:
            queue[0] = queue[1]
            queue[1] = queue[2]
            queue[2] = pendulum.origin
        elif len(queue) < queue_length:
            queue.append(pendulum.origin)

        pendulum_line.set_data(*pendulum.position())
        weight_line.set_data(pendulum_line.get_data()[0][1], pendulum_line.get_data()[1][1])

    return line1, line2, pendulum_line, test_line1, test_line2, test_line3, weight_line

# run animation
t0 = time.time()
animate(0)
t1 = time.time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, blit=True)
plt.show()
time.sleep(1)