import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import style

style.use("Solarize_Light2")

GRAV = 9.81


class Model:
    def __init__(self, cx, cy, mass=1, rope=4, initial_theta=30, friction=0.5, stepsize=1.0):
        """"""
        "Params"
        self.anchor = cx, cy
        self.mass = mass
        self.rope = rope
        self.friction = friction
        "Starting params"
        self.theta = initial_theta
        self.omega = 0
        self.time = 0
        self.time0 = 0
        "Sim params"
        self.stepsize = stepsize
        "Collect Data"
        self.data = {
                "theta": [],
                "omega": [],
                "x": [],
                "y": [],
                "tau": [],
        }
        "Initialization"
        self.x, self.y = self.position_on_arc(initial_theta)

    def position_on_arc(self, theta_deg):
        x, y = self.anchor
        theta_rad = theta_deg * math.pi / 180
        x = x + self.rope * np.sin(theta_rad)
        y = y - self.rope * np.cos(theta_rad)
        return x, y

    def step(self, step_size=None):
        if not step_size:
            step_size = self.stepsize
        self.time += step_size
        _TAU = round(self.time - self.time0, 4)

        # self.theta = self.deritive(self.theta)
        # return self.x, self.y
        theta_old = self.theta
        omega_old = self.omega

        sn = math.sin(math.radians(theta_old))
        omega = omega_old * (1 - self.friction) - GRAV / self.rope * sn * step_size
        theta = (theta_old + omega_old * step_size)  # % 360

        self.omega = omega
        self.theta = theta
        x, y = self.position_on_arc(theta)
        self.data['theta'].append(theta)
        self.data['omega'].append(omega)
        self.data['x'].append(x)
        self.data['y'].append(y)
        return x, y


center = (0, 0)
model = Model(*center, initial_theta=90, stepsize=0.05, friction=0.001)
initial = model.x, model.y
all_trace = []

for ti in range(5_000):
    step = model.step()
    all_trace.append(step)
    x, y = step
trace = list([*zip(*all_trace)])

fig = plt.figure()
plt.plot(*trace)
plt.scatter(*center, c='k', marker='.')
plt.scatter(*initial)
plt.axis('equal')

fig2 = plt.figure()
plt.plot(model.data['x'], label="x")
plt.plot(model.data['y'], label='y')
plt.legend()

fig3 = plt.figure()
plt.subplot(211)
plt.plot(model.data['theta'], label="theta")
plt.legend()
plt.subplot(212)
plt.plot(model.data['omega'], label="omega")
plt.legend()

# fig4 = plt.figure()
# plt.plot(model.data['tau'], label="tau")
# plt.legend()

plt.show()
