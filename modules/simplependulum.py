import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import style

style.use("Solarize_Light2")

GRAV = 9.81


class Model:
    def __init__(self, cx, cy, mass=1, rope=5, initial_theta=30, initial_omega=0, friction=0.5, stepsize=1.0):
        """"""
        "Params"
        self.anchor = cx, cy
        self.mass = mass
        self.rope = rope
        self.friction = friction
        "Starting params"
        self.theta = initial_theta
        self.omega = initial_omega
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
                "ep": [],
                "ek": [],
                "total_en": [],
        }
        "Initialization"
        self.x, self.y = self.position_on_arc(initial_theta)

    def position_on_arc(self, theta_deg):
        x, y = self.anchor
        theta_rad = math.radians(theta_deg)
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

        sinn = math.sin(math.radians(theta_old))
        omega = omega_old * (1 - self.friction) - GRAV / self.rope * sinn * step_size
        theta = (theta_old + omega * step_size)  # % 360

        self.omega = omega
        self.theta = theta

        x, y = self.position_on_arc(theta)
        self.data['theta'].append(theta)
        self.data['omega'].append(omega)
        self.data['x'].append(x)
        self.data['y'].append(y)

        lin_distance = omega / 360 * self.rope
        lin_ve = lin_distance * 2 * math.pi * self.rope
        # lin_ve = omega * self.rope * step_size
        # omg_rad = math.radians(omega)
        # lin_ve = omg_rad * self.rope

        # lin_ve = omega / 180 * math.pi * self.rope * self.rope
        # lin_ve = omega * self.rope

        ep = self.mass * (self.rope + y) * GRAV
        ek = (1 / 2) * self.mass * (lin_ve ** 2)
        total_en = ep + ek
        self.data['ep'].append(ep)
        self.data['ek'].append(ek)
        self.data['total_en'].append(total_en)

        return x, y


center = (0, 0)
model = Model(*center, mass=1, rope=5, initial_theta=170, initial_omega=0, stepsize=0.1, friction=0)
initial = model.x, model.y
all_trace = []
offset_x = 300
offset_y = 300

for ti in range(750):
    array = np.zeros((600, 600, 3), dtype=np.uint8) + 170
    step = model.step()
    all_trace.append(step)
    trace = list([*zip(*all_trace)])
    poly = np.array(all_trace[-30:])
    poly[:, 1] *= -1
    poly[:, :] *= 50
    poly[:, :] += 300
    # poly = poly.reshape((-1, 1, 2))
    # poly = np.array(poly, dtype=np.int32)
    x, y = step
    x = int(x * 50) + 300
    y = int(-y * 50) + 300

    cv2.line(array, (offset_x, offset_y), (x, y), (0, 0, 0), 2)  # Line
    cv2.circle(array, (offset_x, offset_y), 10, (0, 0, 160), -1)  # Anchor
    for n, tr in enumerate(poly):
        tr = np.array(tr, dtype=np.int32)
        col = np.array([50, 200, 0]) * (n / 30)
        cv2.circle(array, tuple(tr), 8, col, -1)
    # cv2.polylines(array, [poly], False, (180, 0, 0), 5)  # Trail
    cv2.circle(array, (x, y), 10, (150, 200, 0), -1)  # Ball

    cv2.imshow("Pendulum", array)

    key = cv2.waitKey(5)
    if key == ord("q"):
        break

    # if model.theta < -30:
    #     break

trace = list([*zip(*all_trace)])

fig1 = plt.figure()
plt.title("Pendulum energy state")
plt.plot(model.data['total_en'], label="total en")
plt.plot(model.data['ek'], label="kinetic")
plt.plot(model.data['ep'], label="potential")
plt.legend(loc=1)

fig2 = plt.figure()
plt.subplot(311)
plt.plot(model.data['x'], label="x")
plt.plot(model.data['y'], label='y')
plt.legend()

plt.subplot(312)
plt.plot(model.data['theta'], label="theta")
plt.legend()

plt.subplot(313)
plt.plot(model.data['omega'], label="omega")
plt.legend()

plt.show()
