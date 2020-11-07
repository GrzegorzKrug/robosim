import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import style

style.use("Solarize_Light2")

GRAV = 9.81


class Model:
    def __init__(self, cx, cy, mass=1, rope=5, initial_theta=1.0, initial_omega=0, friction=0.5, stepsize=1.0):
        """"""
        "Params"
        self.anchor = cx, cy
        self.mass = mass
        self.rope = rope
        self.friction = friction

        "Starting params"
        self.theta = initial_theta
        self.initial_theta = initial_theta
        self.omega = initial_omega
        self.time = 0
        self.time0 = 0
        "Sim params"

        self.stepsize = stepsize
        "Collect Data"
        self.data = {
                "theta": [],
                "theta_deg": [],
                "omega": [],
                "veloc": [],
                "lin_ve": [],
                "radial": [],
                "radial_deg": [],
                "x": [],
                "y": [],
                "ep": [],
                "ek": [],
                "total_en": [],
                "factor": [],
        }
        "Initialization"
        self.x, self.y = self.position_on_arc(math.degrees(initial_theta))

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

        theta_old = self.theta
        omega_old = self.omega

        omega = omega_old * (1 - self.friction) - GRAV / self.rope * math.sin(theta_old) * step_size
        theta = (theta_old + omega * step_size)  # % 360

        self.omega = omega
        self.theta = theta

        theta_deg = math.degrees(theta)
        x, y = self.position_on_arc(theta_deg)

        self.data['theta_deg'].append(theta_deg)
        self.data['omega'].append(omega)

        self.data['x'].append(x)
        self.data['y'].append(y)

        velocity = math.sqrt(
                2 * GRAV * self.rope * abs(math.cos(theta) - math.cos(self.initial_theta))
        )

        # print(omega)
        lin_ve = (theta - theta_old) / step_size * self.rope
        # lin_ve = math.radians(omega) * self.rope / step_size

        # lin_ve = math.radians(omega / step_size) * self.rope
        # lin_ve = math.radians(theta_old - theta) * self.rope / step_size
        # omg_rad = math.radians(omega)
        # lin_ve = velocity

        # lin_ve = omega / 180 * math.pi * self.rope * self.rope
        # lin_ve = omega * self.rope  / 360

        ep = self.mass * GRAV * (self.rope + y)
        ek = (1 / 2) * self.mass * (lin_ve ** 2)
        total_en = ep + ek

        self.data['veloc'].append(velocity)
        self.data['lin_ve'].append(lin_ve)
        self.data['ep'].append(ep)
        self.data['ek'].append(ek)
        self.data['total_en'].append(total_en)
        self.data['factor'].append(velocity / omega)

        return x, y


"Model Creation"
center = (0, 0)
model = Model(*center, mass=1, rope=5, initial_theta=math.radians(90), initial_omega=0, stepsize=.01, friction=0)
initial = model.x, model.y
all_trace = []
offset_x = 300
offset_y = 300

for ti in range(1000):
    array = np.zeros((600, 600, 3), dtype=np.uint8) + 170
    step = model.step()
    all_trace.append(step)
    trace = list([*zip(*all_trace)])
    poly = np.array(all_trace[-30:])
    poly[:, 1] *= -1
    poly[:, :] *= 50
    poly[:, :] += 300
    x, y = step
    x = int(x * 50) + 300
    y = int(-y * 50) + 300

    cv2.line(array, (offset_x, offset_y), (x, y), (0, 0, 0), 2)  # Line
    cv2.circle(array, (offset_x, offset_y), 10, (0, 0, 160), -1)  # Anchor
    for n, tr in enumerate(poly):
        tr = np.array(tr, dtype=np.int32)
        col = np.array([50, 200, 0]) * (n / 30)
        cv2.circle(array, tuple(tr), 8, col, -1)
    cv2.circle(array, (x, y), 10, (250, 100, 50), -1)  # Ball

    cv2.imshow("Pendulum", array)

    key = cv2.waitKey(5)
    if key == ord("q"):
        break

trace = list([*zip(*all_trace)])

fig1 = plt.figure()
plt.title("Pendulum energy state")
plt.plot(model.data['total_en'], label="total en")
plt.plot(model.data['ek'], label="kinetic")
plt.plot(model.data['ep'], label="potential")
plt.legend(loc=1)

fig2 = plt.figure()
ax1 = plt.axes([0.1, 0.05, 0.8, 0.6])
ax2 = plt.axes([0.1, 0.7, 0.8, 0.2])

ax1.plot(model.data['veloc'], label="velocity (from mgh)", c='r')
ax1.plot(model.data['lin_ve'], label="lin_ve", c=(1, 1, 0.1), linewidth=3)
ax1.plot(model.data['omega'], label="omega", dashes=[10, 5], linewidth=3)
# ax1.plot(model.data['radial'], label="radial")
# ax1.plot(model.data['radial_deg'], label="radial_deg")
# ax1.plot(model.data['factor'], label="factor")
plt.plot()
ax1.legend()

ax2.plot(model.data['theta_deg'], label="theta degrees")
ax2.legend()

plt.figure()
plt.subplot(312)
plt.plot(model.data['x'], label="x")
plt.plot(model.data['y'], label='y')
plt.legend()

plt.show()
