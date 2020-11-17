import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
from matplotlib import style
from PIL import Image

style.use("Solarize_Light2")

GRAV = 9.81


class SinglePendulum:
    def __init__(self, cx, cy, mass=1, rope=5, initial_theta=1.0, initial_omega=0.0, friction=0.5, step_size=1.0):
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

        self.step_size = step_size
        "Collect Data"
        self.data = {
                "theta": [],
                "theta_deg": [],
                "omega": [],
                "veloc": [],
                "lin_ve": [],
                "radial": [],
                "x": [],
                "y": [],
                "ep": [],
                "ek": [],
                "total_en": [],
                "h": [],
        }
        "Initialization"
        self.x, self.y = self.position_on_arc(initial_theta)

    def position_on_arc(self, theta):
        x, y = self.anchor

        x = x + self.rope * np.sin(theta)
        y = y - self.rope * np.cos(theta)
        return x, y

    def step(self, step_size=None):
        if not step_size:
            step_size = self.step_size
        self.time += step_size

        theta_old = self.theta
        omega_old = self.omega

        accel = -GRAV * math.sin(theta_old) / self.rope
        omega = omega_old * (1 - self.friction * step_size) + accel * step_size
        theta = theta_old + omega * step_size

        self.omega = omega
        self.theta = theta
        self.save_data()

    def save_data(self):
        theta = self.theta
        omega = self.omega

        theta_deg = math.degrees(theta)

        self.data['theta_deg'].append(theta_deg)
        self.data['theta'].append(theta)
        self.data['omega'].append(omega)

        x, y = self.position_on_arc(theta)
        self.data['x'].append(x)
        self.data['y'].append(y)

        velocity = math.sqrt(
                2 * GRAV * self.rope * abs(math.cos(theta) - math.cos(self.initial_theta))
        )

        lin_ve = omega * self.rope

        h = self.rope * (1 - math.cos(theta))

        ep = self.mass * GRAV * h
        ek = (1 / 2) * self.mass * (lin_ve ** 2)
        total_en = ep + ek

        self.data['veloc'].append(velocity)
        self.data['lin_ve'].append(lin_ve)
        self.data['ep'].append(ep)
        self.data['ek'].append(ek)
        self.data['total_en'].append(total_en)
        self.data['h'].append(h)

        return x, y


def val_matrix_to_color_matrix(arr):
    arr = arr[:, :, np.newaxis]
    high = arr.max()
    low = arr.min()
    arr = (arr - low) / high
    color = (1 - arr) * (1, 0, 0)
    return color


def pendulum_gradient(plot_range=5, N=45, rope=3, step_size=0.001, friction=0.001):
    T = plot_range
    vec = (np.arange(N) - N / 2) / N * T * 2
    vec_x = vec
    vec_y = vec
    X, Y = np.meshgrid(vec_x, -vec_y)
    arr = np.stack([X, Y], axis=-1)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    C = np.zeros((N, N))
    for row_ind, row in enumerate(arr):
        for col_ind, (prev_th, prev_om) in enumerate(row):
            omega = (
                    prev_om * (1 - friction * step_size)
                    - GRAV / rope * math.sin(prev_th) * step_size
            )
            theta = prev_th + prev_om * step_size

            om_change = (omega - prev_om)
            th_change = prev_om * step_size

            # plt.arrow(this_th, this_om, th_change, om_change, head_width=0.3, head_length=0.3,
            #           length_includes_head=True)
            U[row_ind, col_ind] = th_change
            V[row_ind, col_ind] = om_change
            C[row_ind, col_ind] = abs(om_change) + abs(th_change)

    plt.pcolormesh(X, Y, C, shading='auto')
    # plt.contourf(X, Y, C)

    dot = None
    red_dot = None
    for num in range(-T, T + 1):
        stable = num % (2 * np.pi)
        unstable = num % (np.pi)
        if stable < 1:
            pos = num - stable
            dot = plt.scatter(pos, 0, c=[(0, 0.7, 0.2)])
        elif unstable < 1:
            pos = num - unstable
            red_dot = plt.scatter(pos, 0, c=[(0.9, 0.3, 0.1)])

    if dot:
        dot.set_label("stable")
    if red_dot:
        red_dot.set_label("unstable")

    plt.quiver(X, Y, U, V, width=0.001)
    plt.xlabel("Theta")
    plt.ylabel("Omega")
    plt.title("Pendulum gradient")
    plt.suptitle(f"friction: {friction}, step-size:{step_size}")
    # leg = plt.legend().get_label()
    # print(leg)

    plt.legend()
    plt.colorbar()
    plt.axis([-plot_range, plot_range, -plot_range, plot_range])
    # plt.axis('equal')

    return ax


"Model Creation"


def run_simulation_pendulum(name=None, gradient_ax=None, friction=0.0, time_range=1000, step_size=0.01,
                            initial_theta=0.0, initial_omega=0.0):
    center = (0, 0)
    model = SinglePendulum(*center, mass=1, rope=3, initial_theta=initial_theta, initial_omega=initial_omega,
                           step_size=step_size,
                           friction=friction)
    all_trace = []
    offset_x = 300
    offset_y = 300

    for ti in range(time_range):
        model.step()
        x, y = model.data['x'][-1], model.data['y'][-1]
        all_trace.append((x, y))
        if RENDER or SHOW:
            array = np.zeros((600, 600, 3), dtype=np.uint8) + 170
            poly = np.array(all_trace[-30:])
            poly[:, 1] *= -1
            poly[:, :] *= 50
            poly[:, :] += 300
            x = int(x * 50) + 300
            y = int(-y * 50) + 300

            cv2.line(array, (offset_x, offset_y), (x, y), (0, 0, 0), 2)  # Line
            cv2.circle(array, (offset_x, offset_y), 10, (0, 0, 160), -1)  # Anchor
            for n, tr in enumerate(poly):
                tr = np.array(tr, dtype=np.int32)
                col = np.array([50, 200, 0]) * (n / 30)
                cv2.circle(array, tuple(tr), 8, col, -1)
            cv2.circle(array, (x, y), 10, (250, 100, 50), -1)  # Ball
            cv2.putText(array, f"{ti:,}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), )

            if SHOW:
                cv2.imshow("Pendulum", array)
                key = cv2.waitKey(5)
                if key == ord("q"):
                    break

    if PLOT:
        fig1 = plt.figure()
        plt.title("Pendulum energy state")
        plt.plot(model.data['total_en'], label="total en")
        plt.plot(model.data['ek'], label="kinetic")
        plt.plot(model.data['ep'], label="potential")
        ticks = plt.xticks()[0]
        plt.xticks(np.linspace(ticks[0], ticks[-1], 15))
        plt.legend(loc=1)

        fig2 = plt.figure()
        ax1 = plt.axes([0.1, 0.05, 0.8, 0.6])
        ax2 = plt.axes([0.1, 0.7, 0.8, 0.2])

        ax1.plot(model.data['lin_ve'], label="lin_ve", c=(1, 0, 0.7), linewidth=1)
        ax1.plot(model.data['omega'], label="omega", linewidth=1)
        plt.title("Position and speed")
        ax1.legend()

        ax2.plot(model.data['theta_deg'], label="theta degrees")
        ax2.legend()

        plt.figure()
        plt.plot(model.data['x'], label="x", )
        plt.plot(model.data['y'], label='y', c=[1, 0.2, .6])
        plt.plot(model.data['h'], label='h', dashes=[4, 4, 2, 4], c=[0.8, 0.2, .2])

        ticks = plt.xticks()[0]
        plt.xticks(np.linspace(ticks[0], ticks[-1], 15))
        plt.legend()

    if gradient_ax:
        if not name:
            name = "Pendulum"
        # omg = np.array(model.data['omega']) * -1
        omg = np.array(model.data['omega'])
        c = np.random.random(3)
        c[1] = c[1] * 0.4 + 0.5
        gradient_ax.plot(model.data['theta'], omg, label=name, color=c)
        gradient_ax.legend(loc=1)


friction = 0.4
# friction = 0
step_size = 0.01
SHOW = False
RENDER = False
PLOT = False

# ax = None
initial_theta = math.radians(179 + 180)
initial_omega = -7.2 + 0.4
time_range = 5_000
ax = pendulum_gradient(N=70, plot_range=10, friction=friction, step_size=step_size)

run_simulation_pendulum(f"Pendulum, {step_size}", ax,
                        friction=friction, time_range=time_range, step_size=step_size,
                        initial_theta=initial_theta, initial_omega=initial_omega)
run_simulation_pendulum(f"Pendulum2 {step_size * 10}", ax,
                        friction=friction, time_range=time_range // 10, step_size=step_size * 10,
                        initial_theta=initial_theta, initial_omega=initial_omega)
run_simulation_pendulum(f"Pendulum3 {step_size / 10}", ax,
                        friction=friction, time_range=time_range * 10, step_size=step_size / 10,
                        initial_theta=initial_theta, initial_omega=initial_omega)
fig = ax.figure
fig.savefig("pendulum.jpg")

plt.show()
