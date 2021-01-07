from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import re

from collections import deque
from quaternion import quaternion
import quaternion as quat

CIRC_POINTS = 100


def draw_circle(ax, x,y,radius):
    X = np.linspace(0, 2*math.pi, CIRC_POINTS)
    Y = np.cos(X)*radius + y
    X = np.sin(X)*radius + x
    ax.plot(X, Y)

timeCache = {}

def timeit(fun):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fun(*args, **kwargs)
        end = time.time() - t0
        cache = timeCache.get(fun.__name__)

        if cache:
            cache.append(end)
            tend = np.mean(cache)
        else:
            cache = deque(maxlen=50)
            cache.append(end)
            tend = end

        if tend > 60:
            print(f"F({fun.__name__}) mean duration: {tend/60:>3.3f} m")
        elif tend < 1:
            print(f"F({fun.__name__}) mean duration: {tend*1000:>3.3f} ms")
        else:
            print(f"F({fun.__name__}) mean duration: {tend:>3.3f} s")
        return out
    return wrapper

points = np.array([
    #(1.4, 1.7),
    #(3, 0.5),
    #(1,-1),
    #(4,2),
    #*[(math.sin(p)*5, math.cos(p)*5) for p in np.linspace(0, 2*math.pi,500)],
    #*[(math.sin(p)*2+1, math.cos(p)*1, 0) for p in np.linspace(0, 2*math.pi,100)],
    #*[(math.sin(p)*2+1, math.cos(p)*1, 0.3) for p in np.linspace(0, 2*math.pi,200)],
    *[(math.sin(p)*1+1, math.cos(p)*2, 0.3) for p in np.linspace(0, 2*math.pi,200)],
    #*[(math.sin(p)*2+1, math.cos(p)*1, 1) for p in np.linspace(0, 2*math.pi,100)],
    #*[(math.sin(p)*4, math.cos(p)*1+1) for p in np.linspace(0, 2*math.pi,500)],
])


@timeit
def find_closest_points(pts):
    assert isinstance(pts, np.ndarray), "This should be numpy array!"
    if pts.shape[0] != 3:
        pts = pts.T

    out = np.empty((3,0))
    dists = [0] * pts.shape[1]
    for ind, pt in enumerate(pts.T):
        pt = np.array([pt]).T
        #print(pt)
        #print(box)
        dist = box - pt
        dist = dist * dist
        dist = dist.sum(axis=0)
        dist = np.sqrt(dist)
        hidist = np.min(dist)
        loc = np.where(hidist == dist)[0]
        loc = box[:, loc]
        #print(loc)
        out = np.concatenate([out, loc], axis=1)
        dists[ind] = hidist
        #out[:, ind] = loc

    return out, dists

@timeit
def find_closest(pt):
    if len(pt) == 2:
        pt = [*pt, 0]
    elif len(pt) != 3:
        raise ValueError("Point has incorrect shape")

    pt = np.array([pt]).T
    dist = box - pt
    #print(pt)
    #print("dist\n", dist)
    dist = dist * dist
    dist = dist.sum(axis=0)
    dist = np.sqrt(dist)
    hidist = np.min(dist)
    loc = np.where(hidist == dist)[0]
    #print(loc)
    loc = box[:, loc].T[0]
    #print("LOC", loc)
    return loc, hidist

def plot_single(ax):
    maxError = 0.001

    box = np.array([
        [0,0,1],
        [-0.2,0.8,0.3],
        [0,1,0.2],
        [0.5, 0.9, 0.50],
        [1,1,0],
        [1,0,0.8],
        [0.5, 0.5, 1],
    ]) + np.array([1,0,0])
    box = box.T

    q_30 = quaternion(math.radians(15), 0,0,1)

    for num in range(box.shape[1]):
        XY = box[:3, [num, num-1]]
        ax.plot(*XY, c=(0,0,0.7), linewidth=3)

    for ind, point in enumerate(points):
        if len(point) == 2:
            point = (*point, 0.0)
        #last_pt = ax.scatter(*point, c=[(0,0.3,1)])

        #draw_circle(ax, *point, 0.75)
        mark, dist = find_closest(point)
        dx, dy, dz = mark[:3] - point

        textx = point[0] + dx/2
        texty = point[1] + dy/2
        #color = [0.2,0.8,0.4]
        color = np.clip((*point[:2], dist), 0, 0.8)
        line = np.stack([point, mark], axis=0).T
        ax.plot(*line, color=color)
        border = np.stack([point, points[ind-1]], axis=0).T
        ax.plot(*border,color=[0,0,0], linewidth=3)

        #plt.text(textx, texty, f"{dist:>4.1f}")

box = np.array([
    #[0,0,1],
    #[-0.2,0.1, 0.3],
    [0, 0, 0.5],
    [0.2, 0, 0.5],
    [0, 0.3, 0.8],
    [-0.4, 0, 0.3],

]) + np.array([1,0,0])
box = box.T

def animate(i):
    i = i/15
    points = np.array([
        *[(math.sin(p)*0.5+1, math.cos(p)*.5, math.sin(i)*0.3+0.2) for p in np.linspace(0, 2*math.pi, 50)],
    ])
    ax.clear()
    color_mod = (0.2, 5, 1.9)
    dests, dists = find_closest_points(points)
    dests = dests.T
    dists = np.clip(dists, 0.1, 5)*5

    color = np.clip(dests*color_mod + points*0.4, 0, 1)

    segs = np.array([
        [[0,1,2], [1,1,1]],
        [[2,3,4], [1,1,1]],
    ])
    segs = np.stack([points, dests], axis=1)
    line_segments = Line3DCollection(segs, linewidths=dists, colors=color, linestyle='solid')
    ax.add_collection(line_segments)

    for num in range(box.shape[1]):
        XY = box[:3, [num, num-1]]
        ax.plot(*XY, linewidth=5, color=(0,0,0))

    lim = 1
    ax.set_xlim([1-lim/2, 1+lim/2])
    ax.set_ylim([-lim/2, lim/2])
    ax.set_zlim([-lim/2, lim/2])
    #ax.autoscale(enable=True)
    pitch = 10 + math.cos(i/10)*20
    pitch = 20
    yaw = 60 + i*3
    ax.view_init(pitch, yaw)

#plt.grid()
#plt.axis('equal')

fig = plt.gcf()
#ax = plt.gca(projection='3d')
ax = Axes3D(fig)
lim = 5
ani = FuncAnimation(fig, animate, interval=10)
ax.set_xlim([-1, lim-1])
ax.set_ylim([-2, lim-2])
ax.set_zlim([-lim/2, lim/2])
plt.show()








