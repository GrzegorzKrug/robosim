import numpy as np
import math
import time
import sys
import os

from collections import deque
from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def timeit(smooth=50):
    timeCache = {}
    def decorator(fun):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            out = fun(*args, **kwargs)
            end = time.time() - t0

            name = fun.__qualname__

            if smooth:
                cache = timeCache.get(name)
                if cache:
                    cache.append(end)
                    tend = np.mean(cache)
                else:
                    cache = deque(maxlen=int(smooth))
                    cache.append(end)
                    tend = np.mean(cache)
                    timeCache[name] = cache
            else:
                tend = end

            if tend > 60:
                print(f"F({name}) mean duration: {tend/60:>3.3f} m")
            elif tend < 1:
                print(f"F({name}) mean duration: {tend*1000:>3.3f} ms")
            else:
                print(f"F({name}) mean duration: {tend:>3.3f} s")
            return out
        return wrapper
    return decorator


class BruteDistance:
    def __init__(self, points):
        """
        Storing points in horizonal vectors
        """
        if points.shape[-1] != 3:
            points = points.T
        assert points.shape[-1] == 3, "This class is for 3D points"

        self.points = points

    #@timeit(0)
    def closest_point(self, poi):
        poi = np.array(poi)
        assert poi.shape[-1] == 3

        dist = self.points - poi
        dist = np.sqrt((dist*dist).sum(axis=-1))
        low = dist.min()
        pt = self.points[np.where(dist == low)][0]
        #print(pt)
        return pt, low


class DistanceStruct:
    def __init__(self, points, depth=0):
        if type(points) is list:
            points = np.array(points)
        assert isinstance(points, np.ndarray), "Points have to be in numpy array"

        self._min = None
        self._max = None
        self._gap = None
        self._pivot = None
        self._depth = depth

        "Transpose matrix to row vectors"
        if points.shape[-1] != 3:
            points = points.T
        self.data = points
        try:
            size = reduce(lambda x,y: x*y, points.shape[:-1])
        except TypeError:
            size = 0

        self._value = None
        self._size = size
        if size > 1:
            #print("size", size)
            self._compile()
        elif size == 1:
            self._value = points[0, :]
            self._pivot = self._value.copy()

    @property
    def depth(self):
        return self._depth

    def point_in_zone(self, poi):
        if self.min is not None and self.max is not None:
            xlow, ylow, zlow = self._min
            xhi, yhi, zhi = self._max

            x, y, z = poi
            if xlow <= x <= xhi:
                inX = True
            else:
                inX = False

            if ylow <= y <= yhi:
                inY = True
            else:
                inY = False
            if zlow <= z <= zhi:
                inZ = True
            else:
                inZ = False

            return (inX, inY, inZ)


    def _compile(self):
        assert self.data.shape[-1] == 3, "This data structure is for points in 3d"

        data = self.data
        dims = len(data.shape)
        axis = [*range(dims-1)]
        pivot = np.median(self.data, axis=axis)
        low = self.data
        top = self.data
        #pivot = self.data
        for ax in axis:
            #pivot = np.mean(pivot, axis=0)
            low = np.min(low, axis=0)
            top = np.max(top, axis=0)

        strlen = 10

        self._min = low
        self._max = top
        self._gap = top - low
        self._gap_magn = np.sqrt((self._gap * self._gap).sum())
        #print(self.gap_magn)
        self._pivot = pivot

        divisions = [
            [
                [[], []],
                [[], []],
            ],
            [
                [[], []],
                [[], []],
            ],
        ]
        for val in self.data:
            x,y,z = val
            px, py, pz = pivot
            if x - px >= 0:
                i = 1
            else:
                i = 0
            if y - py >= 0:
                j = 1
            else:
                j = 0
            if z - pz >= 0:
                k = 1
            else:
                k = 0
            divisions[i][j][k].append(val)

        keys = [''.join(x) for x in product(['0','1'], repeat=3)]
        struct = dict.fromkeys(keys, None)

        for i, arrX in enumerate(divisions):
            for j, arrY in enumerate(arrX):
                for k, arrZ in enumerate(arrY):
                    arr = np.array(arrZ).reshape(-1, 3)
                    if len(arr) < 1:
                        continue
                    ob = DistanceStruct(arr, self.depth+1)
                    key = f"{i}{j}{k}"
                    struct[key] = ob
        self._struct = struct

    def pivot_distance(self, pt):
        dist = self._pivot - pt
        dist = np.sqrt((dist*dist).sum())
        return dist

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def gap(self):
        return self._gap

    @property
    def gap_magn(self):
        return self._gap_magn

    @property
    def size(self):
        return self._size

    @property
    def value(self):
        return self._value

    def __len__(self):
        return self.size

    #@timeit(0)
    def closest_point(self, poi):
        valid, pt, dist = self._closest_point(poi)
        if not valid:
            raise ValueError("Why no point is returned?")
        return pt, dist

    @staticmethod
    def distance(point1, point2):
        dist = point1-point2
        dist = np.sqrt((dist*dist).sum())
        return dist

    def _closest_point(self, poi):
        if self.size == 1:
            dist = self.distance(self.value, poi)
            return True, self.value, dist
        elif self.size == 0:
            return False, None, None

        x,y,z = poi
        px, py, pz = self._pivot
        if x - px >= 0:
            i = 1
        else:
            i = 0
        if y - py >= 0:
            j = 1
        else:
            j = 0
        if z - pz >= 0:
            k = 1
        else:
            k = 0


        key = f"{i}{j}{k}"
        best_dist = math.inf
        best_pt = None
        solution = None

        inZone = self.point_in_zone(poi)
        val = sum(inZone)

        if val == 3:
            for key, ob in self._struct.items():
                if ob:
                    valid, pt, dist = ob._closest_point(poi)
                    if valid and dist < best_dist:
                        best_dist = dist
                        best_pt = pt
            #assert best_pt is not None, f"No point? is this ob emtpy? {self.size}"
            return True, best_pt, best_dist

        elif val == 0:
            ob = self._struct.get(key)
            if ob:
                valid, pt, dist = ob._closest_point(poi)
                return valid, pt, dist
            keys = [f"{1-i}{j}{k}", f"{i}{1-j}{k}", f"{i}{j}{1-k}"]
            #else:
                #return False, None, None
        else:
            keys = [key]

        "Invert keys"

        #if val and (self.size <= 100 or self.depth <= 10) and False:
        #if self.size < 10 or self.depth < 5:
            #if val == 0:
                #ini, inj, ink = 1,1,1
            #else:
                #ini, inj, ink = inZone
            #for x in range(int(ini)+1):
                #for y in range(int(inj)+1):
                    #for z in range(int(ink)+1):
                        #keys.append(
                            #f"{x ^ i}" \
                            #+ f"{y ^ j}"\
                            #+ f"{z ^ k}"
                        #)
            #keys = list(set(keys))


        for key in keys:
            ob = self._struct.get(key)
            if ob is None:
                continue
            valid, pt, dist = ob._closest_point(poi)
            if valid and dist < best_dist:
                best_dist = dist
                best_pt = pt

        if best_pt is not None:
            return True, best_pt, best_dist


        "use pivot to get closest"
        piv_dist = math.inf
        for key, ob in self._struct.items():
            if ob and key not in keys:
                dist = ob.pivot_distance(poi)
                #arc = 2 * math.pi * dist
                if self.size > 100 and self.gap_magn**2 < dist:
                    #print(self.gap_magn, 2*dist)
                    continue

                valid, pt, dist = ob._closest_point(poi)
                if valid and dist < best_dist:
                    best_pt = pt
                    best_dist = dist

        if best_pt is not None:
            return True, best_pt, best_dist

        return False, None, None


rand_pts = np.random.random((500, 3))*7+2
ds = DistanceStruct(rand_pts)
bs = BruteDistance(rand_pts)

N = 3
MERGE = True

color = [*zip(
        np.linspace(0, 0.7, N),
        np.linspace(0.9, 0.1, N),
        np.linspace(0.6, 0.2, N),
)]

all_dists = []

def animate(i):
    i = i/10
    #print(i)
    inner = np.array([
        *[(math.sin(p+5*i)*5+5, math.cos(p+i)*3+5, math.sin(i)*2+3)
        for p in np.linspace(0, 2*math.pi/(N)*(1+N), N)],
    ])
    outer = np.array([
        *[(math.sin(p+5*i)*5+35, math.cos(p+i)*3+35, math.sin(i)*2+20)
        for p in np.linspace(0, 2*math.pi/(N)*(1+N), N)],
    ])
    if MERGE:
        points = np.concatenate([outer, inner], axis=0)
    else:
        point = outer
    #print(points.shape)
    ax.clear()

    #dests = [(0,0,0) if ds.point_in_zone(pt) else ds.closest_point(pt) for pt in points]
    try:
        print("= ="*10, "NEW", "= ="*10)
        out = [ds.closest_point(pt) for pt in points]
    except ValueError:
        print("empty points")
        return 0
        out =[(0,0,0), [0]]

    dests, dists = [*zip(*out)]
    #dists = points - dests
    #dists = dists*dists
    #dists = np.sqrt(dists.sum(axis=-1))

    out = [bs.closest_point(pt) for pt in points]
    brute_dests, brute_dists = [*zip(*out)]
    brute_dests = np.array(brute_dests)
    brute_dists = np.array(brute_dists)

    merged = [*dists, *brute_dists]
    all_dists.append(merged)

    #color = np.clip(np.random.random((10, 3)), 0, 1)
    #
    #segs = np.array([
        #[[0,1,2], [1,1,1]],
        #[[2,3,4], [1,1,1]],
    #])

    segs = np.stack([points, dests], axis=1)
    brute_segs = np.stack([points, brute_dests], axis=1)

    line_segments = Line3DCollection(segs, colors=(1,0,0.2), linestyle='solid')
    brute_segments = Line3DCollection(brute_segs, colors=(0,1,0), linestyle='solid')

    ax.add_collection(line_segments)
    ax.add_collection(brute_segments)

    for (x,y,z), distance in zip(points, dists):
        ax.text(x,y,z, f"{distance:>3.1f}")


    for pind, point in enumerate(rand_pts):
        if pind > 100:
            break
        XY = point
        ax.scatter(*XY, linewidth=2, color=(0,0,0,1))

    lim = 10
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])
    ax.set_zlim([0, lim])

    ##ax.autoscale(enable=True)
    #pitch = 10 + math.cos(i/3)*20
    #pitch = 50
    #yaw = 60 + i*3
    #yaw = 30
    #ax.view_init(pitch, yaw)

def benchmark(repeat=1000):
    series = [*list(range(1, 500, 15)), *[500*i for i in range(1, 21)], 30000]
    times_ds = []
    times_bs = []

    for ser_size in series:
        print(ser_size)
        points = np.random.random((ser_size, 3))*10
        ds = DistanceStruct(points)
        bs = BruteDistance(points)

        tms_ds = []
        tms_br = []
        for x in range(repeat):
            poi = np.random.random(3) * 15
            t0 = time.time()
            ds.closest_point(poi)
            tend = time.time()
            dur = tend - t0
            tms_ds.append(dur)

            t0 = time.time()
            bs.closest_point(poi)
            tend = time.time()
            dur = tend - t0
            tms_br.append(dur)

        time_ds_mean = np.mean(tms_ds)
        time_br_mean = np.mean(tms_br)

        times_ds.append(time_ds_mean)
        times_bs.append(time_br_mean)

    plt.figure(figsize=(8,6))
    plt.plot(series, times_ds, label="Data structure")
    plt.plot(series, times_bs, label="Numpy calculations")
    plt.grid()
    plt.legend(loc="best")
    plt.ylabel("Time to find closest point")
    plt.xlabel("Points in shape")
    plt.title("Time comparison of algos")

    plt.show()
    sys.exit(0)


if __name__ == "__main__":
    #benchmark()


    fig = plt.gcf()
    ax = Axes3D(fig)
    lim = 5
    ani = FuncAnimation(fig, animate, interval=10)
    plt.show()

    plt.figure()
    if MERGE:
        N = N * 2
    colors = [
        *[(1,0,n/N) for n in range(N)],
        *[(0,1,n/N) for n in range(N)],
    ]

    values = [*zip(*all_dists)]

    for val, col in zip(values, colors):
        plt.plot(val, color=col)

    plt.title("Distance graph, Median")
    plt.show()


