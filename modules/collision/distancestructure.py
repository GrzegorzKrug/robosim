import numpy as np
import math
import time

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


class DistanceStruct:
    divide = 2
    def __init__(self, points):
        if type(points) is list:
            points = np.array(points)
        assert isinstance(points, np.ndarray), "Points have to be in numpy array"

        #self.splits = []
        #self.refs = []

        self._min = None
        self._max = None
        self._pivot = None

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


    #@timeit(0)
    def _compile(self):
        assert self.data.shape[-1] == 3, "This data structure is for points in 3d"

        data = self.data
        dims = len(data.shape)
        axis = [*range(dims-1)]
        pivot = np.median(self.data, axis=axis)
        low = self.data
        top = self.data
        for ax in axis:
            low = np.min(low, axis=0)
            top = np.max(top, axis=0)

        strlen = 10
        #print()
        #print('pivot'.ljust(strlen), pivot)
        #print('min'.ljust(strlen), low)
        #print('max'.ljust(strlen), top)

        self._min = low
        self._max = top
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

            if x >= pivot[0]:
                i = 1
            else:
                i = 0
            if y >= pivot[1]:
                j = 1
            else:
                j = 0
            if z >= pivot[2]:
                k = 1
            else:
                k = 0
            divisions[i][j][k].append(val)

        keys = [''.join(x) for x in product(['0','1'], repeat=3)]
        struct = dict.fromkeys(keys, None)

        for i, arrX in enumerate(divisions):
            for j, arrY in enumerate(arrX):
                for k, arrZ in enumerate(arrY):
                    #print()
                    #print(i, j, k)
                    #print(arrZ)
                    arr = np.array(arrZ).reshape(-1, 3)
                    ob = DistanceStruct(arr)
                    key = f"{i}{j}{k}"
                    struct[key] = ob
        self._struct = struct

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max
    
    @property
    def size(self):
        return self._size

    @property
    def value(self):
        return self._value

    def __len__(self):
        return self.size
    
    @timeit(10)
    def closest_point(self, poi):
        valid, ret = self._closest_point(poi)
        if not valid:
            raise ValueError("Why no point is returned?")
        return ret
    
    def _closest_point(self, poi):
        if self.size == 1:
            return True, self.value
        elif self.size == 0:
            return False, None

        pivot = self._pivot
        x,y,z = poi

        if x >= pivot[0]:
            i = 1
        else:
            i = 0
        if y >= pivot[1]:
            j = 1
        else:
            j = 0
        if z >= pivot[2]:
            k = 1
        else:
            k = 0

        key = f"{i}{j}{k}"
        ob = self._struct.get(key)
        valid, pt = ob._closest_point(poi)
        if valid:
            #print("Found point:", pt)
            return True, pt
        
        point_in_box = False

        if point_in_box:
            "Check outer scope"
            #print("RETURN IN BOX", poi)
            raise NotImplementedError
            return True, poi
        else:
            nextkeys = [f"{1-i}{j}{k}", f"{i}{1-j}{k}", f"{i}{j}{1-k}"]
            pts = []
            #print("asd")
            for key in nextkeys:
                ob = self._struct.get(key)
                valid, pt = ob._closest_point(poi)
                if valid:
                    pts.append(pt)
                    #print("valid pt:", pt)

            pts = np.array(pts)
            #print(pts.shape)
            #print(pts)
            #print(poi)

            if pts.shape[0] > 1:
                dst = poi - pts
                dst = np.sqrt(np.absolute(dst * dst).sum(axis=1))
                close = dst.min()
                pt = pts[dst == close][0]
                #print("closer point:", pt)
                return True, pt
            elif pts.shape[0] == 1:
                return True, pts[0, :]

        return False, None

line = [[0,0,0], [1,2,3]]
rand_pts = np.random.random((10, 3))*7+2
#rand_pts[:,2] = 0

ds = DistanceStruct(rand_pts)
t0 = time.time()
for x in range(50):
    pt = ds.closest_point(np.random.random(3)*3)
searchtime = time.time() - t0
print(f"Search time: {searchtime}s")

N = 5
color = [*zip(
        np.linspace(0, 0.7, N),
        np.linspace(0.9, 0.1, N),
        np.linspace(0.6, 0.2, N),
)]

all_dists = []

def animate(i):
    i = i/20
    #print(i)
    points = np.array([
        *[(math.sin(p+1)*5+5, math.cos(p+i/3)*5+5, math.sin(i)*4+4) 
        for p in np.linspace(0, math.pi, N)],
    ])
    ax.clear()
    #color_mod = (0.2, 5, 1.9)
    dests = np.array([ds.closest_point(pt) for pt in points])
    dists = points - dests
    dists = dists*dists
    dists = dists.sum(axis=-1)
    all_dists.append(dists)
    #print(dests)
    #dests = dests.T
    #dists = np.clip(dists, 0.1, 3)*3
#
    #color = np.clip(np.random.random((10, 3)), 0, 1)
    #
    #segs = np.array([
        #[[0,1,2], [1,1,1]],
        #[[2,3,4], [1,1,1]],
    #])
    segs = np.stack([points, dests], axis=1)
    line_segments = Line3DCollection(segs, colors=color, linestyle='solid')
    ax.add_collection(line_segments)
    for (x,y,z), distance in zip(points, dists):
        ax.text(x,y,z, f"{distance:>3.1f}")


    for point in rand_pts:
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


fig = plt.gcf()
ax = Axes3D(fig)
lim = 5
ani = FuncAnimation(fig, animate, interval=10)
plt.show()

plt.figure()
plt.plot(all_dists)
plt.title("Distance graph")
plt.show()


