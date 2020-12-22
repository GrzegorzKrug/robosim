import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


class PlainModel:
    def __init__(self, segments, anchor=None):
        """
        Segments - list of lengths
        Anchor - anchor point, root
        """
        self.dimension = 2

        if anchor is None:
            anchor = [0, 0]
        assert self.dimension == len(anchor)

        self.anchor = np.array(anchor)
        self.segments = segments
        self._joints = np.array(segments) * 0

        self._cartesian = None, None
        self._pos = self.get_cartesian(self.joints)

    @property
    def joints(self):
        return self._joints

    @joints.setter
    def joints(self, joints):
        """Set nodes position from joints in radians"""
        self._pos = self.get_cartesian(joints)
        self._joints = np.array(joints)

    @property
    def joints_deg(self):
        joints = [math.degrees(deg) for deg in self._joints]
        return joints

    @joints_deg.setter
    def joints_deg(self, joints):
        """Convert degress to radians and change position"""
        joints = [math.radians(deg) for deg in joints]
        self.joints = joints

    @property
    def cartesian(self):
        return self._pos

    def get_cartesian(self, joints: "Array[rads]" = None):
        """
        Returns end points of each segment
        """
        if joints is None:
            return self._pos

        assert len(joints) == len(
            self.segments), "You passed invalid joint value:"+str(joints)
        all_pos = np.zeros((self.dimension, len(self.segments)))
        absangle = 0
        for ind, (seg, angle) in enumerate(zip(self.segments, joints)):
            absangle += angle
            if ind == 0:
                prev_pos = self.anchor.copy()
            else:
                prev_pos = all_pos[:, ind-1]

            print(f"Prev: {prev_pos}")
            x = math.sin(absangle)*seg
            y = math.cos(absangle)*seg
            all_pos[:, ind] = prev_pos + [x, -y]

        return all_pos


def plot_model_position(ax, model):
    prev_ = model.anchor
    ax.scatter(*model.anchor, c=(1, 0, 0), s=100, marker='s')

    for pt in model.cartesian.T:
        "[[xa, xb], [ya, yb]"
        arr = np.stack([prev_, pt], axis=1)
        ax.plot(*arr, c=(0, 0, 0))
        prev_ = pt

    ax.scatter(*model.cartesian, s=50)


if __name__ == "__main__":
    model = PlainModel([5, 20, 10, 5, 5])
    fig = plt.figure(figsize=(9, 9))
    ax = plt.gca()

    "Plot"
    model.joints_deg = [0, -50, 0, 0, 50]
    plot_model_position(ax, model)

    plt.grid()
    ticks = list(range(-40, 45, 5))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.ylim([-50, 50])
    plt.xlim([-50, 50])
    plt.show()
