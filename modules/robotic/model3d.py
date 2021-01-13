from quaternion import quaternion
from functools import wraps
from copy import copy, deepcopy

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import quaternion as quat
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


def point_fromA(orien, offset=None, point=None):
    """
    Convert points in origin frame to relative
    """
    if offset is None:
        offset = [0,0,0]
    assert len(point) >= 3, "Point has to have at least 3 coords"
    point = np.array(point) - offset
    orien = orien.T
    return np.dot(orien, point)


def point_fromB(orien, offset=None, point=None):
    """
    Convert points in relative axis to parent
    """
    if offset is None:
        offset = [0,0,0]
    assert len(point) >= 3, "Point has to have at least 3 coords"
    point = np.array(point)
    pointinA = np.dot(orien, point)
    pointinA += offset
    return pointinA


def get_quat(deg, axis):
    """
    Generate quad from degrees and axis vector.
    Remeber to normalize quaternion!
    Args:
        deg - value in degrees
        axis - axis vector should be normalized to 1
    """
    axis = np.array(axis)
    radians = math.radians(deg)
    c1 = math.cos(radians/2)
    s1 = math.sin(radians/2)
    sins = s1* np.array(axis)
    q1 = quaternion(c1, *sins)
    return q1

def get_quat_normalised(deg, axis):
    """
    Generate quad from degrees and axis vector.
    Remeber to normalize quaternion!
    Args:
        deg - value in degrees
        axis - axis vector should be normalized to 1
    """
    axis = np.array(axis)
    axis = axis / np.sqrt(np.sum(axis*axis))
    radians = math.radians(deg)
    c1 = math.cos(radians/2)
    s1 = math.sin(radians/2)
    sins = s1* np.array(axis)
    q1 = quaternion(c1, *sins)
    return q1

class RelativeCoordinate:
    def __init__(self, pos=0, angle=0, null=0, *args, offset=None, **kwargs):
        """
        Define new coord in relation to root.
        Otherwise define A in relation to B and use `switch`, which is less intuitive.
        A stands for root axis,
        B stands for relative frame of interests

        Description:
        Sets Transformation from given objects, only first definition is set
        Args:
            Transformation: 4x4 Matrix contatingn 3x3 roation and 3x1 Translation
            Rotation: 3x3 Matrix
            Translation: Matrix/List/Tuple of size 3
        """
        self._transformation = None
        self._quaternion = None


        if offset is not None:
            self._offset = np.array(offset)
        else:
            self._offset = np.array([0,0,0])

        self.set_transformation(*args, **kwargs)

    def set_transformation(self, axis=None, up=None, offset=None):
        """
        Define transformation
        """
        qt = self.get_qt_from_ax(axis, up)
        self.quaternion = qt

    @property
    def offset(self):
        "Returns child offset"
        return self._offset

    @offset.setter
    def offset(self, offset):
        offset = np.array(offset)
        assert offset.shape == (3,), "Offset is in shape 3"
        self._offset = offset
        return self._offset

    @classmethod
    def get_qt_from_ax(self, axis=None, up=None):
        """
        Get rotation that describes given axis in relation to origin.
        You can specify any possible axis combination, "xyz", "-xyz".
        If axis will be same as up, vector up will be default(quickest rotation to make forward).
        Args:
            axis - forward ax that will be in origin X. accepts: x,y,z
            up - axis that will be in origin Z. accepts: x,y,z
        """
        if axis:
            axis = axis.lower()
        if up:
            up = up.lower()

        if axis == "x" or axis == "-x" or axis is None:
            if axis == '-x':
                q1 = get_quat(180, [0,0,1])
            else:
                q1 = quaternion(1, 0,0,0)

            if up == "y":
                q2 = get_quat(90, [1,0,0])
            elif up == "-y":
                q2 = get_quat(-90, [1,0,0])
            elif up == "-z":
                q2 = get_quat(180, [1,0,0])
            else:
                q2 = quaternion(1, 0,0,0)

            Q = q1 * q2
            return Q

        elif axis == "y" or axis == "-y":
            if up == "x":
                q2 = get_quat(-90, [0,1,0])
            elif up == "-x":
                q2 = get_quat(90, [0,1,0])
            elif up == "-z":
                q2 = get_quat(180, [0,1,0])
            else:
                q2 = quaternion(1, 0,0,0)

            if axis == "y":
                q1 = get_quat(-90, [0,0,1])
            else:
                q1 = get_quat(90, [0,0,1])
            Q = q1 * q2
            return Q

        else:
            if up is not None:
                if up == "y":
                    q1 = get_quat(90, [1,0,0])
                    angle = 90
                elif up == "-y":
                    q1 = get_quat(-90, [1,0,0])
                    angle = -90
                elif up == "-x":
                    q1 = get_quat(90, [0,1,0])
                    angle = 0
                else:
                    q1 = get_quat(-90, [0,1,0])
                    angle = 180

                if axis == "z":
                    q2 = get_quat(angle, [0,0,1])
                else:
                    angle = 180 + angle
                    q2 = get_quat(angle, [0,0,1])

                Q = q2 * q1
            else:
                if axis == "z":
                    Q = get_quat(90, [0,1,0])
                else:
                    Q = get_quat(-90, [0,1,0])
            return Q

    def get_rotation(self, deg, axis=None):
        """
        Get quaternion from axis component. Use XYZ or vector
        Supports quad normalization
        ARGS:
            deg - value in degrees
            axis - rotation vector, 'xyz' letter or size3 List / Array
        RETURN:
            qt - normalized quaternion
        """
        if axis:
            try:
                axis = axis.lower()
            except AttributeError:
                pass

        if axis == "x":
            ax_vect = [1, 0, 0]
        elif axis == "y":
            ax_vect = [0, 1, 0]
        elif axis == "z":
            ax_vect = [0, 0, 1]
        else:
            ax_vect = np.array(axis)
            assert ax_vect.shape == (3,), "Size of Vect must be 3."
            "NORMALIZE VECTOR TO 1"
            ax_vect = ax_vect / np.sqrt((ax_vect**2).sum())

        qt = get_quat(deg, ax_vect)
        return qt

    def add_rotation(self, *args, **kwargs):
        """
        Applies rotation to model. Specify rotation in format `get_rotation`
        ARGS:
            deg - value in degrees
            axis - xyz letter or size3 List / Array
        """
        qt = self.get_rotation(*args, **kwargs)
        self._quaternion = self._quaternion * qt
        return qt

    @staticmethod
    def get_rotation_matrix(rad: "Angle in Radians", axis):
        """
        Rotation matrix
        """
        axis = axis.lower()
        if axis == 'x':
            rot = [
                [1, 0, 0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad), math.cos(rad)],
            ]
        elif axis == 'y':
            rot = [
                [math.cos(rad), 0, -math.sin(rad)],
                [0, 1, 0],
                [math.sin(rad), 0, math.cos(rad)],
            ]
        else:
            rot = [
                [math.cos(rad),  math.sin(rad), 0],
                [-math.sin(rad), math.cos(rad), 0],
                [0, 0, 1],
            ]
        return rot

    def rotateEuler(self, x=None, y=None, z=None):
        raise NotImplementedError

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, new_val):
        self._transf_need_up = True
        self._angle = new_val

    @property
    def transf(self):
        "Redundandt property"
        return self.transformation

    @property
    def orientation_mat(self):
        return quat.as_rotation_matrix(self.quaternion)

    @property
    def quaternion(self):
        return copy(self._quaternion)

    @quaternion.setter
    def quaternion(self, new_qt):
        assert isinstance(new_qt, quaternion)
        self._quaternion = new_qt

    @property
    def orientation(self):
        "Redundant of quaternion"
        return self.quaternion

    @property
    def transformation(self):
        orient = self.orientation_mat
        transf = np.eye(4)
        transf[:3, :3] = orient
        transf[:3, -1] = self.offset
        return transf

    @property
    def endFrame(self):
        return self.transformation

    def get_point_fromA(self, point, apply_translation=True):
        """
        Convert points in origin frame to relative
        """
        assert len(point) >= 3, "Point has to have at least 3 coords"
        point = np.array(point) - self.offset
        orien = self.orientation_mat.T
        return np.dot(orien, point)

    def get_point_fromB(self, point, apply_translation=True):
        """
        Convert points in relative axis to parent
        """
        assert len(point) >= 3, "Point has to have at least 3 coords"
        point = np.array(point)
        pointinA = np.dot(self.orientation_mat, point)
        pointinA += self.offset
        return pointinA

    def visualise(self, points=None, block=True, prec=3, show_back=True):
        """
        Render origin frames and relative frame.
        Scatter 1,1,1 point on each frame.
        Show point values in both frames.
        """
        fig = plt.figure(figsize=(16,9))
        ax = Axes3D(fig)
        dashes = [0.8, 0.5]
        axis_width = 4
        relative_intense = 0.7
        shadow_width = 2

        ax.plot([0, 1], [0, 0], [0, 0], c=(1, 0, 0),
                label="{A} origin X", linewidth=axis_width)
        ax.plot([0, 0], [0, 1], [0, 0], c=(0, 1, 0),
                label="{A} origin Y", linewidth=axis_width)
        ax.plot([0, 0], [0, 0], [0, 1], c=(0, 0, 1),
                label="{A} origin Z", linewidth=axis_width)

        new_axis = np.array([
            [0, 0, 0, 0], [1, 0, 0, 0],
            [0, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 0, 0], [0, 0, 1, 0],
        ]).T

        shadow = np.dot(self.transformation, new_axis)
        new_axis[3, :] = 1
        absolut = np.dot(self.transformation, new_axis)

        a111_asb = self.get_point_fromA([1,1,1])
        b111_asa = self.get_point_fromB([1,1,1])

        for ind in range(0, 6, 2):
            start = absolut[:, ind]
            end = absolut[:, ind+1]
            sh_start = shadow[:, ind]
            sh_end = shadow[:, ind+1]

            lab = "X" if ind == 0 else "Y" if ind == 2 else "Z"
            col = (relative_intense, 0, 0) if ind == 0 \
                else (0, relative_intense, 0) if ind == 2 \
                else (0, 0, relative_intense)

            "Plot aboslute Axis"
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    label=f"{{B}} rel. {lab}", c=col, dashes=dashes, linewidth=axis_width)
            "Plot Shadow, no translation"
            ax.plot(
                [sh_start[0], sh_end[0]],
                [sh_start[1], sh_end[1]],
                [sh_start[2], sh_end[2]],
                label=f"{{B}} shadow {lab}", c=col, dashes=dashes, linewidth=shadow_width)

        default = np.array([[1, 1, 1], b111_asa[:3]])
        default = np.array([
            [1,1,1],
            self.get_point_fromB([1,0,0]),
            self.get_point_fromB([0,1,0]),
            self.get_point_fromB([0,0,1]),
        ])
        if points:
            points = np.array(points)
            try:
                if points.shape[1] != 3:
                    points = points.T
            except IndexError:
                points = points[np.newaxis, :]

            draw_points = np.concatenate([default, points], axis=0)

        else:
            draw_points = default

        for point in draw_points:
            B_pt = self.get_point_fromA(point)
            col = np.random.random(3)*0.8

            ax.scatter(*point, c=(col,))
            ax.text(*point,
                    # Label A
                    f"{{A}}:{point[0]:>3.{prec}f}, " +
                    f"{point[1]:>3.{prec}f}, {point[2]:>3.{prec}f}\n" +
                    # Label B
                    f"{{B}}:{B_pt[0]:>3.{prec}f}, " +
                    f"{B_pt[1]:>3.{prec}f}, {B_pt[2]:>3.{prec}f}",

                    horizontalalignment="left",
                    #bbox = {'facecolor':col},
                    fontdict={"size": 7, "color": col, "weight": 800},
                    )

            A_pt = self.get_point_fromB(point)
            rev_A = self.get_point_fromB(B_pt)
            rev_B = self.get_point_fromA(A_pt)
            #print()
            #print("point A:")
            #print(point)
            #print("revrese A:")
            #print(rev_A)
            #print("reverse B:")
            #print(rev_B)

            if show_back:
                ax.scatter(*A_pt, c=(col,))
                ax.text(*A_pt,
                        # Label A
                        f"{{B}}:{point[0]:>3.{prec}f}, " +
                        f"{point[1]:>3.{prec}f}, {point[2]:>3.{prec}f}\n" +
                        # Label B
                        f"{{A}}:{A_pt[0]:>3.{prec}f}, " +
                        f"{A_pt[1]:>3.{prec}f}, {A_pt[2]:>3.{prec}f}",

                        horizontalalignment="left",
                        #bbox = {'facecolor':col},
                        fontdict={"size": 7, "color": col, "weight": 800},
                        )

        "Combine all points to fine plot limits"
        stack = np.concatenate(
            [shadow[:3, :], absolut[:3, :], draw_points.T], axis=1)

        margin = 0.5
        ax_min = np.min(stack, axis=1)
        ax_max = np.max(stack, axis=1)
        ax.set_xlim([ax_min[0]-margin, ax_max[0]+margin])
        ax.set_ylim([ax_min[1]-margin, ax_max[1]+margin])
        ax.set_zlim([ax_min[2]-margin, ax_max[2]+margin])

        plt.title("Relative axis")
        plt.legend(loc=2)
        plt.show(block=block)
        plt.close()


class Segment(RelativeCoordinate):
    def __init__(self, /, rotation_axis=None, name=None,
            sgtype="joint", null=0, angle=0, pos=0,
            **kwargs):
        super().__init__(**kwargs)
        self._limit_low = 0
        self._limit_up = math.pi
        self.name = name
        self._rotation_vec = None
        self.rotation_vec = rotation_axis

        "Linear move"
        self.pos = pos
        "Joint Move"
        self._angle = angle
        self.null = null

        if sgtype != "joint":
            raise NotImplementedError(
                "Only joint segments supported in current state")

    @property
    def rotation_vec(self):
        return self._rotation_vec

    @rotation_vec.setter
    def rotation_vec(self, vec):
        self._rotation_vec = None
        if type(vec) is str:
            vec = vec.lower()
            if "x" in vec:
                out = [1, 0, 0]
            elif "y" in vec:
                out = [0, 1, 0]
            else:
                out = [0, 0, 1]

            rot_vec = np.array(out)
            if "-" in vec:
                rot_vec = rot_vec * -1

            self._rotation_vec = rot_vec
        elif vec:
            vec = np.array(vec)
            assert vec.shape in ((3,1), (3,)), f"Is this vector of 3? {vec.shape}"
            self._rotation_vec = vec
        else:
            vec = np.array([0,0,1])
            self._rotation_vec = vec

    @property
    def orientation_state(self):
        "Redundant of quaternion state"
        return self.quaternion_state

    @property
    def orientation_mat_state(self):
        return quat.as_rotation_matrix(self.quaternion_state)

    @property
    def transformation_state(self):
        orient = self.orientation_mat_state
        transf = np.eye(4)
        transf[:3, :3] = orient
        transf[:3, -1] = self.offset
        return transf

    @property
    def transf_state(self):
        "Redundand property"
        return self.transformation_state

    @property
    def quaternion_state(self):
        angle = self._angle - self.null
        qz = get_quat_normalised(angle, self.rotation_vec)
        Q = self._quaternion * qz
        #Q = qz * self.quaternion
        return Q

class Model3D:
    def __init__(self, anchor=None):
        self._anchor = anchor if anchor else (0, 0, 0)
        self._segments = dict()
        self._seg_map = dict()
        self._transf_mats = None
        self._transf_quats = None
        self._transf_need_up = True
        self._ax_directions = dict()
        #self._prev_axis = (None, None)

        self.add_segment(name="anchor")

    def onChangeDec(fun):
        @wraps(fun)
        def wrapper(self, *args, axis=None, up=None, **kwargs):
            ret = fun(self, *args, axis=axis, up=up, **kwargs)
            self._transf_need_up = True
            return ret
        return wrapper

    @onChangeDec
    def add_segment(self, *args, rotation_axis=None, axis=None, **kwargs):
        """
        Return:
            parent_id: integer, key to parent segment
        """
        self._ax_directions[len(self._segments)] = dict.fromkeys(["abs", "axis", "up"], None)
        if axis:
            ret = self._add_relative_segment(*args, axis=axis, **kwargs)
        elif rotation_axis:
            ret = self._add_absolute_segment(*args, rotation_axis=rotation_axis, **kwargs)
        else:
            ret = self._add_absolute_segment(*args, **kwargs)
        return ret

    @property
    def joints(self):
        return tuple(self[num].angle for num in range(len(self._segments)))

    def _add_absolute_segment(self, parent_id=None, axis=None, up=None, rotation_axis=None,
            *args, offset=None, **kwargs):

        #raise ValueError("ASD")
        assert len(args) == 0, "Some Positional arguments are wasted: " + str(args)

        num = len(self._segments)
        #rotation_axis = rotation_axis.lower()
        #self._ax_directions[num]['abs'] = rotation_axis
        #prev = self._ax_directions[parent_id]
        #self.transf_mats
        #quats = self.transf_quats
        #par_orientation = quats[parent_id][0]


        #if rotation_axis == "x":
            #qt = get_quat_normalised(-90, [0, 1, 0])
        #elif rotation_axis == '-x':
            #qt = get_quat_normalised(90, [0, 1, 0])
        #elif rotation_axis == 'y':
            #qt = get_quat_normalised(-90, [1, 0, 0])
        #elif rotation_axis == '-y':
            #qt = get_quat_normalised(90, [1, 0, 0])
        #elif rotation_axis == 'z':
            #qt = quaternion(1, 0,0,0)
        #elif rotation_axis == '-z':
            #qt = get_quat_normalised(180, [1, 0, 0])
        #else:
            #raise ValueError(f"This axis is wrong {rotation_axis}")

        #par_inv = par_orientation.inverse()
        #absqt = par_inv * qt

        #rel_offset = point_fromB(quat.as_rotation_matrix(par_inv), point=offset)
        new_seg = Segment(rotation_axis=rotation_axis, offset=offset, **kwargs)
        #new_seg.quaternion = absqt
        self._add_segment(num, new_seg, parent_id)

        return num

    def _add_relative_segment(self, parent_id=None, axis=None, up=None, **kwargs):
        num = len(self._segments)
        if parent_id:
            parent_id = int(parent_id)
        self._ax_directions[num]['axis'] = axis
        self._ax_directions[num]['up'] = up

        new_segment = Segment(axis=axis, up=up, **kwargs)
        self._add_segment(num, new_segment, parent_id)
        return num

    def _add_segment(self, num, new_segment, parent_id):
        """
        Add segment to class hierarchy
        """
        if parent_id is not None:
            assert parent_id in self._segments, f"This parent is not defined {parent_id}"
        else:
            parent_id = None if num == 0 else 0
        self._segments.update({num: new_segment})
        self._seg_map.update({num: parent_id})
        return num

    def visualise(self, *args, block=True, **kwargs):
        plt.figure(figsize=(16, 9))
        ax = plt.gca(projection="3d")
        self.draw(ax, *args, **kwargs)
        plt.show(block=block)

    def draw(self, ax, block=True, *args, **kwargs):
        ax.set_title("Model PLOT")
        transfDict = self.transf_mats
        small = [1,1,1]
        big = [0,0,0]

        for key, transf in transfDict.items():
            seg = self._segments.get(key)
            small = np.min([small, transf[:3,-1]], axis=0)
            #print(transf[:3, -1])
            big = np.max([big, transf[:3,-1]], axis=0)

            text = f"base {key}" if not seg.name else seg.name
            self._draw_axis(ax, transf, text=text, *args, **kwargs)

        small = small - 0.2
        big = big + 0.2
        ax.set_xlim([small[0], big[0]])
        ax.set_ylim([small[1], big[1]])
        ax.set_zlim([small[2], big[2]])
        ax.view_init(15, 110)

    def __getitem__(self, key):
        self._transf_need_up = True
        return self._segments.get(key, None)

    @staticmethod
    def _draw_axis(ax, transf, ax_size=1, line_width=3, text=None, textsize=15, weight=800):
        "Draw axis for univeral styling"
        points_pairs = np.array([
            [0, 0, 0, 1], [ax_size, 0, 0, 1],
            [0, 0, 0, 1], [0, ax_size, 0, 1],
            [0, 0, 0, 1], [0, 0, ax_size, 1],
            [0.5*ax_size, 0.5*ax_size, 0.5*ax_size, 1]
        ]).T
        abs_points = np.dot(transf, points_pairs)
        Cols = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for ind in range(0, 6, 2):
            roi = abs_points[:3, ind:ind+2]
            ax.plot(*roi, c=Cols[ind//2], linewidth=line_width)

        if text:
            ax.text(*abs_points[:3, -1], text,
                    fontdict={"size": textsize, "weight": weight}
                    )

    def calculate_transformations(self, inquaternions=True):
        "Calculatre transformations to current state"
        if self._transf_need_up:
            transfDict = dict()

            "Create values inside dict"
            for key, parent in self._seg_map.items():
                transf = transfDict.get(key, None)
                if transf is None:
                    self._save_qt_to_dict(transfDict, key)

            self._transf_quats = transfDict.copy()

            "Create matrices"
            transf_mats = dict()
            for key, (qt, offset) in transfDict.items():
                trmat = np.eye(4)
                trmat[:3, :3] = quat.as_rotation_matrix(qt)

                #print(offset, type(offset))
                trmat[:3, -1] = offset
                transf_mats.update({key: trmat})

            self._transf_mats = transf_mats.copy()
            self._transf_need_up = False
        return self._transf_mats.copy()

    def _save_qt_to_dict(self, transfDict, num=0):
        """
        Get single transformation from absolute parent transf
        """
        parent_num = self._seg_map.get(num)
        if parent_num is not None:
            parent = transfDict.get(parent_num, None)
            if parent is None:
                parent = self._save_qt_to_dict(transfDict, parent_num)

            seg = self[num]
            par_qt, par_off = parent
            qt_abs = par_qt * seg.orientation_state

            par_transf = quat.as_rotation_matrix(par_qt)
            abs_transf = quat.as_rotation_matrix(qt_abs)

            abs_offset = point_fromB(par_transf, par_off, seg.offset)

            pair = (qt_abs, abs_offset)
            transfDict.update({num: pair})
            return pair

        else:
            "Parent is None, so this is root!"
            seg = self._segments.get(num)
            qt = seg.orientation_state
            offset = seg.offset
            pair = (qt, offset)
            transfDict.update({num: pair})
            return pair

    @property
    def transf_mats(self):
        self.calculate_transformations()
        return deepcopy(self._transf_mats)

    @property
    def transf_quats(self):
        self.calculate_transformations()
        return deepcopy(self._transf_quats)

def create_robot():
    mod = Model3D()

    val = mod.add_segment(name="J1", rotation_axis = "-z",  offset=[0,0,0.345])
    val = mod.add_segment(val, name="J2", rotation_axis = "y",  offset=[0.02,0,0])
    val = mod.add_segment(val, name="J3", rotation_axis = "y",  offset=[0.26,0,0])
    val = mod.add_segment(val, name="J4", rotation_axis ="-x",  offset=[0,0,0.02])
    val = mod.add_segment(val, name="J5", rotation_axis = "y",  offset=[0.26,0,0])
    val = mod.add_segment(val, name="J6", rotation_axis= "-x",  offset=[0.075,0,0])
    return mod

def animate(i):
    #print(i)
    if i == 0:
        time.sleep(1)
    interval = 80
    amplitude = 10
    global phase

    cycle = 40
    step = -(360) / cycle
    traillen = 2 * cycle

    if not ((i+1) % cycle):
        phase = (phase + 1) % 6

    #robot[1].angle = i*4
    #robot[2].angle = -i/2
    #robot[3].angle = i*2
    #robot[4].angle = i*1.6
    #robot[5].angle = i*7
    #robot[6].angle = i
    if phase == 0:
        robot[1].angle += step
    elif phase == 1:
        robot[2].angle += step# * 0.5
    elif phase == 2:
        robot[3].angle += step# * 2
    elif phase == 3:
        robot[4].angle += step# * 3
    elif phase == 4:
        robot[5].angle += step# * 5
    else:
        #robot[1].angle += -90 + step
        #phase = 1
        robot[6].angle += step

    ax.clear()
    trf = robot.transf_mats
    robot.draw(ax, ax_size=0.08, textsize=12)
    end_trf = robot.transf_mats[6]
    orien = end_trf[:3, :3]
    offset = end_trf[:3, -1]
    end_point = point_fromB(orien, offset=offset, point=[0,0,0])


    #for t in trf.values():
        #print(t)
    #print(end_point)
    trail.append(end_point)
    pts = np.array(trail[-traillen:]).T
    cols = np.clip(np.absolute(pts).T*2+[0,0,-0.3], 0, 1)
    ax.scatter(pts[0, :], pts[1, :], pts[2, :], c=cols)

    ed = 0.6
    ax.set_xlim([-ed, ed])
    ax.set_ylim([-ed, ed])
    ax.set_zlim([0, ed])

if __name__ == "__main__":
    "DRAW some axis"
    robot = create_robot()
    phase = 0
    #robot[1].angle = -50
    #robot[2].angle=20
    fig = plt.figure(figsize=(10,7))
    #ax = plt.gca(projection="3d")
    ax = Axes3D(fig)
    trail = []


    robot.draw(ax)
    ani = FuncAnimation(fig, animate, interval=10)
    plt.show()

    ed = 2

