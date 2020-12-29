from mpl_toolkits.mplot3d import Axes3D
import quaternion as quat
from quaternion import quaternion

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


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
        self.inverse = None
        self._quaternion = None

        "Linear move"
        self.pos = pos
        "Joint Move"
        self.angle = angle
        self.null = null

        if offset:
            self.offset = np.array(offset)
        else:
            self.offset = np.array([0,0,0])

        self.set_transformation(*args, **kwargs)

    def set_transformation(self, axis=None, up=None, offset=None):
        """
        Define transformation
        """
        qt = self.get_qt_from_ax(axis, up)
        self._quaternion = qt

    @staticmethod
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
                q1 = self.get_quat(180, [0,0,1])
            else:
                q1 = quaternion(1, 0,0,0)

            if up == "y":
                q2 = self.get_quat(90, [1,0,0])
            elif up == "-y":
                q2 = self.get_quat(-90, [1,0,0])
            elif up == "-z":
                q2 = self.get_quat(180, [1,0,0])
            else:
                q2 = quaternion(1, 0,0,0)

            Q = q1 * q2
            return Q

        elif axis == "y" or axis == "-y":
            if up == "x":
                q2 = self.get_quat(-90, [0,1,0])
            elif up == "-x":
                q2 = self.get_quat(90, [0,1,0])
            elif up == "-z":
                q2 = self.get_quat(180, [0,1,0])
            else:
                q2 = quaternion(1, 0,0,0)

            if axis == "y":
                q1 = self.get_quat(-90, [0,0,1])
            else:
                q1 = self.get_quat(90, [0,0,1])
            Q = q1 * q2
            return Q

        else:
            if up is not None:
                if up == "y":
                    q1 = self.get_quat(90, [1,0,0])
                    angle = 90
                elif up == "-y":
                    q1 = self.get_quat(-90, [1,0,0])
                    angle = -90
                elif up == "-x":
                    q1 = self.get_quat(90, [0,1,0])
                    angle = 0
                else:
                    q1 = self.get_quat(-90, [0,1,0])
                    angle = 180

                if axis == "z":
                    q2 = self.get_quat(angle, [0,0,1])
                else:
                    angle = 180 + angle
                    q2 = self.get_quat(angle, [0,0,1])

                Q = q2 * q1
            else:
                if axis == "z":
                    Q = self.get_quat(90, [0,1,0])
                else:
                    Q = self.get_quat(-90, [0,1,0])
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
        print("axis", axis)
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

        qt = self.get_quat(deg, ax_vect)

        return qt

    def add_rotation(self, *args, **kwargs):
        """
        Applies rotation to model. Specify rotation in format `get_rotation`
        ARGS:
            deg - value in degrees
            axis - xyz letter or size3 List / Array
        """
        qt = self.get_rotation(*args, **kwargs)
        self._quaternion = qt * self._quaternion
        return qt

    @staticmethod
    def get_rotation_matrix(axis, rad: "Angle in Radians"):
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
    def transf(self):
        return self.transformation

    @property
    def orientation(self):
        return quat.as_rotation_matrix(self._quaternion)

    @property
    def orientation_state(self):
        angle = self.angle - self.null
        qz = self.get_quat(angle, [0,0,1])
        Q = self._quaternion * qz
        return quat.as_rotation_matrix(Q)

    ##@property
    #def get_inverse(self):
        #orient = self.orientation
        #transf = np.eye(4)
        #transf[:3, :3] = orient.T
        #transf[:3, -1] = self.offset
        #transf[:3, 3] = -np.dot(transf[:3, :3], transf[:3, 3])
        #return transf

    @property
    def transformation(self):
        orient = self.orientation
        transf = np.eye(4)
        transf[:3, :3] = orient
        transf[:3, -1] = self.offset
        return transf

    @property
    def transformation_state(self):
        orient = self.orientation_state
        transf = np.eye(4)
        transf[:3, :3] = orient
        transf[:3, -1] = self.offset
        return transf

    def get_transformation(self):
        return self.transformation

    @property
    def endFrame(self):
        return self.transformation

    def get_point_fromA(self, point, apply_translation=True):
        """
        Convert points in origin frame to relative
        """
        assert len(point) >= 3, "Point has to have at least 3 coords"
        point = np.array(point) - self.offset
        orien = self.orientation_state.T
        return np.dot(orien, point)

    def get_point_fromB(self, point, apply_translation=True):
        """
        Convert points in relative axis to parent
        """
        assert len(point) >= 3, "Point has to have at least 3 coords"
        point = np.array(point)
        pointinA = np.dot(self.orientation_state, point)
        pointinA += self.offset
        return pointinA

    def visualise(self, points=None, block=True, prec=3, show_back=True):
        """
        Render origin frames and relative frame.
        Scatter 1,1,1 point on each frame.
        Show point values in both frames.
        """
        fig = plt.figure()
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

        shadow = np.dot(self.transformation_state, new_axis)
        new_axis[3, :] = 1
        absolut = np.dot(self.transformation_state, new_axis)

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
    def __init__(self, name=None, *args,
            sgtype="joint", null=0, angle=0,
            **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_low = 0
        self._limit_up = math.pi
        self.name = name

        if sgtype != "joint":
            raise NotImplementedError(
                "Only joint segments supported in current state")


class Model3D:
    def __init__(self, anchor=None):
        self._anchor = anchor if anchor else (0, 0, 0)
        self._segments = dict()
        self._seg_map = dict()

    def add_segment(self, parent_id=None, *args, **kwargs):
        num = len(self._segments)
        if parent_id:
            parent_id = int(parent_id)

        new_segment = Segment(*args, **kwargs)

        self._segments.update({num: new_segment})
        self._seg_map.update({num: parent_id})
        return num

    def draw(self, block=True, *args, **kwargs):
        plt.figure(figsize=(16, 9))
        ax = plt.gca(projection="3d")
        transfDict = self.get_transformations()
        for key, transf in transfDict.items():
            seg = self._segments.get(key)
            text = f"base {key}" if not seg.name else seg.name
            self._draw_axis(ax, transf, text=text, *args, **kwargs)

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 5])
        ax.view_init(15, 110)
        plt.show(block=block)

    @staticmethod
    def _draw_axis(ax, transf, ax_size=1, line_width=3, text=None):
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
                    fontdict={"size": 15, "weight": 800}
                    )

    def get_transformations(self):
        transfDict = dict()
        for key, parent in self._seg_map.items():
            transf = transfDict.get(key, None)
            if transf is None:
                self._get_single_transf(transfDict, key)
        return transfDict

    def _get_single_transf(self, transfDict, num):
        parent = self._seg_map.get(num)
        if parent is not None:
            par_transf = transfDict.get(parent, None)
            if par_transf is None:
                par_transf = self._get_single_transf(parent)

            rel_transf = self._segments.get(num).transformation
            rel_transf = 1 if rel_transf is None else rel_transf

            transf = np.dot(par_transf, rel_transf)
            transfDict.update({num: transf})
            return transf

        else:
            transf = self._segments.get(num).transformation
            if transf is None:
                transf = 1
            transfDict.update({num: transf})
            return transf


def create_robot():
    mod = Model3D()
    #for x in range(8):
        #seg_num = mod.add_segment(
            #seg_num,
            #rotation="x", angle=math.radians(20),
            #translation=[1, 1, 0])

    val = mod.add_segment(name="anchor")
    val = mod.add_segment(val, name="J1", up="-z", offset=[0,0,2])
    rot = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]]
    "2"
    val = mod.add_segment(val, name="J2", axis="y", up="-x", offset=[2,0,0])

    "3"
    val = mod.add_segment(val, name="J3", axis="-x", offset=[3,0,0])

    "4"
    val = mod.add_segment(val, name="J4", axis="x", up="-y", offset=[0,-2,0])

    "5"
    val = mod.add_segment(val, name="J5", axis="x", up="y")  # 5

    "6"
    val = mod.add_segment(val, name="J6", axis="x", up="-y", offset=[0,-1,0])

    #transfDict = mod.get_transformations()
    mod.draw(line_width=3)


if __name__ == "__main__":
    "DRAW some axis"
    #create_robot()
    ed = 2

    line = [[x, -1, 0.5] for x in np.linspace(0,4,9)]
    print(line)
    points = [
        *line,
        #[ 1, 2,-1],
        #[ 1, 0,0],
        #[ 0, 1,0],
        #[ 0, 0,1],
        #[ 1, 1,0],
        #[-1,-1,0],
        #[-1, 0,0],
        #[ 0,-1,0],
        [1,2,3],
        [-ed,-ed, -ed],
        [ ed, ed, ed],
    ]
    mod = Model3D()

    #cord = RelativeCoordinate(axis="y", angle=45, offset=[2,0,0])
    cord = RelativeCoordinate()
    cord.add_rotation(30, "z")
    cord.add_rotation(85, "x")
    #qt = cord.add_rotation(90, "y")
    #qt = cord.add_rotation(45, "z")
    #print(qt)
    #print(qt.absolute())

    #cord.visualise()
    cord.visualise(points, show_back=False)

