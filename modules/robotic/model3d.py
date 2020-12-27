from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


class RelativeCoordinate:
    def __init__(self, transformation=None, rotation_mat=None,
                 rotation=None, angle=None, translation=None,):
        """
        Define new coord in relation to root.
        Otherwise define A in relation to B and use `switch`, which is less intuitive.
        A stands for root axis,
        B stands for relative frame of interests

        To change transformation matrix with inverse matrix call `switch`.
        Show axis with `visualise` method.

        Description:
        Sets Transformation from given objects, only first definition is set
        Args:
            Transformation: 4x4 Matrix contatingn 3x3 roation and 3x1 Translation
            Rotation: 3x3 Matrix
            Translation: Matrix/List/Tuple of size 3
        """
        self.transformation = None
        self.inverse = None

        self.set_transformation(transformation=transformation,
                                rotation_mat=rotation_mat, angle=angle,
                                rotation=rotation,
                                translation=translation
                                )
        self.set_inverse()
        self.inverse_old = False

    def set_transformation(self,
                           transformation=None, rotation_mat=None, translation=None,
                           rotation=None, angle=None):
        """
        Define transformation
        """
        self.transformation = None
        self.inverse_old = False

        if transformation is not None:
            assert transformation.shape == (
                4, 4), "Transformations has to be 4x4 matrix"
            self.transformation = np.array(transformation)
            self.inverse_old = True
            return 

        trans = np.eye(4)
        if rotation is not None or rotation_mat is not None:
            assert not ((rotation_mat is not None) and (rotation is not None)), \
                "Only onre rotation can be provided, along axis or whole matrix"

            if rotation is not None:
                assert rotation is not None and angle is not None, \
                    "Please specify angle for rotation"
                rotation_mat = self.get_rotation_matrix(rotation, angle)
            trans[:3, :3] = rotation_mat
            self.transformation = trans 

        if translation is not None:
            trans[:3, 3] = list(translation)
            self.transformation = trans

    def rotate(self, x=None, y=None, z=None, translation=False):
        if x and y or x and z or y and z:
            print("Rotation is not commutative! XYZ rotating")
        raise NotimplementedError
        self.inverse_old = True

    @staticmethod
    def get_rotation_matrix(axis, rad: "Angle in Radians"):
        axis = axis.lower()
        if axis == 'x':
            rot = [
                [1, 0, 0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad), math.cos(rad)],
            ]
        elif axis == 'y':
            rot = [
                [math.cos(rad), 0, math.sin(rad)],
                [0, 1, 0],
                [-math.sin(rad), 0, math.cos(rad)],
            ]
        else:
            rot = [
                [math.cos(rad), -math.sin(rad), 0],
                [math.sin(rad), math.cos(rad), 0],
                [0, 0, 1],
            ]
        return rot

    def rotateEuler(self, x=None, y=None, z=None):
        raise NotImplementedError

    def set_inverse(self):
        if self.transformation is not None:
            new_inverse = self.transformation.copy()
            new_inverse[:3, :3] = new_inverse[:3, :3].T
            new_inverse[:3, 3] = - \
                np.dot(new_inverse[:3, :3], new_inverse[:3, 3])
            self.inverse = new_inverse
            self.inverse_old = False

    def get_inverse(self):
        if self.inverse_old is not None:
            self.set_inverse()
        return self.inverse

    def switch(self):
        "Switch inverse matrix with transformation"
        temp = self.transformation
        self.transformation = self.inverse
        self.inverse = temp

    def get_transformation(self):
        return self.transformation

    @property
    def endFrame(self):
        if self.transformation is not None:
            return self.transformation.copy()
        else:
            return None

    def get_point_fromA(self, point, apply_translation=True):
        transl = 1 if apply_translation else 0
        if self.inverse is not None:
            assert len(point) >= 3, "Point has to have at least 3 coords"
            point = (point[0], point[1], point[2], transl)
            return np.dot(self.inverse, point)
        else:
            return point

    def get_point_fromB(self, point, apply_translation=True):
        transl = 1 if apply_translation else 0
        if self.inverse is not None:
            assert len(point) >= 3, "Point has to have at least 3 coords"
            point = (point[0], point[1], point[2], transl)
            return np.dot(self.transformation, point)
        else:
            return point

    def visualise(self, points=None, block=True, prec=3):
        """
        Render origin frames and relative.
        Scatter 1,1,1 point on each frame.
        """
        fig = plt.figure()
        #ax = plt.gca(projection="3d")
        ax = Axes3D(fig)
        dashes = [0.8, 0.5]
        axis_width = 5
        relative_intense = 0.7
        shadow_width = 3

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

        if self.transformation is not None:
            shadow = np.dot(self.transformation, new_axis)
            new_axis[3, :] = 1
            absolut = np.dot(self.transformation, new_axis)

            a111_asb = np.dot(self.get_inverse(), [1, 1, 1, 1])
            b111_asa = np.dot(self.transformation, [1, 1, 1, 1])

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
                # np.random.seed(np.absolute(point).sum().astype("int"))
                col = np.random.random(3)*0.6

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
                        fontdict={"size": 9, "color": col, "weight": 800},
                        )
            stack = np.concatenate(
                [shadow[:3, :], absolut[:3, :], draw_points.T], axis=1)

            margin = 0.5
            ax_min = np.min(stack, axis=1)
            ax_max = np.max(stack, axis=1)
            ax.set_xlim([ax_min[0]-margin, ax_max[0]+margin])
            ax.set_ylim([ax_min[1]-margin, ax_max[1]+margin])
            ax.set_zlim([ax_min[2]-margin, ax_max[2]+margin])

        else:
            ax.plot([0, 1], [0, 0], [0, 0], c=(relative_intense, 0, 0),
                    label="frame X", dashes=dashes, linewidth=axis_width)
            ax.plot([0, 0], [0, 1], [0, 0], c=(0, relative_intense, 0),
                    label="frame Y", dashes=dashes, linewidth=axis_width)
            ax.plot([0, 0], [0, 0], [0, 1], c=(0, 0, relative_intense),
                    label="frame Z", dashes=dashes, linewidth=axis_width)
            ax.scatter(*[1, 1, 1])
            ax.text(*[1, 1, 1], "Point 1,1,1")

        plt.title("Relative axis")
        plt.legend(loc=2)
        #ax.set_axis([-5, 5, -5, 5, -5, 5])
        #ax.ylim([-5, 10])
        plt.show(block=block)
        plt.close()


class Segment(RelativeCoordinate):
    def __init__(self, name=None, *args, sgtype="joint", **kwargs):
        super().__init__(*args, **kwargs)
        #print(args)
        #print(kwargs)
        self._limit_low = 0
        self._limit_up = math.pi
        self._pos = 0
        self._zerostate = 0
        self.name = name
        if sgtype != "joint":
            raise NotImplementedError(
                "Only joint segments supported in current state")


class Model3D:
    def __init__(self, anchor=None):
        self._anchor = anchor if anchor else (0, 0, 0)
        self._segments = dict()
        self._seg_map = dict()

    def add_segment(self, parent_id=None, name=None, sgtype="joint",
                    transformation=None, translation=None, 
                    rotation_mat=None, rotation=None, angle=None):

        num = len(self._segments)
        if parent_id:
            parent_id = int(parent_id)

        new_segment = Segment(
            name=name,
            sgtype=sgtype,
            translation=translation,
            transformation=transformation,
            rotation_mat=rotation_mat,
            rotation=rotation,
            angle=angle,
        )

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
    val = mod.add_segment(val, name="J1", translation=[0,0,2])
    rot = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]]
    "2"
    val = mod.add_segment(val, name="J2", rotation_mat=rot, translation=[2,0,0])
    "3"
    val = mod.add_segment(val, name="J3", rotation="z", 
        angle=math.pi, translation=[3,0,0])  # 3
    "4"
    val = mod.add_segment(val, name="J4", 
        rotation="x", angle=-math.pi/2, translation=[0,-2,0])  # 4
    "5"
    val = mod.add_segment(val, name="J5", 
        rotation="x", angle=math.pi/2)  # 5
    "6"
    val = mod.add_segment(val, name="J6", 
        rotation="x", angle=-math.pi/2, translation=[0,-1,0])  # 6

    #transfDict = mod.get_transformations()
    mod.draw(line_width=3)


if __name__ == "__main__":
    "DRAW some axis"
    create_robot()
