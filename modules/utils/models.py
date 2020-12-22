import numpy as np
import math

from itertools import product, cycle
from OpenGL.GL import *
from OpenGL.GLU import *


class BaseShape:
    def __init__(self, size=1, radius=1, debug=False, **kwargs):
        self.pos = np.zeros(3)
        #self.size = 1
        self.original, self.edge, self.faces = self.get_initial_shape(size=size, radius=radius, **kwargs)
        self.vert = self.original.copy()

        self.face_color = (0, 0.5, 0.5)
        self.edge_color = (1, 1, 1)
        self.debug = debug
    
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        pos = np.array(new_pos)
        assert pos.shape == (3,), f"New possition has other shape then (3,): {pos.shape}"
        self._pos = pos

    @pos.getter
    def pos(self):
        return self._pos

    def set_rotation(self, angle):
        angle = [math.radians(deg) for deg in angle]
        self.vert = self.original.copy()
        for num, radians in enumerate(angle):

            rot = self.get_rotationMatrix(radians, num)
            vert = np.dot(rot, self.vert)
            self.vert = vert

    def rotate(self, angle):
        angle = [math.radians(deg) for deg in angle]
        for num, radians in enumerate(angle):
            if not radians:
                continue
            rot = self.get_rotationMatrix(radians, num)
            vert = np.dot(rot, self.vert)
            self.vert = vert

    @staticmethod
    def get_initial_shape(scale=1):
        """
        Abstract Method
        Should return:
            vertices - points in 3d space
            edges - relations betwen vertices
        """
        raise NotImplementedError("Define your shape, this is abstract method!")
    
    def get_faces(self):
        raise NotImplementedError("Define your faces, this is abstract method!")
        
    @staticmethod
    def get_rotationMatrix(rad: "Radians", axis):
        if axis == 0:
            rot = [
                [1, 0, 0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad), math.cos(rad)],
            ]
        elif axis == 1:
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
        rot = np.array(rot)
        return rot

    def translate(self, vector):
        self._pos += np.array(vector)

    def draw(self, drawmode="f", face_colors=None):
        """
        drawmode:
            w - wiremode, skelet
            f - faces

        """
        drawmode = drawmode.lower()
        edges = self.edge
        vertices = self.vert.copy()
        vertices = vertices + self.pos[:, np.newaxis]
        if self.debug:
            print(self.pos)
        self.current_vertices = vertices

        if face_colors and not isinstance(face_colors, (list,tuple)):
            face_colors = tuple(face_colors)
        #if face_colors and isinstance(face_colors 
    
        if "w" in drawmode:
            glBegin(GL_LINES)
            glColor3fv(self.edge_color)
            for edge in edges:
                v1,v2 = edge
                vert1 = vertices[:, v1]
                vert2 = vertices[:, v2]

                glVertex3fv(vert1)
                glVertex3fv(vert2)
            glEnd()
        
        if "f" in drawmode:
            glBegin(GL_QUADS)
            if face_colors:
                faceC = cycle(face_colors)
            else:
                faceC = cycle([self.face_color])
           
            for vert in self.get_faces():
                color = next(faceC)
                glColor3f(*color)
                
                [glVertex3fv(cord) for cord in vert.T]
            glEnd()

    def get_faces(self):
        return [self.current_vertices[:, face] for face in self.faces]


class Cube(BaseShape):
    #def __init__(self, size=1, pos=None, rot=None):
        #super().__init__(size=size)

    @staticmethod
    def get_initial_shape(size=1, radius=None):
        """
        Create basic cube of dimension 1, if no scaler is passed
        Returns 2d Array of vertices and edges
        """
        #print("CUBE SHAPE, size:", scale)
        scale = size / 2
        vertices = np.array([*product([-1, 1], repeat=3)])
        edges = [(ind, ind2)
                 for ind, vertx in enumerate(vertices)
                 for ind2 in np.where(
            np.absolute((vertices - vertx)).sum(axis=1) == 2)[0]
            if ind2 > ind]

        vertices = vertices*scale
        vertices = vertices.T
        #print(vertices.shape)
        #print("vert generated:", vertices)
        faces = (
            (1, 0, 2, 3),
            (7, 5, 4, 6),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (0, 4, 6, 2),
            (1, 5, 7, 3),
        )

        return vertices, edges, faces


class Tube(BaseShape):
    @staticmethod
    def get_initial_shape(size=1, radius=1, precision=25):
        assert precision >= 3, "precision has to be at least 3, for 3 faces"

        size = np.array([[0], [0], [size]])
        base = np.linspace(0, math.pi*2, precision+1)
        prev, base = base[0], base[1:]
        
        vertices = np.empty(shape=(3,0))
        edges = []
        faces = []
        for ind, pt in enumerate(base):
            ind *= 4

            vtx = np.array([
                [math.cos(pt)*radius],
                [math.sin(pt)*radius],
                [0],
            ])
            vtx2 = np.array([
                [math.cos(prev)*radius],
                [math.sin(prev)*radius],
                [0],
            ])

            vertices = np.concatenate([vertices, vtx], axis=1)
            vertices = np.concatenate([vertices, vtx2], axis=1)
            vertices = np.concatenate([vertices, vtx2+size], axis=1)
            vertices = np.concatenate([vertices, vtx+size], axis=1)

            edges.append((ind, ind+1))
            edges.append((ind+1, ind+2))
            edges.append((ind+2, ind+3))
            edges.append((ind, ind+3))
            
            faces.append((ind, ind+1, ind+2, ind+3))
            prev = pt
            
        #print("verts tube:\n", vertices)
        return vertices, edges, faces


