import pygame
from pygame.locals import *

import numpy as np
import math
import time
import os

from itertools import product, cycle
from OpenGL.GL import *
from OpenGL.GLU import *


class BaseModel:
    def __init__(self):
        self.pos = np.zeros((3, 1))
        #self.size = 1
        self.vert, self.edges = self.get_initial_shape()
        self.original = self.vert.copy()
        self.face_color = (1, 1, 1)
        self.edge_color = (0, 0, 0)

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
        """Empty abstract method"""
        print("BASE SHAPE")
        return None, None

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
        self.pos += np.array(vector)

    def draw(self):
        edges = self.edge
        vertices = self.vert.copy()
        vertices = vertices + self.pos[:, np.newaxis]

        # glBegin(GL_LINES)
        # glColor3fv(self.edge_color)
        # for edge in edges:
        #v1,v2 = edge
        #vert1 = vertices[:, v1]
        #vert2 = vertices[:, v2]
#
        # glVertex3fv(vert1)
        # glVertex3fv(vert2)
        # glEnd()

        glBegin(GL_QUADS)
        glColor3fv(self.face_color)
        face = [vertices[:, 1], vertices[:, 0], vertices[:, 2], vertices[:, 3]]
        for vert in face:
            glVertex3fv(vert)
        glEnd()


class Cube(BaseModel):
    def __init__(self, size=1, pos=None, rot=None):
        super().__init__()

        self.original, self.edge = self.get_initial_shape(size)
        if pos:
            self.pos = np.array(pos)
        print(self.original)
        self.vert = self.original.copy()

    @staticmethod
    def get_initial_shape(scale=1):
        """
        Create basic cube of dimension 1, if no scaler is passed
        Returns 2d Array of vertices and edges
        """
        print("CUBE SHAPE, size:", scale)
        scale = scale / 2
        vertices = np.array([*product([-1, 1], repeat=3)])
        edges = [(ind, ind2)
                 for ind, vertx in enumerate(vertices)
                 for ind2 in np.where(
            np.absolute((vertices - vertx)).sum(axis=1) == 2)[0]
            if ind2 > ind]

        vertices = vertices*scale
        vertices = vertices.T
        print("vert generated:", vertices)
        return vertices, edges


def lambder(fun, *args, **kwargs):
    return lambda: fun(*args, **kwargs)


class Animator:
    def __init__(self):
        self.actors = [Cube(size=0.5) for x in range(88)]
        print("first:")
        print(self.actors[0].vert)
        self.circle(3)
        plan = [
            (lambder(self.circle, speed=0.4), 20)
        ]
        self.plan = cycle(plan)

    def circle(self, radius=3, x=0, y=0, speed=1):
        N = len(self.actors) + 1
        offset = time.time()*speed
        alfa = (np.linspace(0, 2*np.pi, N) + offset) % (2*np.pi)

        for act, ang in zip(self.actors, alfa):
            act.pos = np.array(
                [math.cos(ang)*radius*1.5*math.sin(offset) + math.sin(ang*2),
                 math.sin(ang)*1.5*radius*math.sin(offset/4),
                 0
                 ])
            act.set_rotation(
                [0, -math.degrees(ang*4)*0.5, math.degrees(ang)*2])
            act.face_color = (
                math.sin(ang) * 0.4 + 0.6,
                math.cos(ang*6) * 0.4 + 0.6,
                math.sin(offset*5) * 0.3 + 0.5
            )

    def step(self):
        fun, time = next(self.plan)
        fun()

    def drawAll(self):
        [cub.draw() for cub in self.actors]


def main():
    pygame.init()
    display = (1200, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0, -12)
    glRotatef(0, 0, 0, 0)

    animator = Animator()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBegin(GL_POINTS)
        glVertex3fv([0, 0, 0])
        glEnd()

        animator.step()
        animator.drawAll()

        pygame.display.flip()
        pygame.time.wait(5)


if __name__ == '__main__':
    main()
