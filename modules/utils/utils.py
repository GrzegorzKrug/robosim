import numpy as np
import math
import time
import matplotlib.pyplot as plt

from itertools import product, cycle


from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import pygame


def render_axis(render_abs_axis=True, render_small_axis=True):
    glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    "Params"
    glLineWidth(3.0)
    ax_len = 1
    glBegin(GL_LINES)

    "OFFSET"
    offset = 0,0,0

    if render_abs_axis:
        "Render Absolute axis in origin"
        xoff, yoff, zoff = offset
        "X RED"
        glColor(1,0,0)
        glVertex3f(xoff, yoff, zoff)
        glVertex3f(xoff+ax_len, yoff, zoff)
        "Y GREEN"
        glColor(0,1,0)
        glVertex3f(xoff, yoff, zoff)
        glVertex3f(xoff, yoff+ax_len, zoff)
        "Z Blue"
        glColor(0,0,1)
        glVertex3f(xoff, yoff, zoff)
        glVertex3f(xoff, yoff, zoff+ax_len)
        glEnd()
        glPopMatrix()

    if render_small_axis:
        "Render small axis in corner"
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        model = glGetDoublev(GL_MODELVIEW_MATRIX).T
        x,y,z = model[:3, -1]
        rot = model[:3, :3]

        axX = np.dot(rot, [1,0,0])
        axY = np.dot(rot, [0,1,0])
        axZ = np.dot(rot, [0,0,1])

        glPushMatrix()
        glLoadIdentity()
        glOrtho(-5, 5, -5, 5, 0.1, 50)
        glTranslate(9, -5, -15)

        glBegin(GL_LINES)
        "X RED"
        glColor(1,0,0)
        glVertex3f(0,0,0)
        glVertex3f(*axX)
        "Y GREEN"
        glColor(0,1,0)
        glVertex3f(0,0,0)
        glVertex3f(*axY)
        "Z Blue"
        glColor(0,0,1)
        glVertex3f(0,0,0)
        glVertex3f(*axZ)

        glEnd()
        glPopMatrix()

        glPushMatrix()
        glLoadIdentity()
        glOrtho(-10, 10, -10, 10, 0.1, 50)
        glTranslate(-5.5,-2, 0)
        #glTranslate(-9, -5, -15)

        glBegin(GL_LINES)
        "X RED"
        glColor(1,0,0)
        glVertex3f(0,0,0)
        glVertex3f(*axX)
        "Y GREEN"
        glColor(0,1,0)
        glVertex3f(0,0,0)
        glVertex3f(*axY)
        "Z Blue"
        glColor(0,0,1)
        glVertex3f(0,0,0)
        glVertex3f(*axZ)

        glEnd()
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)

def render_plane(verts):
    glMatrixMode(GL_MODELVIEW)
    
    glLineWidth(1)
    glBegin(GL_LINES)
    glColor(0.8,0.8,0.8)

    for vert in verts:
        glVertex3f(*vert[0])
        glVertex3f(*vert[1])

    glEnd()
    #glPopMatrix()
