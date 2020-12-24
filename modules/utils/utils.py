import numpy as np
import math
import time
import matplotlib.pyplot as plt

from itertools import product, cycle


from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import pygame


def render_axis(render_abs_axis=True, perspective=False):
    glMatrixMode(GL_MODELVIEW)
    
    "Params"
    glLineWidth(3.0)
    ax_len = 1

    if render_abs_axis:
        offset = 0,0,0
        glBegin(GL_LINES)
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

    model = glGetDoublev(GL_MODELVIEW_MATRIX).T
    rot = model[:3, :3]
    scrX = np.dot(rot, [1,0,0])
    scrY = np.dot(rot, [0,1,0])
    scrZ = np.dot(rot, [0,0,1])

    if perspective:
        "Render small axis in corner"
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        "Perspective Axis Darker"
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        offset = 4, -2, -8
        
        glBegin(GL_LINES)
        "X RED"
        glColor(0.7,0,0)
        glVertex3f(*offset)
        glVertex3f(*(scrX+offset))
        "Y GREEN"
        glColor(0,.7,0)
        glVertex3f(*offset)
        glVertex3f(*(scrY+offset))
        "Z Blue"
        glColor(0,0,0.7)
        glVertex3f(*offset)
        glVertex3f(*(scrZ+offset))

        glEnd()
        glPopMatrix()

    else:
        "Orthographic"
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-5, 5, -3, 3, 0.1, 50)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        offset = 4, -2,-2
        
        glBegin(GL_LINES)
        "X RED"
        glColor(1,0,0)
        glVertex3f(*offset)
        glVertex3f(*(scrX+offset))
        "Y GREEN"
        glColor(0,1,0)
        glVertex3f(*offset)
        glVertex3f(*(scrY+offset))
        "Z Blue"
        glColor(0,0,1)
        glVertex3f(*offset)
        glVertex3f(*(scrZ+offset))

        glEnd()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
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
