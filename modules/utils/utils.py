import numpy as np
import math
import time
import matplotlib.pyplot as plt

from itertools import product, cycle


from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import pygame


def render_axis(offset=None):
    glPushMatrix()
    #glMatrixMode(GL_MODELVIEW)
    #model = glGetDoublev( GL_MODELVIEW_MATRIX).T

    "Params"
    glLineWidth(3.0)
    ax_len = 10
    glBegin(GL_LINES)

    "OFFSET"
    if offset is None:
        offset = 0,0,0
    xoff, yoff, zoff = offset
    if True:
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
    else:
        "Mini model try"
        st = model[(0,2,1), -1] + [0, -20, 0]

        "X RED"
        glColor(1,0,0)
        end = st + [10, 0, 0]
        glVertex3f(*st)
        glVertex3f(*end)

        "Y GREEN"
        glColor(0,1,0)
        end = st + [0, 10, 0]
        glVertex3f(*st)
        glVertex3f(*end)

        "Z Blue"
        glColor(0,0,1)
        end = st + [0, 0, 10]
        glVertex3f(*st)
        glVertex3f(*end)

    glEnd()
    glPopMatrix()


def render_plane(verts):
    glPushMatrix()
    #glMatrixMode(GL_MODELVIEW)
    
    glLineWidth(1)
    glBegin(GL_LINES)
    glColor(0.8,0.8,0.8)

    for vert in verts:
        glVertex3f(*vert[0])
        glVertex3f(*vert[1])

    glEnd()
    glPopMatrix()
