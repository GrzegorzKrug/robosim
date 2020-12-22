import numpy as np
import math
import time
import matplotlib.pyplot as plt

from itertools import product, cycle

from models import Cube, Tube

from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pygame



def get_texture_from_string(text):
    font = pygame.font.Font(None, 64)
    textSurface = font.render(text, True, (255,255,255,255),
                  (0,0,0,255))
    ix, iy = textSurface.get_width(), textSurface.get_height()
    image = pygame.image.tostring(textSurface, "RGBX", True)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    i = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, i)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    return i


def render_axis():
    glPushMatrix()
    #glMatrixMode(GL_MODELVIEW)
    #model = glGetDoublev( GL_MODELVIEW_MATRIX).T

    "Params"
    glLineWidth(3.0)
    ax_len = 10
    glBegin(GL_LINES)

    "OFFSET"
    xoff, yoff, zoff = 0, 0, 0
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


def main():
    pygame.init()

    display = (1200, 700)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
    glEnable(GL_DEPTH_TEST)
    
    #glRotate(90, 1, 0, 0)
    #glTranslate(10, -0, 5)
    gluLookAt(-5,15,10, 0,0,0, 0,0,1)
    N = 20
    pts = 20
    
    plane = []
    for val in np.linspace(-N/2, N/2, pts):
        plane.append([(-N/2, val, 0),(N/2, val, 0)])
        plane.append([(val, -N/2, 0),(val, N/2, 0)])
    #print(plane)

    end_it = False
    pause = False
    while True and not end_it:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_axis()
        render_plane(plane)
        glRotate(0.1, 0,0,1)
        #gluLookAt(-0.1,0.1,0.1, 0,0,0, 0,0,1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end_it = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    end_it = True
                    break
                if event.key == pygame.K_SPACE:
                    pause ^= True

                if event.key == pygame.K_UP:
                    glTranslate(0, 0, 1)
                elif event.key == pygame.K_DOWN:
                    glTranslate(0, 0, -1)
                elif event.key == pygame.K_LEFT:
                    glTranslate(-2, 0, 0)
                elif event.key == pygame.K_RIGHT:
                    glTranslate(2, 0, 0)
                elif event.key == pygame.K_KP_PLUS:
                    glTranslate(0, 1, 0)
                elif event.key == pygame.K_KP_MINUS:
                    glTranslate(0,-1, 0)

                ang = 15 
                if event.key == pygame.K_KP8:
                    glRotatef(15, 0, -1, 0)
                elif event.key == pygame.K_KP2:
                    glRotatef(15, 0, 1, 0)
                elif event.key == pygame.K_KP4:
                    glRotatef(15, 0, 0, -1)
                elif event.key == pygame.K_KP6:
                    glRotatef(15, 0, 0, 1)
                elif event.key == pygame.K_KP7:
                    glRotatef(15, -1, 0, 0)
                elif event.key == pygame.K_KP9:
                    glRotatef(15, 1, 0, 0)

        if pause == True:
            pygame.time.wait(1000)
            continue


        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()



main()

