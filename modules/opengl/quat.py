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
    #glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    #viewPort = glGetIntegerv(GL_VIEWPORT)  # [0, 0, width, hegith]
    model = glGetDoublev( GL_MODELVIEW_MATRIX).T
    #proj = glGetDoublev( GL_PROJECTION_MATRIX)
    print("mod:\n", model)


    #gluLookAt(10, 10, 0, 0, 0, 0, 0, 0, 1)

    "Params"
    glLineWidth(3.0)
    ax_len = 10

    glBegin(GL_LINES)
    "OFFSET"
    xoff, yoff, zoff = -3, 0, -5
    if False:
        "X RED"
        glColor(1,0,0)
        glVertex3f(xoff, yoff, zoff)
        glVertex3f(xoff+ax_len, yoff, zoff)
        "Y GREEN"
        glColor(0,1,0)
        glVertex3f(xoff, yoff, zoff)
        glVertex3f(xoff, yoff, zoff+ax_len)
        "Z Blue"
        glColor(0,0,1)
        glVertex3f(xoff, yoff, zoff)
        glVertex3f(xoff, yoff+ax_len, zoff)
    else:
        st = model[(0,2,1), -1] + [0, -20, 0]
        print("start")
        print(st)

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
    #glPopMatrix()


def main():
    pygame.init()

    display = (1200, 700)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
    
    #glRotatef(90, -1, 0, 0)
    #glTranslatef(0, -20, 0)
    gluLookAt(0, 20, 0, 0,0,0, 0,0,1)
    #glLineWidth(3.0)
    glEnable(GL_DEPTH_TEST)

    sh1 = Tube(size=2, precision=15)
    sh1.translate([-15, 0, 0])
    c2 = Cube(size=5)
    c3 = Cube(size=5)

    sh1.edge_color = (1,1,1)

    t1 = Tube(size=50, radius=0.5, precision=15)
    t2 = Tube(size=40, radius=1, precision=15)
    t3 = Tube(size=30, radius=2, precision=20)
    t4 = Tube(size=20, radius=4, precision=40)
    t5 = Tube(size=10, radius=6, precision=10)

    #t1.translate([-8, 8, 0])
    #t2.translate([-7, 7, 0])
    #t3.translate([-6, 6, 0])
    #t4.translate([-4, 4, 0])
    #t5.translate([0, 0, 0])

    fr = 2 
    end_it = False
    face_colors = [
        (0, 0.6, 1),
        (0.4, 0, 1),
        (0.3, 0.8, 0.7),
        (1,   0.1,   0.7),
        (1,   0.7,   0.1),
        (0.6,   0.3,   0.3),
    ]
    #face_colors = list(np.random.random((35,3)))
    pause = False
    while True and not end_it:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_axis()

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
                    glTranslate(0, 5, 0)
                    #sh1.rotate([ang, 0, 0])
                elif event.key == pygame.K_DOWN:
                    glTranslate(0, -5, 0)
                    #sh1.rotate([-ang, 0, 0])
                elif event.key == pygame.K_LEFT:
                    glTranslate(-5, 0, 0)
                    #sh1.rotate([0, -ang, 0])
                elif event.key == pygame.K_RIGHT:
                    glTranslate(5, 0, 0)
                    #sh1.rotate([0, ang, 0])
                    #sh1.rotate([0, 0, ang])
                #elif event.key == pygame.K_COMMA:
                    #sh1.rotate([0, 0, -ang])

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
            pygame.time.wait(50)
            continue

        c2.rotate([0.4, 0.1, 0.7])
        c3.rotate([0.4, 2, 0.1])
        sym = (t1, t2, t3, t4, t5)
        for ind, _t in enumerate(sym):
            _t.pos = (math.sin(time.time()/(ind+1))*ind, -math.sin(time.time()/(ind+1))*ind, 0)
            _t.set_rotation([0, 0, (time.time()*6*(5+ind))])
            _t.rotate([0, 90, -45])
            #_t.rotate([0, 0, time.time()*30])

        #sh1.draw(face_colors=face_colors)
        #c2.draw(face_colors=face_colors)
        #c3.draw(face_colors=face_colors)
        
        t1.draw(face_colors=face_colors)
        t2.draw(face_colors=face_colors)
        #t3.draw(face_colors=face_colors)
        #t4.draw(face_colors=face_colors)
        #t5.draw(face_colors=face_colors)

        #"Draw texture"
        #ttx = get_texture_from_string("Hello")
        #glBegin(GL_QUADS)
        #glColor3fv((0,0.5,1))
        #glVertex3fv([-fr, -fr, 0])   
        #glVertex3fv([ fr, -fr, 0])   
        #glVertex3fv([ fr,  fr, 0])   
        #glVertex3fv([-fr,  fr, 0])   
        #glBindTexture(GL_TEXTURE_2D, ttx)
        #glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()



main()

