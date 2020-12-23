import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os

from itertools import product, cycle

from robotic.model3d import Segment, Model3D
from urdfpy import URDF
from stl import mesh

from utils.models import Cube, Tube
from utils.utils import render_axis, render_plane

from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pygame


def create_robot():
    mod = Model3D()
    #for x in range(8):
        #seg_num = mod.add_segment(
            #seg_num,
            #rotation="x", angle=math.radians(20),
            #translation=[1, 1, 0])
    
    val = mod.add_segment(name="anchor")
    val = mod.add_segment(val, name="J1", translation=[0,0,0.345])
    rot = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]]
    "2"
    val = mod.add_segment(val, name="J2", rotation_mat=rot, translation=[0.02,0,0])
    "3"
    val = mod.add_segment(val, name="J3", rotation="z", 
        angle=math.pi, translation=[0.26,0,0])  # 3
    "4"
    val = mod.add_segment(val, name="J4", 
        rotation="x", angle=-math.pi/2, translation=[0,-0.02,0])  # 4
    "5"
    val = mod.add_segment(val, name="J5", 
        rotation="x", angle=math.pi/2, translation=[0.26, 0, 0])  # 5
    "6"
    val = mod.add_segment(val, name="J6", 
        rotation="x", angle=-math.pi/2, translation=[0,-0.26,0])  # 6
    #mod.draw(ax_size=0.4)
    return mod


def main():
    pygame.init()

    display = (1200, 700)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    robot_dir = os.path.join("kuka_experimental", "kuka_kr3_support")

    #collision = os.path.abspath(
        #os.path.join(robot_dir, "meshes", "kr3r540", "collision", "link_1.stl")
    #)

    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )
    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )
    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )
    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )
    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )
    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )

    stl1_vis = mesh.Mesh.from_file(stl1_vis)

    glRotate(-90, 1, 0, 0)
    glRotate(90, 0, 0, 1)
    glRotate(15, 0, -1, 0)
    glTranslate(1, 0, -0.5)
    glPushMatrix()

    #gluLookAt(0,2,0.4, 0,0,0, 0,0,1)
    N = 20
    pts = 20
    
    robot = create_robot()
    plane = []
    for val in np.linspace(-N/2, N/2, pts):
        plane.append([(-N/2, val, 0),(N/2, val, 0)])
        plane.append([(val, -N/2, 0),(val, N/2, 0)])

    end_it = False
    pause = False
    while True and not end_it:
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

                model = glGetDoublev(GL_MODELVIEW_MATRIX)
                """
                GL_MODELVIEW_MATRIX
                Matrix describes BA Transform
                """
                rot = model[:3, :3]
                step = 0.5
                if event.key == pygame.K_UP:
                    pt = np.dot(rot, (0,0,step))
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_DOWN:
                    pt = np.dot(rot, (0,0,-step))
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_LEFT:
                    pt = np.dot(rot, (step,0,0))
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_RIGHT:
                    pt = np.dot(rot, (-step,0,0))
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_KP_PLUS:
                    glTranslate(0, 0, -step)
                elif event.key == pygame.K_KP_MINUS:
                    glTranslate(0, 0,step)

                ang = 15 
                if event.key == pygame.K_KP8:
                    glRotatef(ang, 0, -1, 0)
                elif event.key == pygame.K_KP2:
                    glRotatef(ang, 0, 1, 0)
                elif event.key == pygame.K_KP4:
                    glRotatef(ang, 0, 0, 1)
                elif event.key == pygame.K_KP5:
                    pass
                    glPopMatrix()
                    glPushMatrix()
                    #gluLookAt(5,0,2, 0,0,0, 0,0,1)
                elif event.key == pygame.K_KP6:
                    glRotatef(ang, 0, 0, -1)
                elif event.key == pygame.K_KP7:
                    glRotatef(ang, -1, 0, 0)
                elif event.key == pygame.K_KP9:
                    glRotatef(ang, 1, 0, 0)

        if pause == True:
            pygame.time.wait(300)
            continue
        "END EVENTS"

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_axis()
        render_plane(plane)
        
        glMatrixMode(GL_MODELVIEW)
        glRotate(0.1, 0,0,1)

        "Draw robot skeleton"
        transf = robot.get_transformations()
        glPushMatrix()
        prev = [0,0,0,0]

        for stl_mesh, key in zip([stl1_vis], range(1, len(transf))):
            
            glDisable(GL_DEPTH_TEST)
            glLineWidth(15)
            glColor(1,0.6,0)

            glBegin(GL_LINES)
            trf = transf.get(key)
            point = np.dot(trf, [0,0,0,1])
            glVertex3f(*prev[:3])
            glVertex3f(*point[:3])
            prev = point
            glEnd()
 
            "Mesh render"
            glEnable(GL_DEPTH_TEST)
            glColor(0.1,0.7,1)
            vect = stl1_vis.vectors
            glLineWidth(1)
            glTranslate(*trf[:3, 3])
            #glRotate(30,0,1,0)
            glBegin(GL_LINES)
            for vex in vect:
                for pt in vex:
                    glVertex3f(*pt)
                glVertex3f(*vex[0])
        glEnd()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()



main()


