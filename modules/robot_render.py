import matplotlib.pyplot as plt
import quaternion
import numpy as np
import math
import time
import os

from collections import deque
from itertools import product, cycle
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

from robotic.model3d import Segment, Model3D
from urdfpy import URDF
from stl import mesh

from utils.models import Cube, Tube
from utils.utils import render_axis, render_plane
from robotic.model3d import point_fromB, point_fromA, get_quat

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
    #val = mod.add_segment(name="anchor")
    val = mod.add_segment(name="J1", rotation_axis = "-z",  offset=[0,0,0.345])
    val = mod.add_segment(val, name="J2", rotation_axis = "y",  offset=[0.02,0,0])
    val = mod.add_segment(val, name="J3", rotation_axis = "y",  offset=[0.26,0,0])
    val = mod.add_segment(val, name="J4", rotation_axis ="-x",  offset=[0,0,0.02])
    val = mod.add_segment(val, name="J5", rotation_axis = "y",  offset=[0.26,0,0])
    val = mod.add_segment(val, name="J6", rotation_axis= "-x",  offset=[0.075,0,0])

    #mod.visualise(ax_size=0.1)
    return mod

def control(key):
    model = glGetDoublev(GL_MODELVIEW_MATRIX)
    """
    GL_MODELVIEW_MATRIX
    Matrix describes BA Transform
    M.T = AB+P
    model.T = T_VA
    model.T = Transformv_Visual_Absolute
    """
    rot = model[:3, :3]
    ang = 3
    step = 0.05
    if key == pygame.K_UP:
        pt = np.dot(rot, (0,0,step))
        pt[1] = pt[1] / (1-pt[2])
        pt[2] = 0
        glTranslate(*pt)
    elif key == pygame.K_DOWN:
        pt = np.dot(rot, (0,0,-step))
        pt[1] = pt[1] / (1-pt[2])
        pt[2] = 0
        glTranslate(*pt)
    elif key == pygame.K_LEFT:
        pt = np.dot(rot, (step,0,0))
        pt[0] = pt[0] / (1-pt[2])
        pt[2] = 0
        glTranslate(*pt)
    elif key == pygame.K_RIGHT:
        pt = np.dot(rot, (-step,0,0))
        pt[0] = pt[0] / (1-pt[2])
        pt[2] = 0
        glTranslate(*pt)
    elif key == pygame.K_KP_PLUS:
        glTranslate(0, 0, step)
    elif key == pygame.K_KP_MINUS:
        glTranslate(0, 0,-step)

    rotVec = None
    if key == pygame.K_KP8:
        rotVec = 0, -1, 0
    elif key == pygame.K_KP2:
        rotVec = 0, 1, 0
    elif key == pygame.K_KP4:
        rotVec = 0, 0, 1
    elif key == pygame.K_KP5:
        glPopMatrix()
        glPushMatrix()
    elif key == pygame.K_KP6:
        rotVec = 0, 0, -1
    elif key == pygame.K_KP7:
        rotVec = -1, 0, 0
    elif key == pygame.K_KP9:
        rotVec = 1, 0, 0

    if rotVec:
        glMatrixMode(GL_MODELVIEW)
        #q1 = get_quat(ang, rotVec)
        #rot = quaternion.as_rotation_matrix(q1)
        #rot_mat = np.eye(4, dtype=np.float)
        #rot_mat[:3, :3] = rot
        #pos = view_matrix[-1, :3]
        #offset = point_fromB(rot, [0,0,0], pos)

        #glTranslate(*(-pos))
        #glMultMatrixf(rot_mat)
        glRotate(ang, *rotVec)
        #print(view_matrix)
        #print(pos)
        #glTranslate(*pos)

    #view_matrix = glGetFloat(GL_MODELVIEW_MATRIX)
    #print(view_matrix)

def main():
    keyCache = dict()
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
    stl2_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_2.stl")
    )
    stl3_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_3.stl")
    )
    stl4_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_4.stl")
    )
    stl5_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_5.stl")
    )
    stl6_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_6.stl")
    )

    stl1_vis = mesh.Mesh.from_file(stl1_vis)
    stl2_vis = mesh.Mesh.from_file(stl2_vis)
    stl3_vis = mesh.Mesh.from_file(stl3_vis)
    stl4_vis = mesh.Mesh.from_file(stl4_vis)
    stl5_vis = mesh.Mesh.from_file(stl5_vis)
    stl6_vis = mesh.Mesh.from_file(stl6_vis)

    meshes = [None, stl1_vis, stl2_vis, stl3_vis, stl4_vis, stl5_vis, stl6_vis]

    glRotate(-90, 1, 0, 0)
    glRotate(90, 0, 0, 1)
    glTranslate(0.8, -0.35, -0.2)
    glRotate(20, 0, -1, 0)
    glRotate(35, 0, 0, 1)
    glTranslate(1, -1,-0.71)
    glPushMatrix()

    N = 20
    pts = 20

    MESH_COLORS = cycle([
        (1,1,1),
        (1,0.8,0.5),
        (1,0.2,0.7),
        (0.3,0.7,1),
        (0.5,0.4,0.8),
        (0.3,1,0.2),
        (1,0,0.3),
    ])
    rob1 = create_robot()
    rob2  = create_robot()
    rob2[0].offset = [0, 1, 0]

    part1 = Model3D()
    part1.add_segment(rotation_axis="-z", offset=[0, 0, 0.345])
    part2 = Model3D()
    part2.add_segment(rotation_axis="y", offset=[0.02, 0, 0])
    part3 = Model3D()
    part3.add_segment(rotation_axis="y", offset=[0.26, 0, 0])
    part4 = Model3D()
    part4.add_segment(rotation_axis="-x", offset=[0, 0, 0.04])
    part5 = Model3D()
    part5.add_segment(rotation_axis="y", offset=[0.26, 0, 0])
    part6 = Model3D()
    part6.add_segment(rotation_axis="-x", offset=[0.075, 0, 0])

    PARTS = (part1, part2, part3, part4, part5, part6)

    plane = []
    for val in np.linspace(-N/2, N/2, pts):
        plane.append([(-N/2, val, 0),(N/2, val, 0)])
        plane.append([(val, -N/2, 0),(val, N/2, 0)])


    for stl_mesh, key in zip(meshes, range(0, 7)):
        color = next(MESH_COLORS)
        glColor(color)
        if stl_mesh:
            vect = stl_mesh.vectors
            glLineWidth(1)
            glPushMatrix()
            #glTranslate(*abs_offset)
            #glRotate(*Qvec)

            glNewList(key, GL_COMPILE)
            glBegin(GL_LINES)
            for vex in vect:
                for pt in vex:
                    glVertex3f(*pt)
                glVertex3f(*vex[0])
            glEnd()
            glEndList()
            glPopMatrix()

    phase = 0
    i = 0
    end_it = False
    pause = False
    trail = deque(maxlen=100)
    while True and not end_it:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end_it = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    end_it = True
                    break
                elif event.key == pygame.K_SPACE:
                    pause ^= True
                else:
                    keyCache[event.key] = True
                    #print(event.key)


            if event.type == pygame.KEYUP:
                try:
                    keyCache.pop(event.key)
                except KeyError:
                    print(f"Key is not in keycache: {event.key}")

        for key in keyCache.keys():
            control(key)


        if pause == True:
            pygame.time.wait(10)
        else:
            dur = 220
            step = -(360) / dur
            #traillen = 2 * cycle
            i += 1

            if not ((i+1) % dur):
                phase = (phase + 1) % 6

            if phase == 0:
                rob1[1].angle += step
                rob2[1].angle += step
            elif phase == 1:
                rob1[2].angle += step# * 0.5
                rob2[2].angle += step# * 0.5
            elif phase == 2:
                rob1[3].angle += step# * 2
                rob2[3].angle += step# * 2
            elif phase == 3:
                rob1[4].angle += step# * 3
                rob2[4].angle += step# * 3
            elif phase == 4:
                rob1[5].angle += step# * 5
                rob2[5].angle += step# * 5
            else:
                rob1[6].angle += step
                rob2[6].angle += step

            rob1[1].angle = (time.time()*45)%360
            #robot[1].angle = 0
            rob1[2].angle = math.sin(time.time()/2)*40-90
            #robot[2].angle = -90
            rob1[3].angle = math.sin(time.time())*60
            rob1[4].angle = math.sin(time.time()/3)*30-40
            rob1[5].angle = math.sin(time.time()*3)*60
            rob1[6].angle = (time.time()*50)%360
            rob1.calculate_transformations()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_plane(plane)
        #render_axis()

        #print(" ".join(f"{ang%360:>5.1f}" for ang in rob1.joints))

        glMatrixMode(GL_MODELVIEW)
        #glRotate(0.1, 0,0,1)

        "Draw robot skeleton"
        #robot.calculate_transformations()
        model = glGetFloatv(GL_MODELVIEW_MATRIX)
        #print()
        #print(model)
        #rob1[1].add_rotation(5, [0,0,1])

        for rb in [rob1, rob2]:
            prev = [0,0,0]
            transf_mats = rb.transf_mats
            transf_quats = rb.transf_quats

            glPushMatrix()
            for stl_mesh, key in zip(meshes, range(0, 7)):
                "Robot Segment"
                segment = rb[key]
                #print(segment.offset)

                glEnable(GL_DEPTH_TEST)
                color = next(MESH_COLORS)
                glColor(color)


                rot_mat = np.eye(4)
                rot_mat[:3, :3] = segment.orientation_mat_state

                QT = segment.orientation_state
                Qvec = math.degrees(QT.angle()), QT.x, QT.y, QT.z

                glTranslate(*segment.offset)
                glRotate(*Qvec)

                if stl_mesh:
                    glLineWidth(1)
                    glCallList(key)

                render_axis(ax_size=0.1, ax_width=5, render_corner=False)
            glPopMatrix()

            glPushMatrix()
            glDisable(GL_DEPTH_TEST)

            #for stl_mesh, key in zip(meshes, range(0, 7)):
                #color = next(MESH_COLORS)
                #glColor(color)
                #transf = transf_mats[key]
                #rot_mat = transf[:3, :3]
                #offset = transf[:3, -1]
                ##skelet = point_fromA(rot_mat, [0,0,0], offset)
                #skelet = offset
                ##skelet = np.dot(transf_mats[key], [0,0,0,1])[:3]
                ##print(skelet, offset)
#
                #glLineWidth(15)
                #glBegin(GL_LINES)
                #glVertex3fv(prev)
                #glVertex3fv(skelet)
                #glEnd()
                #prev = skelet

            glPopMatrix()


        #glColor(next(MESH_COLORS))
        #for ind, part in enumerate(PARTS, 1):
            #glPushMatrix()
            #glPointSize(15)
            #glTranslate(0, ind/2 + 3, 0)
#
            #glPushMatrix()
            #part[1].angle = time.time()*35%360
            #part.calculate_transformations()
            #transf_mats = part.transf_mats
#
            #QT = part[1].orientation_state
            #Qvec = math.degrees(QT.angle()), QT.x, QT.y, QT.z
            #render_axis(render_corner=False, ax_size=0.2)
            #glTranslate(*part[1].offset)
            #glRotate(*Qvec)
            #glColor(next(MESH_COLORS))
            #glLineWidth(0.1)
#
            #glCallList(ind)
            #glColor(255,255,255)
            #glBegin(GL_POINTS)
            ##glVertex3f(*part[0].offset)
            #glVertex3f(0,0,0)
            #glEnd()
            #render_axis(render_corner=False, ax_size=0.1)
            #glPopMatrix()
#
            #trf = transf_mats[1]
            #point = trf[:3, -1]
            #glColor(255,0,0)
            #glPointSize(10)
            #glBegin(GL_POINTS)
            #glVertex3f(*point)
            #glEnd()
#
            #glPopMatrix()

        transf_mats = rob1.transf_mats
        last = transf_mats[6][:3, -1]
        trail.append(last)
        trail_cols = np.linspace((0.2, 0.2, 0), (0, 1, 0), len(trail))
        glPointSize(4)

        for pt, col in zip(trail, trail_cols):
            glColor(*col)
            glBegin(GL_POINTS)
            glVertex3f(*pt)
            glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()


def load_meshes():
    robot_dir = os.path.join("kuka_experimental", "kuka_kr3_support")

    stl1_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_1.stl")
    )
    stl2_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_2.stl")
    )
    stl3_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_3.stl")
    )
    stl4_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_4.stl")
    )
    stl5_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_5.stl")
    )
    stl6_vis = os.path.abspath(
        os.path.join(robot_dir, "meshes", "kr3r540", "visual", "link_6.stl")
    )

    stl1_vis = mesh.Mesh.from_file(stl1_vis)
    stl2_vis = mesh.Mesh.from_file(stl2_vis)
    stl3_vis = mesh.Mesh.from_file(stl3_vis)
    stl4_vis = mesh.Mesh.from_file(stl4_vis)
    stl5_vis = mesh.Mesh.from_file(stl5_vis)
    stl6_vis = mesh.Mesh.from_file(stl6_vis)
    meshes = (stl1_vis, stl2_vis, stl3_vis, stl4_vis, stl5_vis, stl6_vis)
    return meshes


if __name__ == '__main__':
    main()
    #robot = create_robot()
    #meshes = load_meshes()
    #for num, mesh in enumerate(meshes, 1):
        #robot[num].mesh = mesh
#
    #phase = 0
    #fig = plt.figure(figsize=(10,7))
    #ax = plt.gca(projection="3d")
    #trail = []
    ##ani = FuncAnimation(fig, animate, interval=10)
    #animate(0)
    #plt.show()




