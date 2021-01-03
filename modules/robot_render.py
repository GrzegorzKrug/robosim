import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os

from itertools import product, cycle
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

from robotic.model3d import Segment, Model3D
from urdfpy import URDF
from stl import mesh

from utils.models import Cube, Tube
from utils.utils import render_axis, render_plane
from robotic.model3d import point_fromB, point_fromA

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
    val = mod.add_segment(name="anchor")
    val = mod.add_segment(val, name="J1", rotation_axis = "-z",  offset=[0,0,0.345])
    val = mod.add_segment(val, name="J2", rotation_axis = "y",  offset=[0.02,0,0])
    val = mod.add_segment(val, name="J3", rotation_axis = "y",  offset=[0.26,0,0])
    val = mod.add_segment(val, name="J4", rotation_axis ="-x",  offset=[0,0,0.02])
    val = mod.add_segment(val, name="J5", rotation_axis = "y",  offset=[0.26,0,0])
    val = mod.add_segment(val, name="J6", rotation_axis= "-x",  offset=[0.075,0,0])

    #mod.visualise(ax_size=0.1)
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
    glPushMatrix()

    N = 20
    pts = 20

    MESH_COLORS = cycle([
        (1,1,1),
        (1,0.8,0.5),
        (1,0.2,0.7),
        (0.3,0.7,1),
        (0.5,1,1),
        (0.3,0.7,0.4),
        (0,1,0.5),
    ])
    robot = create_robot()

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
                M.T = AB+P
                model.T = T_VA
                model.T = Transformv_Visual_Absolute
                """
                rot = model[:3, :3]
                step = 0.1
                if event.key == pygame.K_UP:
                    pt = np.dot(rot, (0,0,step))
                    pt[1] = pt[1] / (1-pt[2])
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_DOWN:
                    pt = np.dot(rot, (0,0,-step))
                    pt[1] = pt[1] / (1-pt[2])
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_LEFT:
                    pt = np.dot(rot, (step,0,0))
                    pt[0] = pt[0] / (1-pt[2])
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_RIGHT:
                    pt = np.dot(rot, (-step,0,0))
                    pt[0] = pt[0] / (1-pt[2])
                    pt[2] = 0
                    glTranslate(*pt)
                elif event.key == pygame.K_KP_PLUS:
                    glTranslate(0, 0, step)
                elif event.key == pygame.K_KP_MINUS:
                    glTranslate(0, 0,-step)

                ang = 5
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
            pygame.time.wait(100)
            continue
        "END EVENTS"

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_axis()
        render_plane(plane)

        glMatrixMode(GL_MODELVIEW)
        #glRotate(0.1, 0,0,1)

        "Draw robot skeleton"
        all_TR = robot.transf_mats
        all_qt = robot.transf_quats
        glPushMatrix()
        prev = [0,0,0]
        model = glGetFloatv(GL_MODELVIEW_MATRIX)
        #print()
        #print(model)

        for stl_mesh, key in zip(meshes, range(0, 7)):
            glDisable(GL_DEPTH_TEST)

            glLineWidth(15)
            glColor(1,0.6,0)
            segment = robot[key]
            trf = segment.transformation
            #Qvec = segment._quaternion
            some = all_qt.get(key)
            #print(key, some)
            QT, abs_offset = some
            #QT = QT.transpose()
            Qvec = math.degrees(QT.angle()), QT.x, QT.y, QT.z
            #print(Qvec, QT)
            #point = segment.get_point_fromB([0,0,0])
            #point = point + prev

            point = np.dot(all_TR[key], [0,0,0,1])[:3]
            #print(str(prev).ljust(30), str(point))

            glBegin(GL_LINES)
            glVertex3fv(prev)
            glVertex3fv(point)
            glEnd()

            #glLineWidth(2)
            #glBegin(GL_LINES)
            #glColor(0.7,0,0)
            #glVertex3fv(prev + segment.get_point_fromB([0,0,0]))
            #glVertex3fv(prev + segment.get_point_fromB([1,0,0]))
            #glColor(0,0.7,0)
            #glVertex3fv(prev + segment.get_point_fromB([0,0,0]))
            #glVertex3fv(prev + segment.get_point_fromB([0,1,0]))
            #glColor(0,0,0.7)
            #glVertex3fv(prev + segment.get_point_fromB([0,0,0]))
            #glVertex3fv(prev + segment.get_point_fromB([0,0,1]))
            #glEnd()
            prev = point

            "Mesh render"
            glEnable(GL_DEPTH_TEST)

            color = next(MESH_COLORS)
            glColor(color)
            if stl_mesh:
                #vect = stl_mesh.vectors
                glLineWidth(1)
                glPushMatrix()
                glTranslate(*abs_offset)
                glCallList(key)
                ##glRotate(*Qvec)
#
                #glBegin(GL_LINES)
                #for vex in vect:
                    #for pt in vex:
                        #glVertex3f(*pt)
                    #glVertex3f(*vex[0])
                #glEnd()
                glPopMatrix()

        glPopMatrix()

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


def animate(i):
    #if i == 0:
        #time.sleep(1)
    global phase

    cycle = 40
    step = -(360) / cycle
    traillen = 2 * cycle

    if not ((i+1) % cycle):
        phase = (phase + 1) % 6

    robot[1].angle = i*4
    robot[2].angle = -i/2
    robot[3].angle = i*2
    robot[4].angle = i*1.6
    robot[5].angle = i*7
    robot[6].angle = i




    ax.clear()
    trf = robot.transf_mats
    robot.draw(ax, ax_size=0.03, textsize=8)
    end_trf = robot.transf_mats[6]
    orien = end_trf[:3, :3]
    offset = end_trf[:3, -1]

    end_point = point_fromB(orien, offset=offset, point=[0,0,0])
    trail.append(end_point)
    pts = np.array(trail[-traillen:]).T
    cols = np.clip(np.absolute(pts).T*2+[0,0,-0.3], 0, 1)
    ax.scatter(pts[0, :], pts[1, :], pts[2, :], c=cols)

    for num in range(1,7):
        msh = robot[num].mesh
        #print(vect)
        #vect = vect.T
        vecs = msh.vectors + robot.transf_mats[num][:3,-1]
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
            vecs[list(range(0, vecs.shape[0], 2)),:,:])
        )
        #for num, vex in enumerate(vect):
            ##print(vex)
            #plt.plot(vex, vect[:, num-1], c=(1,0,0))

    ed = 0.6
    ax.set_xlim([-ed, ed])
    ax.set_ylim([-ed, ed])
    ax.set_zlim([0, ed])


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




