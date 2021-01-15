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
from robotic.model3d import point_fromB, point_fromA, get_quat, get_quat_normalised

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
    model.T = Transform_Visual_Absolute
    """
    glMatrixMode(GL_MODELVIEW)

    rot = model[:3, :3]
    global camera_state
    global camera_initial_state
    qcam_ini, pos_ini = camera_initial_state
    qcam, pos = camera_state
    ang = 3
    step = 0.05

    if key == pygame.K_UP:
        pt = np.dot(rot, (0,0,step))
        pt[1] = pt[1] / (1-pt[2])
        pt[2] = 0
        #glTranslate(*pt)
        pos += pt
    elif key == pygame.K_DOWN:
        pt = np.dot(rot, (0,0,-step))
        pt[1] = pt[1] / (1-pt[2])
        pt[2] = 0
        #glTranslate(*pt)
        pos += pt
    elif key == pygame.K_LEFT:
        pt = np.dot(rot, (step,0,0))
        pt[0] = pt[0] / (1-pt[2])
        pt[2] = 0
        #glTranslate(*pt)
        pos += pt
    elif key == pygame.K_RIGHT:
        pt = np.dot(rot, (-step,0,0))
        pt[0] = pt[0] / (1-pt[2])
        pt[2] = 0
        #glTranslate(*pt)
        pos += pt
    elif key == pygame.K_KP_PLUS:
        #glTranslate(0, 0, step)
        pos += np.array([0,0,step])
    elif key == pygame.K_KP_MINUS:
        #glTranslate(0, 0,-step)
        pos += np.array([0,0,-step])
    else:
        pass
        #pt = 0,0,0

    rotVec = None
    if key == pygame.K_KP8:
        rotVec = -1, 0, 0
        q1 = get_quat(ang, rotVec)
        qcam = q1 * qcam
    elif key == pygame.K_KP2:
        rotVec = 1, 0, 0
        q1 = get_quat(ang, rotVec)
        qcam = q1 * qcam
    elif key == pygame.K_KP4:
        rotVec = 0, 0, -1
        q1 = get_quat(ang, rotVec)
        qcam = qcam * q1
    elif key == pygame.K_KP6:
        rotVec = 0, 0, 1
        q1 = get_quat(ang, rotVec)
        qcam = qcam * q1
    elif key == pygame.K_KP7:
        rotVec = 0, 0, -1
        q1 = get_quat(ang, rotVec)
        qcam = q1 * qcam
    elif key == pygame.K_KP9:
        rotVec = 0, 0, 1
        q1 = get_quat(ang, rotVec)
        qcam = q1 * qcam
    elif key == pygame.K_KP5:
        camera_state = camera_initial_state
        return None

    camera_state = (qcam, pos)
    return True


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

    #glRotate(-90, 1, 0, 0)
    #glRotate(90, 0, 0, 1)
    #glTranslate(0.8, -0.35, -0.2)
    #glRotate(20, 0, -1, 0)
    #glRotate(35, 0, 0, 1)
    #glTranslate(1, -1,-0.71)
    glPushMatrix()
    q1 = get_quat_normalised(90, [0, 1, 0])
    q2 = get_quat_normalised(90, [0, 0, 1])
    q3 = get_quat_normalised(20, [1, 0, 0])

    qcam = q3 * q2 * q1
    global camera_state, camera_initial_state
    camera_initial_state = (qcam, [3, 0, -1])
    camera_state = camera_initial_state

    N = 20
    pts = 20

    line_shader = """
    #version 330
    in vec4 position;
    void main()
    {
        gl_Position = position;
    }
    """

    fragment_shader = """
    #version 330

    void main()
    {
        gl_FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }

    """

    MESH_COLORS = cycle([
        (1,1,1), # root
        (1,1,0.2),
        (1,0,0.5),
        (0,0.7,1),
        (0,0.9,0.5),
        (1,0.8,0.2),
        (0.6,0,1),
    ])
    rob1 = create_robot()
    rob1[0].offset = [0, -1, 0]

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
        "Define lists"
        color = next(MESH_COLORS)
        if stl_mesh:
            vect = stl_mesh.vectors
            glLineWidth(1)
            glPushMatrix()

            glNewList(key, GL_COMPILE)
            glBegin(GL_TRIANGLES)
            for vex in vect:
                for pt in vex:
                    glVertex3f(*pt)
                #glVertex3f(*vex[0])
            glEnd()
            glEndList()
            glPopMatrix()


    phase = 0
    i = 0
    end_it = False
    pause = False
    trail_cols = np.linspace((0.5, 0.2, 0), (0, 1, 0), 100)
    trail = deque(maxlen=100)
    dist_list = []
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
            dur = 100
            rand = np.random.randint(0,90)
            step = (360) / dur #*3.2
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

            rob1[1].angle = (time.time()*125)%360
            #robot[1].angle = 0
            rob1[2].angle = math.sin(time.time()/2)*80-90
            #robot[2].angle = -90
            rob1[3].angle = math.sin(time.time())*60
            rob1[4].angle = math.sin(time.time())*50-40
            rob1[5].angle = math.sin(time.time()/5)*60
            rob1[6].angle = (time.time()*50)%360
            rob1.calculate_transformations()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        q1, off = camera_state
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rad, *vec = q1.angle(), q1.x, q1.y, q1.z
        ang = math.degrees(rad)
        #print(ang, vec)
        glRotatef(ang, *vec)
        glTranslate(*off)

        render_plane(plane)
        #render_axis()

        #print(" ".join(f"{ang%360:>5.1f}" for ang in rob1.joints))

        #glRotate(0.1, 0,0,1)

        "Draw robot skeleton"
        POINT = (0.7, 0.6, 1)

        for rb_num, rb in enumerate([rob1, rob2]):
            transf_mats = rb.transf_mats
            transf_quats = rb.transf_quats

            glPushMatrix()
            for stl_mesh, key in zip(meshes, range(0, 7)):
                "Robot Segment"
                segment = rb[key]

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

                if rb_num == 1 or rb_num == 0 and key == 6 or key == 0:
                    render_axis(ax_size=0.1, ax_width=5, render_corner=False)

            glPopMatrix()

            glPushMatrix()
            glDisable(GL_DEPTH_TEST)

            prev = rb[0].offset
            for stl_mesh, key in zip(meshes, range(0, 7)):
                color = next(MESH_COLORS)
                transf = transf_mats[key]
                offset = transf[:3, -1]
                skelet = offset

                glLineWidth(6)
                glBegin(GL_LINES)
                glColor(0,0,0)
                glVertex3fv(prev)
                glVertex3fv(skelet)
                glEnd()

                glLineWidth(4)
                glBegin(GL_LINES)
                glColor(color)
                glVertex3fv(prev)
                glVertex3fv(skelet)
                glEnd()
                prev = skelet

            glPopMatrix()

        hi_dist = math.inf
        closest = None
        x = time.time()/10
        POINT = (math.cos(x)*2+1, math.sin(x/3)*2-1, 1.4)
        glColor(1,1,1)
        for seg_num in range(7):
            col = next(MESH_COLORS)
            if seg_num == 0:
                continue
            transf = rob1.transf_mats[seg_num]
            rot = transf[:3, :3]
            pos = transf[:3, -1]

            #closest = point_fromB(rot, transf[:3,-1], [.1,.1,0])
            query = point_fromA(rot, pos, POINT)
            seg = rob2[seg_num]
            msh = meshes[seg_num]
            pts = msh.vectors.reshape((-1, 3))
            dists = (pts - query)**2
            dists = dists.sum(axis=-1)
            low_dist = dists.min()
            if low_dist < hi_dist:
                hi_dist = low_dist
                closest = pts[np.where(dists==low_dist)][0]
                closest = point_fromB(rot, pos, closest)
                glColor(*col)


        glEnable(GL_DEPTH_TEST)

        if closest is not None:
            dist_list.append(hi_dist)
            glLineWidth(8)
            glBegin(GL_LINES)
            glVertex3fv(POINT)
            glColor(1,1,1)
            glVertex3fv(closest)
            glEnd()
            glPushMatrix()
            glTranslate(*POINT)
            render_axis(render_corner=False, ax_size=0.1)
            glPopMatrix()

        "Render Trail of robot1"
        transf_mats = rob1.transf_mats
        last = transf_mats[6][:3, -1]
        trail.append(last)
        #glPointSize(4)

        for num, (pt, col) in enumerate(zip(trail, trail_cols), 1):
            glColor(*col)
            glPointSize(math.sqrt(num))
            glBegin(GL_POINTS)
            glVertex3f(*pt)
            glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

    plt.figure()
    plt.title("Distance of point to robot")
    plt.plot(dist_list)
    plt.grid()
    plt.show()


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




