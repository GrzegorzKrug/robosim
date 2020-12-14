import pybullet_data
import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import os


def separate_lines(n=10):
    print("\n" * n)


physicClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9)

cubePos = [0, 0, 1]
cubeRot = p.getQuaternionFromEuler([0, 0, 0])
plane = p.loadURDF("plane.urdf")

robot_pos = [-1, 1, 0]
pirad = np.pi / 2
robot_rot = p.getQuaternionFromEuler([2 * pirad, 0, 0])
robot_rot = p.getQuaternionFromEuler([0, 0, 0])

robot = p.loadURDF("kuka_experimental/kuka_kr210_support/urdf/kr210l150.urdf",
                   robot_pos, robot_rot, useFixedBase=1)
# boxId = pyb.loadURDF("r2d2.urdf", cubePos, cubeRot)
position, orientation = p.getBasePositionAndOrientation(robot)

print("The robot position is {}".format(position))
print("The robot orientation (x, y, z, w) is {}".format(orientation))

nb_joints = p.getNumJoints(robot)
print("The robot is made of {} joints.".format(nb_joints))
print("The arm does not really have 8 joints. It has 6 revolute joints and 2 fixed joints.")

# print information about joint 2
joint_index = 2
joint_info = p.getJointInfo(robot, joint_index)
# print("Joint index: {}".format(joint_info[0]))
# print("Joint name: {}".format(joint_info[1]))
# print("Joint type: {}".format(joint_info[2]))
# print("First position index: {}".format(joint_info[3]))
# print("First velocity index: {}".format(joint_info[4]))
# print("flags: {}".format(joint_info[5]))
# print("Joint damping value: {}".format(joint_info[6]))
# print("Joint friction value: {}".format(joint_info[7]))
# print("Joint positional lower limit: {}".format(joint_info[8]))
# print("Joint positional upper limit: {}".format(joint_info[9]))
# print("Joint max force: {}".format(joint_info[10]))
# print("Joint max velocity {}".format(joint_info[11]))
# print("Name of link: {}".format(joint_info[12]))
# print("Joint axis in local frame: {}".format(joint_info[13]))
# print("Joint position in parent frame: {}".format(joint_info[14]))
# print("Joint orientation in parent frame: {}".format(joint_info[15]))
# print("Parent link index: {}".format(joint_info[16]))

# print state of joint 2
separate_lines()
joints_index_list = range(nb_joints)
joints_state_list = p.getJointStates(robot, joints_index_list)
print(f"JOINTS INDEX LIST", joints_index_list)
print(f"All joints state list: \n{joints_state_list}")
separate_lines(4)

print("Joint position: {}".format(joints_state_list[joint_index][0]))
print("Joint velocity: {}".format(joints_state_list[joint_index][1]))
print("Joint reaction forces (Fx, Fy, Fz, Mx, My, Mz): {}".format(joints_state_list[joint_index][2]))
print("Torque applied to joint: {}".format(joints_state_list[joint_index][3]))
"Position, Velocity, Forces, Torque"

# print state of link 2
link_state_list = p.getLinkState(robot, 2)
print("Link position (center of mass): {}".format(link_state_list[0]))
print("Link orientation (center of mass): {}".format(link_state_list[1]))
print("Local position offset of inertial frame: {}".format(link_state_list[2]))
print("Local orientation offset of inertial frame: {}".format(link_state_list[3]))
print("Link frame position: {}".format(link_state_list[4]))
print("Link frame orientation: {}".format(link_state_list[5]))

# Define gravity in x, y and z
p.setGravity(0, 0, -9.81)

# define a target angle position for each joint (note, you can also control by velocity or torque)
HOME = np.array([-1, 0, 0, 0, 0, 0, 0, 0])
p.setJointMotorControlArray(robot, joints_index_list, p.POSITION_CONTROL,
                            targetPositions=HOME)
pos = []
vel = []
torq = []
targs = []

TRAJECTORY_6dof = [
    [math.radians(-10), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(90), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(30), math.radians(70), math.radians(0), math.radians(50), math.radians(60)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(90), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(90), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,

    [math.radians(20), math.radians(-10), math.radians(-10), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(30), math.radians(-40), math.radians(-40), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(40), math.radians(-50), math.radians(50), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(90), math.radians(-60), math.radians(90), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
    [math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0), math.radians(0)] ,
]
print(f" Target list: {TRAJECTORY_6dof}")

target = TRAJECTORY_6dof.pop(0)
delay_new_step = 50
count_dur = 0  # Counts stop duration in sim steps

for step in range(2000):
    p.stepSimulation()
    time.sleep(1 / 120)  # slow down the simulation
    
    target_8dof = target + [0, 0]
    # print(f"target: {target}")

    p.setJointMotorControlArray(robot, joints_index_list, p.POSITION_CONTROL,
                                    targetPositions=target_8dof)

    joints_state_list = p.getJointStates(robot, joints_index_list)
    joints_state_list = np.array(joints_state_list, dtype=object)

    current_total_speed = np.array(joints_state_list[:6, 1])
    current_total_speed = np.absolute(current_total_speed).sum()

    vel.append(joints_state_list[:6, 0])
    pos.append(joints_state_list[:6, 1])
    torq.append(joints_state_list[:6, 3])
    targs.append(target)
    if current_total_speed < 0.1 and delay_new_step <= step:
        count_dur += 1
        if count_dur >= 10:
            try:
                target  = TRAJECTORY_6dof.pop(0)
                delay_new_step = step + 50
                count_dur = 0
            except Exception:
                print("No more points to go")
                break
    else:
        count_dur = 0

pos = [*zip(*pos)]
vel = [*zip(*vel)]
torq = [*zip(*torq)]
targs = [*zip(*targs)]

time_range = np.arange(len(vel[0])) / 240
colors = [
        [0.7, 0, 0],
        [0, 0.7, 0],
        [0, 0, 0.7],
        [0.6, 0.6, 0],
        [0, 0.6, 0.6],
        [1, 0, 0.6],
        [0.4, 0, 0.1],
        [0.4, 0, 0.6],
]

separate_lines()

plt.figure(figsize=(16, 9))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
# ax3 = plt.subplot(313)
for num, (_pos, _vel, _torq, _targs) in enumerate(zip(pos, vel, torq, targs), 1):
    c = colors[num - 1]
    ax1.plot(time_range, _vel, label=f"{num}", c=c)
    ax1.plot(time_range, _targs, label=f"targ-{num}", c=c, dashes=[4,2])

    ax2.plot(time_range, _pos, label=f"{num}", c=c)
    #ax3.plot(time_range, _torq, label=f"{num}", c=c)

ax1.legend()
ax1.set_title("Position")
ax2.legend()
ax2.set_title("Velocity")
#ax3.legend()
#ax3.set_title("Torque")
plt.suptitle("Kuka Manipulator")
plt.subplots_adjust(hspace=0.3)

plt.show()
p.disconnect()
