from invertkin import PlainModel
from model3d import RelativeCoordinate

import pytest
import numpy as np
import math


def test1_Bottom():
    model = PlainModel([6, 4])
    diff =  model.cartesian - np.array([[0, -6], [0, -10]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test2_EastAngle():
    model = PlainModel([6, 4])
    model.joints_deg = [90, 0]
    diff =  model.cartesian - np.array([[6, 0], [10, 0]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test3_West():
    model = PlainModel([6, 4])
    model.joints_deg = [-90, 0]
    diff =  model.cartesian - np.array([[-6, 0], [-10, 0]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test4_Top():
    model = PlainModel([6, 4])
    model.joints = [math.radians(180), 0]
    diff =  model.cartesian - np.array([[0,6], [0, 10]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test5_BendEast():
    model = PlainModel([6, 4])
    model.joints = [0, math.radians(90)]
    diff =  model.cartesian - np.array([[0, 4], [-6, -6]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test6_BendWest():
    model = PlainModel([6, 4])
    model.joints_deg = [0, -90]
    diff =  model.cartesian - np.array([[0, -4], [-6, -6]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test7_BendReverse():
    model = PlainModel([6, 4])
    model.joints_deg = [0, 180]
    diff =  model.cartesian - np.array([[0, 0], [-6, 2]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test8_Model3():
    model = PlainModel([10, 5, 3])
    model.joints = [0, 0 ,0]
    diff =  model.cartesian - np.array([[0, 0, 0], [-10, -15 ,18]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test9_Model3_Bend():
    model = PlainModel([10, 5, 3])
    model.joints_deg = [0, 0, 90]
    diff =  model.cartesian - np.array([[0, 0, 3], [-10, -15, 15]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test9_Model3_Bend_2():
    model = PlainModel([10, 5, 3])
    model.joints_deg = [0, 0, -90]
    diff =  model.cartesian - np.array([[0, 0, -3], [-10, -15, 15]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

#def test11_():
    #model = PlainModel([10, 5, 3])
    #solutions = model.find_pos([0, 18])
    #solutions2 = model.find_pos([0, -18])
    #assert len(solutions) == 1, "This models has only 1 solution"
    #assert len(solutions2) == 1, "This models has only 1 solution"
#
#def test12_():
    #model = PlainModel([10, 5, 3])
    #solutions = model.find_pos([0, 16])
    #solutions2 = model.find_pos([0, -16])
    #assert len(solutions) == 2, "This models has exact 2 solutions"
    #assert len(solutions2) == 2, "This models has exact 2 solutions"
#
#def test13_():
    #model = PlainModel([10, 5, 3])
    #solutions = model.find_pos([0, 12])
    #solutions2 = model.find_pos([0, -12])
    #assert len(solutions) > 2, "This models has more than 2 solutions"
    #assert len(solutions2) > 2, "This models has more than 2 solutions"


def assert_diff(max_error, ptA, ptB, A_pt1, B_pt1):
    diff1 = ptB - B_pt1[:3]
    diff1 = np.absolute(diff1).sum()
    
    assert diff1 < max_error, f"point: {ptA}" + "\n" \
        + f"expected inB: {ptB}, got: {B_pt1} diff: {diff1}"

    if ptA:
        diff2 = ptA - A_pt1[:3]
        diff2 = np.absolute(diff2).sum()
        assert diff2 < max_error, f"point: {ptB}" + "\n" \
            + f"expected inA: {ptA}, got: {A_pt1} diff: {diff2}"

def test14_():
    "Test overlapping coords"
    max_error = 1e-6
    ang = 0
    trans = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)
    "A : B Pairs"
    pairs = (
        ([0,0,0], [0,0,0]),
        ([1,0,0], [1,0,0]),
        ([0,1,0], [0,1,0]),
        ([0,1,0], [0,1,0]),
        ([0,6,0], [0,6,0]),
        ([0,9,8], [0,9,8]),
        ([3,9,8], [3,9,8]),
        ([-3,9,8], [-3,9,8]),
        ([3,-5,8], [3,-5,8]),
        ([3,9,-1], [3,9,-1]),
        ([3,-0.3,0.6], [3,-0.3,0.6]),
    )
    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, ptA, ptB, A_pt1, B_pt1)


def test15_():
    "Test same angles, with translation"
    max_error = 1e-6
    ang = 0
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)

    "A : B Pairs"
    pairs = [
        ((x,y,z),(x-1,y,z-1)) for x,y,z in [
            (0,0,0),
            (1,1,1),
            (4,1,4),
            (-0.3,0.5,0.3),
            (3,2,4),
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
        ]
    ]

    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, ptA, ptB, A_pt1, B_pt1)

def test16_():
    "Test same angles, with translation"
    max_error = 1e-6
    ang = 0
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)

    "A : B Pairs"
    pairs = [
        ((x,y,z),(x-1,y,z-1)) for x,y,z in [
            (0,0,0),
            (1,1,1),
            (4,1,4),
            (-0.3,0.5,0.3),
            (3,2,4),
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
            np.random.random(3)*20-5,
        ]
    ]

    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, ptA, ptB, A_pt1, B_pt1)

def test17_():
    "Test Angle around Z roation and translation"
    max_error = 1e-3
    ang = math.radians(135)
    trans = np.array([
        [math.cos(ang), -math.sin(ang), 0, 1],
        [math.sin(ang), math.cos(ang), 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)

    "A : B Pairs"
    pairs = [
        [[1,2,3],       [1.41412, -1.41421, 2]],
        [[0,0,0],       [0.70711, 0.70711, -1]],
        [[-3,-5,0.1],   [-0.70711, 6.36396, -0.9]],
        [[1,-4,-0.2],   [-2.82843, 2.82843, -1.2]],
        [[1.5,0.1,2],   [-0.28284, -0.42426, 1.0]],
        [[3.1,0.3,0.6], [-1.27279, -1.69706, -0.4]],
    ]

    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, ptA, ptB, A_pt1, B_pt1)

def test18_():
    "Test Angle around Y roation and translation"
    max_error = 1e-3
    ang = math.radians(85)
    OFF = [1,0,1]
    trans = np.array([
        [math.cos(ang), 0, -math.sin(ang), OFF[0]],
        [0, 1, 0, OFF[1]],
        [math.sin(ang), 0, math.cos(ang), OFF[2]],
        [0,0,0,1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)

    "A : B Pairs"
    pairs = [
        [[1,2,3],       [1.99239, 2.0, 0.17431]],
        [[0,0,0],       [-1.08335, 0.0, 0.90904]],
        [[-3,-5,0.1],   [-1.2452, -5.0, 3.90634]],
        [[1,-4,-0.2],   [-1.19543, -4.0, -0.10459]],
        [[1.5,0.1,2],   [1.03977, 0.1, -0.41094]],
        [[3.1,0.3,0.6], [-0.21545, 0.3, -2.12687]],
    ]

    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, ptA, ptB, A_pt1, B_pt1)

def test19_():
    "Test with other rotation"
    max_error = 1e-3
    ang = math.radians(30)
    ang2 = math.radians(85)
    OFF = [1,0,1]
    
    rotZ30 = [
        [math.cos(ang), -math.cos(ang2), 0],
        [math.sin(ang), math.cos(ang), 0],
        [0, 0, 1],
    ]

    rotX85 = [
        [1, 0, 0],
        [0, math.cos(ang2), -math.sin(ang2)],
        [0, math.sin(ang2), math.cos(ang2)],
    ]
    rot = np.dot(rotX85, rotZ30)
    transf = np.eye(4)
    transf[:3, -1] = OFF
    transf[:3, :3] = rot
    cord1 = RelativeCoordinate(transformation=transf)

    "A : B Pairs"
    pairs = [
        [[1,2,3],       [1.0833504, 1.876418, -1.8180779]],
        [[0,0,0],       [-1.36412, -0.775557, -0.08716]],
        [[-3,-5,0.1],   [-4.13028, -0.80523, 4.90253]],
        [[1,-4,-0.2],   [-0.77203, -1.33719, 3.88019]],
        [[1.5,0.1,2],   [0.93547, 0.82670, -0.01246]],
        [[3.1,0.3,0.6], [1.63249, -0.50548, -0.33372]],
    ]

    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, None, ptB, A_pt1, B_pt1)

def test20_inverse_accuracy():
    "Inverse accuracy measure!"
    max_error = 1e-3
    ang = math.radians(30)
    ang2 = math.radians(85)
    OFF = [1,0,1]
    
    rotZ30 = [
        [math.cos(ang), -math.cos(ang2), 0],
        [math.sin(ang), math.cos(ang), 0],
        [0, 0, 1],
    ]

    rotX85 = [
        [1, 0, 0],
        [0, math.cos(ang2), -math.sin(ang2)],
        [0, math.sin(ang2), math.cos(ang2)],
    ]
    rot = np.dot(rotX85, rotZ30)
    transf = np.eye(4)
    transf[:3, -1] = OFF
    transf[:3, :3] = rot
    cord1 = RelativeCoordinate(transformation=transf)

    "A : B Pairs"
    pairs = [
        [[1,2,3],       [1.0833504, 1.876418, -1.8180779]],
        [[0,0,0],       [-1.36412, -0.775557, -0.08716]],
        [[-3,-5,0.1],   [-4.13028, -0.80523, 4.90253]],
        [[1,-4,-0.2],   [-0.77203, -1.33719, 3.88019]],
        [[1.5,0.1,2],   [0.93547, 0.82670, -0.01246]],
        [[3.1,0.3,0.6], [1.63249, -0.50548, -0.33372]],
    ]

    for ptA, ptB in pairs:
        B_pt1 = cord1.get_point_fromA(ptA)
        A_pt1 = cord1.get_point_fromB(ptB)
        assert_diff(max_error, ptA, ptB, A_pt1, B_pt1)


def test21_():
    "EndFrame Check"
    max_error = 1e-6
    ang = 0
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)
    
    diff = np.absolute(trans - cord1.endFrame).sum()
    assert diff < max_error, f"Error is too big, TransAB: {trans}, got: {cord1.endFrame}"

    with pytest.raises(AttributeError):
        cord1.endFrame = 50

def test22_():
    pass

def test23_():
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)
    cord1.visualise(block=False)

def test24_():
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)
    cord1.visualise([4,5,6], block=False)

def test25_():
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)
    cord1.visualise([[4,5,6], [4,3,3]], block=False)

def test26_():
    trans = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ])
    cord1 = RelativeCoordinate(transformation=trans)
    cord1.visualise([[2, 3], [3, 1], [1,2]], block=False)

def test27_():
    pass

def test28_():
    pass

def test29_():
    pass

def test30_():
    pass

def test31_():
    pass

def test32_():
    pass

def test33_():
    pass

def test34_():
    pass

def test35_():
    pass

def test36_():
    pass

def test37_():
    pass

def test38_():
    pass

def test39_():
    pass

def test40_():
    pass

def test41_():
    pass

def test42_():
    pass

def test43_():
    pass

def test44_():
    pass

def test45_():
    pass

def test46_():
    pass

def test47_():
    pass

def test48_():
    pass

def test49_():
    pass

def test50_():
    pass
