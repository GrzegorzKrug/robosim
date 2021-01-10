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

def assert_diff(max_error, ptA, ptB, A_pt1, B_pt1):
    ptA = np.array(ptA)
    ptB = np.array(ptB)

    diff1 = ptB - B_pt1[:3]
    diff1 = np.absolute(diff1).sum()

    assert diff1 < max_error, f"point: {ptA}" + "\n" \
        + f"expected inB: {ptB}, got: {B_pt1} diff: {diff1}"

    if False:
        diff2 = ptA - A_pt1[:3]
        diff2 = np.absolute(diff2).sum()
        assert diff2 < max_error, f"point: {ptB}" + "\n" \
            + f"expected inA: {ptA}, got: {A_pt1} diff: {diff2}"

def test14_():
    "Test overlapping coords"
    max_error = 1e-6
    ang = 0
    #trans = np.array([
        #[1, 0, 0, 0],
        #[0, 1, 0, 0],
        #[0, 0, 1, 0],
        #[0, 0, 0, 1],
        #])
    cord1 = RelativeCoordinate()
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
    cord1 = RelativeCoordinate(offset=[1,0,1])

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
    max_error = 1e-8
    cord1 = RelativeCoordinate(axis="z", pos=math.radians(135))

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
        pb = cord1.get_point_fromA(ptA)
        revA = cord1.get_point_fromB(pb)

        pa = cord1.get_point_fromB(ptB)
        revB = cord1.get_point_fromA(pa)
        assert_diff(max_error, ptA, ptB, revA, revB)

def test18_():
    "Test Angle around Y roation and translation"
    max_error = 1e-8
    cord1 = RelativeCoordinate(offset=[1,0,1])
    cord1.add_rotation(85, "y")

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
        pb = cord1.get_point_fromA(ptA)
        revA = cord1.get_point_fromB(pb)

        pa = cord1.get_point_fromB(ptB)
        revB = cord1.get_point_fromA(pa)
        assert_diff(max_error, ptA, ptB, revA, revB)

def test19_():
    "Test with other rotation"
    max_error = 1e-8
    cord1 = RelativeCoordinate()
    cord1.add_rotation(30, "z")
    cord1.add_rotation(85, "x")

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
        pb = cord1.get_point_fromA(ptA)
        revA = cord1.get_point_fromB(pb)

        pa = cord1.get_point_fromB(ptB)
        revB = cord1.get_point_fromA(pa)
        assert_diff(max_error, ptA, ptB, revA, revB)

def test20_inverse_accuracy():
    "Inverse accuracy measure!"
    max_error = 1e-10

    cord1 = RelativeCoordinate(angle=30)
    cord1.add_rotation(85, "x")

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
        pb = cord1.get_point_fromA(ptA)
        revA = cord1.get_point_fromB(pb)

        pa = cord1.get_point_fromB(ptB)
        revB = cord1.get_point_fromA(pa)
        assert_diff(max_error, ptA, ptB, revA, revB)


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
    cord1 = RelativeCoordinate(offset=[1,0,1])

    diff = np.absolute(trans - cord1.endFrame).sum()
    assert diff < max_error, f"Error is too big, TransAB: {trans}, got: {cord1.endFrame}"

    with pytest.raises(AttributeError):
        cord1.endFrame = 50

def test22_():
    pass

def test23_():
    cord1 = RelativeCoordinate(offset=[1,0,1])
    cord1.visualise(block=False)

def test24_():
    cord1 = RelativeCoordinate(offset=[1,0,1])
    cord1.visualise([4,5,6], block=False)

def test25_():
    cord1 = RelativeCoordinate(offset=[1,0,1])
    cord1.visualise([[4,5,6], [4,3,3]], block=False)

def test26_():
    cord1 = RelativeCoordinate(offset=[1,0,1])
    cord1.visualise([[2, 3], [3, 1], [1,2]], block=False)

def test27_rotDefinition():
    "Check if orientation is being defined correctly"
    "X"
    maxError = 1e-5
    cord = RelativeCoordinate()
    mat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert np.absolute(cord.orientation_mat - mat).sum() < maxError, "Error is too big"


def test32_rotDefinition():
    "Check if orientation is being defined correctly"
    maxError = 1e-5
    cord = RelativeCoordinate()
    mat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert np.absolute(cord.orientation_mat - mat).sum() < maxError, "Error is too big"

def test34_():
    "TEST X : Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="x", up="z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test35_():
    "TEST X : -Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="x", up="-z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test36_():
    "TEST -X : Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-x", up="z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test37_():
    "TEST -X : -Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-x", up="-z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test38_():
    "TEST X : Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="x", up="y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test39_():
    "TEST X : -Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="x", up="-y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test40_():
    "TEST -X : Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-x", up="y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test41_():
    "TEST -X : -Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-x", up="-y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [-1, 0, 0],
        [ 0, 0, -1],
        [ 0,-1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test42_():
    "TEST Y : Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="y", up="z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test43_():
    "TEST Y : -Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="y", up="-z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 1, 0],
        [ 1, 0, 0],
        [ 0, 0, -1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test44_():
    "TEST Y : X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="y", up="x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 1, 0],
        [ 0, 0, 1],
        [ 1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test45_():
    "TEST Y : -X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="y", up="-x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 1, 0],
        [ 0, 0, -1],
        [ -1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test46_():
    "TEST -Y : Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-y", up="z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, -1, 0],
        [ 1, 0, 0],
        [ 0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test47_():
    "TEST -Y : -Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-y", up="-z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, -1, 0],
        [ -1, 0, 0],
        [ 0, 0, -1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test48_():
    "TEST -Y : X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-y", up="x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, -1, 0],
        [ 0, 0, -1],
        [ 1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test49_():
    "TEST -Y : -X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-y", up="-x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, -1, 0],
        [ 0, 0, 1],
        [ -1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test50_():
    "TEST Z : -X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="Z", up="-x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test51_():
    "TEST Z : X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="z", up="x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0, 1],
        [ 0,-1, 0],
        [ 1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test52_():
    "TEST Z : Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="Z", up="Y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0, 1],
        [ 1, 0, 0],
        [ 0, 1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test53_():
    "TEST Z : -Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="z", up="-y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0, 1],
        [-1, 0, 0],
        [ 0,-1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test54_():
    "TEST -Z : X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-z", up="x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0,-1],
        [ 0, 1, 0],
        [ 1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test55_():
    "TEST -Z : -X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-z", up="-x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0,-1],
        [ 0,-1, 0],
        [-1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test56_():
    "TEST -Z : y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-z", up="y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0,-1],
        [-1, 0, 0],
        [ 0, 1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test57_():
    "TEST -Z : -y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-z", up="-y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0,-1],
        [ 1, 0, 0],
        [ 0,-1, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test59_():
    "TEST X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test60_():
    "TEST -X"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-x")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test61_():
    "TEST Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test62_():
    "TEST -Y"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-y")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0,-1, 0],
        [ 1, 0, 0],
        [ 0, 0, 1],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test63_():
    "TEST Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test64_():
    "TEST -Z"
    maxError = 1e-5
    cord = RelativeCoordinate(axis="-z")
    transf = cord.orientation_mat
    points = np.eye(3)
    res = np.dot(transf, points)
    solution = np.array([
        [ 0, 0,-1],
        [ 0, 1, 0],
        [ 1, 0, 0],
    ])
    print("Got")
    print(res)
    print("Expected:")
    print(solution)
    assert np.absolute(res - solution).sum() < maxError, "Error is too big"

def test65_():
    pass

