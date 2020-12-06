from invertkin import PlainModel
import numpy as np
import math

def test1_Bottom():
    model = PlainModel([6, 4])
    diff =  model.pos - np.array([[0, -6], [0, -10]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test2_EastAngle():
    model = PlainModel([6, 4])
    model.set_pos([math.radians(90), 0])
    diff =  model.pos - np.array([[6, 0], [10, 0]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test3_West():
    model = PlainModel([6, 4])
    model.set_pos([math.radians(-90), 0])
    diff =  model.pos - np.array([[-6, 0], [-10, 0]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test4_Top():
    model = PlainModel([6, 4])
    model.set_pos([math.radians(180), 0])
    diff =  model.pos - np.array([[0,6], [0, 10]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test5_BendEast():
    model = PlainModel([6, 4])
    model.set_pos([0, math.radians(90)])
    diff =  model.pos - np.array([[0, 4], [-6, -6]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test6_BendWest():
    model = PlainModel([6, 4])
    model.set_pos([0, math.radians(-90)])
    diff =  model.pos - np.array([[0, -4], [-6, -6]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test7_BendReverse():
    model = PlainModel([6, 4])
    model.set_pos([0, math.radians(180)])
    diff =  model.pos - np.array([[0, 0], [-6, 2]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test8_Model3():
    model = PlainModel([10, 5, 3])
    model.set_pos([0, 0, math.radians(0)])
    diff =  model.pos - np.array([[0, 0, 0], [-10, -15 ,18]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test9_Model3_Bend():
    model = PlainModel([10, 5, 3])
    model.set_pos([0, 0, math.radians(90)])
    diff =  model.pos - np.array([[0, 0, 3], [-10, -15, 15]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test9_Model3_Bend_2():
    model = PlainModel([10, 5, 3])
    model.set_pos([0, 0, math.radians(-90)])
    diff =  model.pos - np.array([[0, 0, -3], [-10, -15, 15]])
    diff = diff.sum()
    assert diff < 0.01, "Model gives different results"

def test11_():
    pass

def test12_():
    pass

def test13_():
    pass

def test14_():
    pass

def test15_():
    pass

def test16_():
    pass

def test17_():
    pass

def test18_():
    pass

def test19_():
    pass

def test20_():
    pass

