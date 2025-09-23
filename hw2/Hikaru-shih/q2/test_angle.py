import math
import pytest
import vector_angle

def test_right_angle():
    assert math.isclose(vector_angle.angle(1, 0, 0, 1), math.pi/2)

def test_same_vector():
    assert math.isclose(vector_angle.angle(1, 1, 2, 2), 0.0)

def test_opposite_vectors():
    assert math.isclose(vector_angle.angle(1, 0, -1, 0), math.pi)

def test_arbitrary():
    result = vector_angle.angle(1, 2, 3, 4)
    expected = math.acos((1*3 + 2*4) / (math.sqrt(1**2+2**2) * math.sqrt(3**2+4**2)))
    assert math.isclose(result, expected)
