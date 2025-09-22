import math
import pytest
from _vector import angle2d

def test_zero_length_vector():
    with pytest.raises(Exception):
        angle2d([0,0], [1,0])
    with pytest.raises(Exception):
        angle2d([1,0], [0,0])

def test_zero_angle():
    assert math.isclose(angle2d([1,0], [2,0]), 0.0)

def test_right_angle():
    assert math.isclose(angle2d([1,0], [0,1]), math.pi/2)

def test_45_deg_angle():
    assert math.isclose(angle2d([1,0], [1,1]), math.pi/4)
