import math
import pytest
import _vector

def test_zero_length_vector():
    with pytest.raises(Exception):
        _vector.angle(0, 0, 1, 0)

def test_zero_angle():
    assert math.isclose(_vector.angle(1, 0, 2, 0), 0.0)

def test_right_angle():
    assert math.isclose(_vector.angle(1, 0, 0, 1), math.pi / 2)

def test_other_angle():
    assert math.isclose(_vector.angle(1, 0, 1, 1), math.pi / 4)