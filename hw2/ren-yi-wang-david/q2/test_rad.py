import math
import _vector
import pytest
def test_angle_orthogonal():
    # (1, 0) and (0, 1) => 90 degrees => pi/2 radians
    assert math.isclose(_vector.angle_between_vectors(1, 0, 0, 1), math.pi/2, rel_tol=1e-6)

def test_angle_opposite():
    # (1, 0) and (-1, 0) => 180 degrees => pi radians
    assert math.isclose(_vector.angle_between_vectors(1, 0, -1, 0), math.pi, rel_tol=1e-6)

def test_angle_zero_vector():
    # One zero vector => angle = 0 by design
    assert math.isclose(_vector.angle_between_vectors(0, 0, 1, 1), 0.0, rel_tol=1e-6)

def test_angle_zero_length_vectors():
    # Both vectors are zero length (invalid input)
    with pytest.raises(ValueError):
        _vector.angle_between_vectors(0, 0, 0, 0)
    

def test_angle_45_degrees():
    # (1, 0) and (1, 1) => 45 degrees => pi/4 radians
    assert math.isclose(_vector.angle_between_vectors(1, 0, 1, 1), math.pi/4, rel_tol=1e-6)

def test_angle_30_degrees():
    # (sqrt(3), 1) and (2, 0) => 30 degrees => pi/6 radians
    assert math.isclose(_vector.angle_between_vectors(math.sqrt(3), 1, 2, 0), math.pi/6, rel_tol=1e-6)