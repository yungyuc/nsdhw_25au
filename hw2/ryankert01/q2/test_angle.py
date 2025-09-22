import math
import pytest
from _vector import angle_of_two_vectors

def test_right_angle():
    # Vectors (1,0) and (0,1) should form a 90-degree angle (π/2 radians)
    angle = angle_of_two_vectors(1, 0, 0, 1)
    assert math.isclose(angle, math.pi / 2, abs_tol=1e-9)

def test_parallel_vectors():
    # Parallel vectors should have 0 radians between them
    angle = angle_of_two_vectors(1, 1, 2, 2)
    assert math.isclose(angle, 0.0, abs_tol=1e-7)

def test_opposite_vectors():
    # Opposite vectors should have π radians between them
    angle = angle_of_two_vectors(1, 0, -1, 0)
    assert math.isclose(angle, math.pi, abs_tol=1e-9)

def test_45_degrees():
    # Vectors (1,0) and (1,1) should form a 45-degree angle (π/4 radians)
    angle = angle_of_two_vectors(1, 0, 1, 1)
    assert math.isclose(angle, math.pi / 4, abs_tol=1e-9)

def test_60_degrees():
    # Vectors (1,0) and (0.5,√3/2) should form a 60-degree angle (π/3 radians)
    angle = angle_of_two_vectors(1, 0, 0.5, math.sqrt(3)/2)
    assert math.isclose(angle, math.pi / 3, abs_tol=1e-9)

def test_zero_magnitude():
    # Should throw an error for zero magnitude vectors
    with pytest.raises(ValueError):
        angle_of_two_vectors(0, 0, 1, 0)
    
    with pytest.raises(ValueError):
        angle_of_two_vectors(1, 0, 0, 0)
