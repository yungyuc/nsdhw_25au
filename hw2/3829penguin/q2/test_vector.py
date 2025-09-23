import math
import pytest
import _vector as vector_angle

def test_angle_90deg():
    angle = vector_angle.angle_between_vectors(1.0, 0.0, 0.0, 1.0)
    assert math.isclose(angle, math.pi / 2, rel_tol=1e-9)

def test_angle_0deg():
    angle = vector_angle.angle_between_vectors(1.0, 0.0, 2.0, 0.0)
    assert math.isclose(angle, 0.0, rel_tol=1e-9)

def test_angle_45deg():
    angle = vector_angle.angle_between_vectors(1.0, 0.0, 1.0, 1.0)
    assert math.isclose(angle, math.pi / 4, rel_tol=1e-9)

def test_angle_60deg():
    angle = vector_angle.angle_between_vectors(1.0, 0.0, 0.5, math.sqrt(3) / 2)
    assert math.isclose(angle, math.pi / 3, rel_tol=1e-9)

def test_zero_vector_raises():
    with pytest.raises(ValueError):
        vector_angle.angle_between_vectors(0.0, 0.0, 1.0, 0.0)
