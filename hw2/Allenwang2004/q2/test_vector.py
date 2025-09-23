import math
import pytest
import _vector

def test_angle_between():
    """Test basic angle calculation between two vectors."""
    assert math.isclose(_vector.angle_between(1, 0, 0, 1), math.pi / 2, rel_tol=1e-5)
    assert math.isclose(_vector.angle_between(1, 0, 1, 0), 0, rel_tol=1e-5)
    assert math.isclose(_vector.angle_between(1, 0, -1, 0), math.pi, rel_tol=1e-5)

def test_zero_angle():
    """Test vectors pointing in the same direction (zero angle)."""
    # Same direction vectors
    assert math.isclose(_vector.angle_between(1, 0, 2, 0), 0, rel_tol=1e-5)
    assert math.isclose(_vector.angle_between(3, 4, 6, 8), 0, rel_tol=1e-5)

def test_right_angle():
    """Test perpendicular vectors (90 degrees)."""
    # Perpendicular vectors
    assert math.isclose(_vector.angle_between(1, 0, 0, 1), math.pi / 2, rel_tol=1e-5)
    assert math.isclose(_vector.angle_between(3, 0, 0, -2), math.pi / 2, rel_tol=1e-5)

def test_other_angles():
    """Test other specific angles."""
    # 45 degrees
    assert math.isclose(_vector.angle_between(1, 0, 1, 1), math.pi / 4, rel_tol=1e-5)
    # 60 degrees  
    assert math.isclose(_vector.angle_between(1, 0, 0.5, math.sqrt(3)/2), math.pi / 3, rel_tol=1e-5)

def test_opposite_vectors():
    """Test vectors pointing in opposite directions (180 degrees)."""
    assert math.isclose(_vector.angle_between(1, 0, -1, 0), math.pi, rel_tol=1e-5)
    assert math.isclose(_vector.angle_between(1, 1, -1, -1), math.pi, rel_tol=1e-5)

if __name__ == "__main__":
    test_angle_between()
    test_zero_angle()
    test_right_angle()
    test_other_angles()
    test_opposite_vectors()
    print("All tests passed!")