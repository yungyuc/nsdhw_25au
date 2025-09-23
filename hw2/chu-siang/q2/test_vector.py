import math
import pytest
import _vector

def test_zero_length_vector():
    with pytest.raises(ValueError):
        _vector.angle(0.0, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        _vector.angle(1.0, 0.0, 0.0, 0.0)

def test_zero_angle():
    assert _vector.angle(1.0, 0.0, 2.0, 0.0) == pytest.approx(0.0, abs=1e-12)

def test_right_angle():
    assert _vector.angle(1.0, 0.0, 0.0, 3.0) == pytest.approx(math.pi/2, rel=1e-12, abs=1e-12)

def test_opposite_angle():
    assert _vector.angle(1.0, 0.0, -1.0, 0.0) == pytest.approx(math.pi, rel=1e-12, abs=1e-12)

def test_other_angle():
    assert _vector.angle(1.0, 1.0, 1.0, 0.0) == pytest.approx(math.pi/4, rel=1e-12, abs=1e-12)

