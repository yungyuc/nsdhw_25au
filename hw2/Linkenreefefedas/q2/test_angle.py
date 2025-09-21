import math
import pytest
import _vector as vecgeom

def test_right_angle():
    # (1,0) 與 (0,1) -> 90 度 = pi/2
    ang = vecgeom.angle_between(1.0, 0.0, 0.0, 1.0)
    assert ang == pytest.approx(math.pi/2, rel=1e-12, abs=1e-12)

def test_same_direction():
    ang = vecgeom.angle_between(2.0, 0.0, 10.0, 0.0)
    assert ang == pytest.approx(0.0, abs=1e-15)

def test_opposite_direction():
    ang = vecgeom.angle_between(1.0, 0.0, -1.0, 0.0)
    assert ang == pytest.approx(math.pi, rel=1e-12, abs=1e-12)

def test_arbitrary():
    ang = vecgeom.angle_between(1.0, 2.0, 3.0, 4.0)
    # 檢查反對稱性 & 合理範圍
    ang2 = vecgeom.angle_between(3.0, 4.0, 1.0, 2.0)
    assert ang == pytest.approx(ang2, rel=1e-12)
    assert 0.0 <= ang <= math.pi

def test_zero_vector_raises():
    with pytest.raises(Exception):
        vecgeom.angle_between(0.0, 0.0, 1.0, 0.0)
