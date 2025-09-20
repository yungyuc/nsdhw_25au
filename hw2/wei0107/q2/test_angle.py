import math
import pytest
import importlib

_vector = importlib.import_module("_vector")

def almost_equal(a, b, eps=1e-12):
    # 比較兩個浮點數是否近似相等
    return abs(a - b) < eps

def test_zero_angle():
    # (1,1) 與 (2,2)：角度為 0 度
    assert almost_equal(_vector.angle_between(1, 1, 2, 2), 0.0)

def test_right_angle():
    # (1,0) 與 (0,1)：角度為 90 度
    assert almost_equal(_vector.angle_between(1, 0, 0, 1), math.pi / 2)

def test_opposite_angle():
    # (1,0) 與 (-1,0)：角度為 180 度
    assert almost_equal(_vector.angle_between(1, 0, -1, 0), math.pi)

def test_other_angle():
    # (1,0) 與 (1,1)：角度為 45 度
    assert almost_equal(_vector.angle_between(1, 0, 1, 1), math.pi / 4)

def test_zero_vector_raises():
    # (0,0) 與 (1,0)：應該拋出例外
    with pytest.raises(Exception):
        _vector.angle_between(0, 0, 1, 0)

def test_tuple_api():
    # 使用 tuple 作為輸入
    assert almost_equal(_vector.angle_between_xy((1, 0), (0, 1)), math.pi / 2)