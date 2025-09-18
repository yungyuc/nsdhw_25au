import pytest
import math
import _vector


class TestAngle:
    def test_zero_length_vector(self):
        with pytest.raises(ValueError):
            _vector.get_angle(0.0, 0.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            _vector.get_angle(1.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            _vector.get_angle(0.0, 0.0, 0.0, 0.0)

    def test_zero_angle(self):
        result = _vector.get_angle(1.0, 0.0, 1.0, 0.0)
        assert math.isclose(result, 0.0)
        result = _vector.get_angle(2.0, 3.0, 4.0, 6.0)
        assert math.isclose(result, 0.0)

    def test_right_angle(self):
        result = _vector.get_angle(1.0, 0.0, 0.0, 1.0)
        assert math.isclose(result, math.pi / 2)
        result = _vector.get_angle(3.0, 0.0, 0.0, -2.0)
        assert math.isclose(result, math.pi / 2)

    def test_other_angle(self):
        # 45 degrees
        result = _vector.get_angle(1.0, 0.0, 1.0, 1.0)
        assert math.isclose(result, math.pi / 4)
        # 60 degrees
        result = _vector.get_angle(1.0, 0.0, 0.5, math.sqrt(3) / 2)
        assert math.isclose(result, math.pi / 3, rel_tol=1e-9)

    def test_opposite_vectors(self):
        result = _vector.get_angle(1.0, 0.0, -1.0, 0.0)
        assert math.isclose(result, math.pi)
        result = _vector.get_angle(2.0, 3.0, -4.0, -6.0)
        assert math.isclose(result, math.pi)

    def test_negative_coordinates(self):
        result = _vector.get_angle(-1.0, 0.0, 0.0, -1.0)
        assert math.isclose(result, math.pi / 2)

    def test_floating_point_precision(self):
        result = _vector.get_angle(1e-10, 0.0, 1.0, 0.0)
        assert math.isclose(result, 0.0, abs_tol=1e-9)
        result = _vector.get_angle(0.0, 1e-10, 1.0, 0.0)
        assert math.isclose(result, math.pi / 2, rel_tol=1e-9)
