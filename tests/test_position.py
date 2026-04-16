"""Tests for Position dataclass and geometry helpers."""

import math

from spectra.environment.position import Position


class TestPositionDistance:
    def test_2d_distance_345_triangle(self):
        a = Position(0.0, 0.0)
        b = Position(3.0, 4.0)
        assert math.isclose(a.distance_to(b), 5.0)

    def test_2d_distance_symmetric(self):
        a = Position(1.0, 2.0)
        b = Position(4.0, 6.0)
        assert math.isclose(a.distance_to(b), b.distance_to(a))

    def test_same_position_zero_distance(self):
        a = Position(5.0, 5.0)
        assert a.distance_to(a) == 0.0

    def test_3d_distance_when_z_provided(self):
        a = Position(0.0, 0.0, z=0.0)
        b = Position(1.0, 2.0, z=2.0)
        assert math.isclose(a.distance_to(b), 3.0)

    def test_2d_ignores_z_when_one_is_none(self):
        a = Position(0.0, 0.0)
        b = Position(3.0, 4.0, z=100.0)
        assert math.isclose(a.distance_to(b), 5.0)


class TestPositionBearing:
    def test_bearing_east(self):
        a = Position(0.0, 0.0)
        b = Position(1.0, 0.0)
        assert math.isclose(a.bearing_to(b), 0.0, abs_tol=1e-10)

    def test_bearing_north(self):
        a = Position(0.0, 0.0)
        b = Position(0.0, 1.0)
        assert math.isclose(a.bearing_to(b), math.pi / 2)

    def test_bearing_west(self):
        a = Position(0.0, 0.0)
        b = Position(-1.0, 0.0)
        assert math.isclose(abs(a.bearing_to(b)), math.pi)


class TestPositionAngle:
    def test_angle_2d(self):
        a = Position(0.0, 0.0)
        b = Position(1.0, 1.0)
        assert math.isclose(a.angle_to(b), math.pi / 4)

    def test_elevation_angle_with_z(self):
        a = Position(0.0, 0.0, z=0.0)
        b = Position(100.0, 0.0, z=100.0)
        assert math.isclose(a.elevation_to(b), math.pi / 4)

    def test_elevation_returns_none_without_z(self):
        a = Position(0.0, 0.0)
        b = Position(100.0, 0.0)
        assert a.elevation_to(b) is None


class TestPositionDefaults:
    def test_z_defaults_to_none(self):
        p = Position(1.0, 2.0)
        assert p.z is None

    def test_z_can_be_set(self):
        p = Position(1.0, 2.0, z=3.0)
        assert p.z == 3.0
