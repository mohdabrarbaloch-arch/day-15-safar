"""Track projection and arc-length parameterisation."""

from __future__ import annotations

import pytest

from sim.track import Track


def straight(length: float = 100.0) -> Track:
    return Track([(0.0, 0.0), (length, 0.0)])


def test_length():
    assert straight().length == pytest.approx(100.0)


def test_needs_at_least_two_points():
    with pytest.raises(ValueError):
        Track([(0.0, 0.0)])


def test_zero_length_track_is_rejected():
    with pytest.raises(ValueError):
        Track([(0.0, 0.0), (0.0, 0.0)])


def test_project_on_centreline():
    projection = straight().project(50.0, 0.0)
    assert projection.s == pytest.approx(50.0)
    assert projection.lateral == pytest.approx(0.0)


def test_project_left_is_positive():
    assert straight().project(50.0, 2.0).lateral == pytest.approx(2.0)


def test_project_right_is_negative():
    assert straight().project(50.0, -2.0).lateral == pytest.approx(-2.0)


def test_project_clamps_past_the_end():
    assert straight().project(150.0, 0.0).s == pytest.approx(100.0)


def test_point_at():
    x, y, heading = straight().point_at(25.0)
    assert x == pytest.approx(25.0)
    assert y == pytest.approx(0.0)
    assert heading == pytest.approx(0.0)


def test_point_at_clamps():
    x, _, _ = straight().point_at(500.0)
    assert x == pytest.approx(100.0)


def test_point_at_offset():
    x, y, _ = straight().point_at_offset(10.0, 2.0)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(2.0)


def test_heading_at():
    assert straight().heading_at(10.0) == pytest.approx(0.0)


def test_remaining():
    assert straight().remaining(30.0) == pytest.approx(70.0)


def test_is_inside_respects_lane_width():
    track = straight()
    assert track.is_inside(1.0) is True
    assert track.is_inside(2.0) is False


def test_closed_track_wraps_arc_length():
    track = Track([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], closed=True)
    assert track.length == pytest.approx(40.0)
    x, _, _ = track.point_at(45.0)
    assert x == pytest.approx(5.0)


def test_curvature_is_zero_on_a_straight():
    assert straight().project(50.0, 0.0).curvature == pytest.approx(0.0, abs=1e-9)
