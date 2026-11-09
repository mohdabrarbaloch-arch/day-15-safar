"""Geometry helpers - the foundation everything else leans on."""

from __future__ import annotations

import math

import pytest

from sim.geometry import (
    circle_gap,
    circles_overlap,
    clamp,
    closing_speed,
    distance,
    heading_vector,
    is_finite,
    left_normal,
    lerp,
    polyline_length,
    project_on_segment,
    signed_side,
    time_to_collision,
    vehicle_radius,
    wrap_angle,
)


def test_wrap_angle_identity():
    assert wrap_angle(0.0) == pytest.approx(0.0)


def test_wrap_angle_wraps_positive():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)


def test_wrap_angle_wraps_negative():
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)


def test_wrap_angle_stays_bounded():
    for k in range(-6, 7):
        wrapped = wrap_angle(k * math.pi + 0.3)
        assert -math.pi - 1e-9 <= wrapped <= math.pi + 1e-9


def test_clamp():
    assert clamp(5.0, 0.0, 3.0) == 3.0
    assert clamp(-5.0, 0.0, 3.0) == 0.0
    assert clamp(1.5, 0.0, 3.0) == 1.5


def test_lerp():
    assert lerp(0.0, 10.0, 0.25) == pytest.approx(2.5)


def test_distance():
    assert distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


def test_project_on_segment_midpoint():
    t, qx, qy, dist = project_on_segment(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert t == pytest.approx(0.5)
    assert qx == pytest.approx(5.0)
    assert qy == pytest.approx(0.0)
    assert dist == pytest.approx(3.0)


def test_project_on_segment_clamps_past_the_end():
    t, _, _, _ = project_on_segment(50.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert t == pytest.approx(1.0)


def test_project_on_segment_handles_degenerate_segment():
    t, _, _, dist = project_on_segment(1.0, 1.0, 2.0, 2.0, 2.0, 2.0)
    assert t == 0.0
    assert dist == pytest.approx(math.hypot(1.0, 1.0))


def test_signed_side_left_is_positive():
    assert signed_side(0.0, 1.0, 0.0, 0.0, 10.0, 0.0) > 0


def test_signed_side_right_is_negative():
    assert signed_side(0.0, -1.0, 0.0, 0.0, 10.0, 0.0) < 0


def test_circles_overlap():
    assert circles_overlap(0.0, 0.0, 1.0, 1.5, 0.0, 1.0) is True
    assert circles_overlap(0.0, 0.0, 1.0, 3.0, 0.0, 1.0) is False


def test_circle_gap_sign():
    assert circle_gap(0.0, 0.0, 1.0, 3.0, 0.0, 1.0) == pytest.approx(1.0)
    assert circle_gap(0.0, 0.0, 1.0, 1.0, 0.0, 1.0) == pytest.approx(-1.0)


def test_closing_speed_approaching_is_positive():
    assert closing_speed(0.0, 0.0, 1.0, 0.0, 10.0, 0.0, -1.0, 0.0) == pytest.approx(2.0)


def test_closing_speed_separating_is_negative():
    assert closing_speed(0.0, 0.0, -1.0, 0.0, 10.0, 0.0, 1.0, 0.0) == pytest.approx(-2.0)


def test_time_to_collision():
    assert time_to_collision(10.0, 2.0) == pytest.approx(5.0)
    assert time_to_collision(10.0, 0.0) == math.inf
    assert time_to_collision(0.0, 5.0) == 0.0


def test_heading_vector():
    hx, hy = heading_vector(0.0)
    assert hx == pytest.approx(1.0)
    assert hy == pytest.approx(0.0)


def test_left_normal_is_perpendicular():
    nx, ny = left_normal(0.0)
    assert nx == pytest.approx(0.0)
    assert ny == pytest.approx(1.0)


def test_vehicle_radius_is_half_diagonal():
    assert vehicle_radius(4.0, 3.0) == pytest.approx(2.5)


def test_polyline_length():
    assert polyline_length([(0.0, 0.0), (3.0, 4.0), (3.0, 10.0)]) == pytest.approx(11.0)


def test_is_finite():
    assert is_finite(1.0, 2.0) is True
    assert is_finite(1.0, math.inf) is False
