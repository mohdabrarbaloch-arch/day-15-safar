"""Kinematic bicycle model."""

from __future__ import annotations

import math

import pytest

from sim.vehicle import Vehicle, VehicleParams, VehicleState


def test_default_radius_is_half_diagonal():
    params = VehicleParams()
    assert params.radius == pytest.approx(0.5 * math.hypot(4.4, 1.85))


def test_effective_brake_scales_with_friction():
    assert VehicleParams(friction_scale=0.5).effective_brake == pytest.approx(4.0)


def test_effective_brake_has_a_floor():
    assert VehicleParams(friction_scale=0.0).effective_brake == pytest.approx(8.0 * 0.05)


def test_step_accelerates_from_rest():
    vehicle = Vehicle(VehicleParams())
    vehicle.step(3.0, 0.0, 0.05)
    assert vehicle.state.speed == pytest.approx(0.15)
    # No distance covered yet: the midpoint rule uses the pre-step speed.
    assert vehicle.state.x == pytest.approx(0.0)


def test_step_moves_once_speed_builds():
    vehicle = Vehicle(VehicleParams(), VehicleState(speed=10.0))
    vehicle.step(0.0, 0.0, 0.1)
    assert vehicle.state.x == pytest.approx(1.0)


def test_acceleration_is_clamped_to_max():
    vehicle = Vehicle(VehicleParams())
    vehicle.step(100.0, 0.0, 0.05)
    assert vehicle.state.accel == pytest.approx(3.0)


def test_braking_is_clamped_to_effective_brake():
    vehicle = Vehicle(VehicleParams(friction_scale=0.5), VehicleState(speed=10.0))
    vehicle.step(-100.0, 0.0, 0.05)
    assert vehicle.state.accel == pytest.approx(-4.0)


def test_steering_is_clamped():
    vehicle = Vehicle(VehicleParams(), VehicleState(speed=5.0))
    vehicle.step(0.0, 10.0, 0.05)
    assert vehicle.state.steer == pytest.approx(0.55)


def test_speed_never_goes_negative():
    vehicle = Vehicle(VehicleParams(), VehicleState(speed=0.1))
    vehicle.step(-8.0, 0.0, 0.5)
    assert vehicle.state.speed == 0.0


def test_speed_is_capped():
    vehicle = Vehicle(VehicleParams(max_speed=5.0), VehicleState(speed=4.9))
    vehicle.step(3.0, 0.0, 1.0)
    assert vehicle.state.speed == pytest.approx(5.0)


def test_turning_changes_heading():
    vehicle = Vehicle(VehicleParams(), VehicleState(speed=10.0))
    vehicle.step(0.0, 0.3, 0.1)
    assert vehicle.state.heading > 0.0


def test_zero_dt_is_rejected():
    vehicle = Vehicle(VehicleParams())
    with pytest.raises(ValueError):
        vehicle.step(1.0, 0.0, 0.0)


def test_braking_distance():
    vehicle = Vehicle(VehicleParams(), VehicleState(speed=10.0))
    assert vehicle.braking_distance() == pytest.approx(100.0 / 16.0)


def test_lateral_acceleration_is_zero_when_straight():
    vehicle = Vehicle(VehicleParams(), VehicleState(speed=10.0, steer=0.0))
    assert vehicle.lateral_acceleration() == pytest.approx(0.0)


def test_snapshot_has_expected_keys():
    vehicle = Vehicle(VehicleParams())
    assert set(vehicle.snapshot()) == {"x", "y", "heading", "speed", "steer", "accel"}


def test_state_copy_is_independent():
    original = VehicleState(x=1.0)
    clone = original.copy()
    clone.x = 5.0
    assert original.x == 1.0
