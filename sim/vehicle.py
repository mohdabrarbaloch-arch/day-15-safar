"""Kinematic bicycle vehicle model.

The model is intentionally the simplest one that produces believable
behaviour: a rear-axle reference point, a single equivalent front wheel,
and no tyre slip. That means it under-reports the drama of a real car at
the limit - which is exactly why the README says this is a simulation, not
a vehicle dynamics package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from sim.geometry import clamp, vehicle_radius, wrap_angle


@dataclass(frozen=True)
class VehicleParams:
    """Physical and actuator limits for a vehicle."""

    wheelbase: float = 2.70
    """Distance between axles, in metres."""

    max_accel: float = 3.0
    """Maximum forward acceleration, m/s^2."""

    max_brake: float = 8.0
    """Maximum deceleration magnitude on dry tarmac, m/s^2."""

    max_steer: float = 0.55
    """Maximum road-wheel steering angle, radians."""

    max_speed: float = 30.0
    """Speed ceiling, m/s (about 108 km/h)."""

    length: float = 4.40
    """Overall body length, metres."""

    width: float = 1.85
    """Overall body width, metres."""

    friction_scale: float = 1.0
    """Multiplier on braking authority. 1.0 dry, ~0.6 wet."""

    @property
    def radius(self) -> float:
        """Circumscribed collision radius."""
        return vehicle_radius(self.length, self.width)

    @property
    def effective_brake(self) -> float:
        """Braking authority after the friction multiplier."""
        return self.max_brake * clamp(self.friction_scale, 0.05, 1.0)


@dataclass
class VehicleState:
    """Full kinematic state of one vehicle."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    steer: float = 0.0
    accel: float = 0.0

    def copy(self) -> VehicleState:
        """Return an independent copy."""
        return replace(self)

    @property
    def velocity(self) -> tuple[float, float]:
        """World-frame velocity vector."""
        return self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)


class Vehicle:
    """A single vehicle driven by acceleration and steering commands."""

    def __init__(self, params: VehicleParams, state: VehicleState | None = None) -> None:
        self.params = params
        self.state = state if state is not None else VehicleState()

    #  integration

    def step(self, accel: float, steer: float, dt: float) -> VehicleState:
        """Advance the state by ``dt`` seconds.

        Uses the midpoint rule for position so that a turning vehicle does not
        visibly cut the corner at coarse step sizes. Acceleration and steering
        are clamped to the vehicle's physical limits before integration, so a
        policy asking for something impossible simply gets the maximum.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        p = self.params
        s = self.state

        accel = clamp(accel, -p.effective_brake, p.max_accel)
        steer = clamp(steer, -p.max_steer, p.max_steer)

        yaw_rate = (s.speed / p.wheelbase) * math.tan(steer)
        mid_heading = s.heading + yaw_rate * dt * 0.5

        new_x = s.x + s.speed * math.cos(mid_heading) * dt
        new_y = s.y + s.speed * math.sin(mid_heading) * dt
        new_heading = wrap_angle(s.heading + yaw_rate * dt)
        new_speed = clamp(s.speed + accel * dt, 0.0, p.max_speed)

        self.state = VehicleState(
            x=new_x,
            y=new_y,
            heading=new_heading,
            speed=new_speed,
            steer=steer,
            accel=accel,
        )
        return self.state

    #  derived quantities

    def yaw_rate(self) -> float:
        """Current yaw rate, rad/s."""
        s = self.state
        return (s.speed / self.params.wheelbase) * math.tan(s.steer)

    def lateral_acceleration(self) -> float:
        """Centripetal acceleration implied by the current speed and yaw rate."""
        return abs(self.state.speed * self.yaw_rate())

    def braking_distance(self) -> float:
        """Minimum stopping distance from the current speed."""
        if self.params.effective_brake <= 0.0:
            return math.inf
        return (self.state.speed**2) / (2.0 * self.params.effective_brake)

    def snapshot(self) -> dict[str, float]:
        """Compact serialisable view of the state, for frames."""
        s = self.state
        return {
            "x": round(s.x, 4),
            "y": round(s.y, 4),
            "heading": round(s.heading, 5),
            "speed": round(s.speed, 4),
            "steer": round(s.steer, 5),
            "accel": round(s.accel, 4),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        s = self.state
        return f"Vehicle(x={s.x:.2f}, y={s.y:.2f}, v={s.speed:.2f})"
