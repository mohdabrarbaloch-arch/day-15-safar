"""Declarative scenario definitions.

A scenario is a plain data object describing a world: the track, the ego
vehicle's start, the NPC traffic, static obstacles, weather and a duration
budget. Everything the engine needs is here and nothing else, which is what
makes a run reproducible from a stored row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from sim.geometry import is_finite
from sim.track import Track
from sim.vehicle import VehicleParams

Weather = Literal["dry", "wet", "fog"]
Difficulty = Literal["easy", "normal", "hard"]

# Braking authority multiplier per surface. Wet tarmac really does cost you
# roughly a third of your stopping power; fog costs grip only slightly but
# hurts visibility, which is handled by the controller's speed cap.
FRICTION_BY_WEATHER: dict[str, float] = {
    "dry": 1.0,
    "wet": 0.62,
    "fog": 0.90,
}

SPEED_CAP_BY_WEATHER: dict[str, float] = {
    "dry": 30.0,
    "wet": 22.0,
    "fog": 16.0,
}


@dataclass
class NPCConfig:
    """One NPC vehicle placed at a lateral offset from the centreline."""

    s: float
    lateral: float = 0.0
    speed: float = 10.0
    desired_speed: float = 12.0
    label: str = "traffic"
    length: float = 4.4
    width: float = 1.85

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not is_finite(self.s, self.lateral, self.speed, self.desired_speed):
            errors.append("NPC has non-finite values")
        if self.s < 0:
            errors.append(f"NPC {self.label}: s must be >= 0")
        if self.speed < 0 or self.desired_speed <= 0:
            errors.append(f"NPC {self.label}: speeds must be positive")
        return errors


@dataclass
class ObstacleConfig:
    """A static hazard on the road: a cone, a parked car, debris."""

    s: float
    lateral: float = 0.0
    radius: float = 0.6
    label: str = "obstacle"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not is_finite(self.s, self.lateral, self.radius):
            errors.append("Obstacle has non-finite values")
        if self.radius <= 0:
            errors.append(f"Obstacle {self.label}: radius must be positive")
        return errors


@dataclass
class Scenario:
    """A complete, validated world definition."""

    name: str
    track_points: list[tuple[float, float]]
    closed_track: bool = False
    lane_width: float = 3.5
    ego_lateral: float = 0.0
    ego_speed: float = 0.0
    weather: Weather = "dry"
    difficulty: Difficulty = "normal"
    duration: float = 20.0
    dt: float = 0.05
    npcs: list[NPCConfig] = field(default_factory=list)
    obstacles: list[ObstacleConfig] = field(default_factory=list)
    description: str = ""

    #  validation

    def validate(self) -> list[str]:
        """Return a list of problems. Empty means the scenario is runnable."""
        errors: list[str] = []

        if not self.name.strip():
            errors.append("scenario needs a name")
        if len(self.track_points) < 2:
            errors.append("track needs at least two points")
        if len(self.track_points) > 500:
            errors.append("track is limited to 500 points")
        if not (0.01 <= self.dt <= 0.2):
            errors.append("dt must be between 0.01 and 0.2 seconds")
        if not (1.0 <= self.duration <= 120.0):
            errors.append("duration must be between 1 and 120 seconds")
        if self.lane_width <= 0.5:
            errors.append("lane_width must exceed 0.5 m")
        if self.weather not in FRICTION_BY_WEATHER:
            errors.append(f"unknown weather '{self.weather}'")
        if self.difficulty not in ("easy", "normal", "hard"):
            errors.append(f"unknown difficulty '{self.difficulty}'")

        for x, y in self.track_points:
            if not is_finite(x, y):
                errors.append("track contains a non-finite point")
                break

        if abs(self.ego_lateral) > self.lane_width:
            errors.append("ego_lateral must be within one lane width")

        for npc in self.npcs:
            errors.extend(npc.validate())
        for obstacle in self.obstacles:
            errors.extend(obstacle.validate())

        return errors

    #  derived

    @property
    def friction_scale(self) -> float:
        """Braking-authority multiplier for this scenario's weather."""
        return FRICTION_BY_WEATHER[self.weather]

    @property
    def speed_cap(self) -> float:
        """Weather-limited speed ceiling the controller respects."""
        return SPEED_CAP_BY_WEATHER[self.weather]

    @property
    def step_count(self) -> int:
        """Number of integration steps this scenario will run."""
        return int(math.floor(self.duration / self.dt))

    def build_track(self) -> Track:
        """Instantiate the runtime track."""
        return Track(
            self.track_points, closed=self.closed_track, default_lane_width=self.lane_width
        )

    def build_vehicle_params(self) -> VehicleParams:
        """Instantiate vehicle parameters, friction-adjusted for the surface."""
        return VehicleParams(friction_scale=self.friction_scale, max_speed=self.speed_cap)


#  built-in scenarios


def straight_road(length: float = 300.0, lane_width: float = 3.5) -> list[tuple[float, float]]:
    """A flat straight, long enough that traffic has room to run.

    The default is 300 m rather than 160 m because NPCs stop when they reach
    the end of the track. On a short road a slow lead vehicle parks at the
    finish within a 20-second scenario and the ego rear-ends a stationary car -
    which is a bug in the scenario, not a finding about the policy.
    """
    return [(0.0, 0.0), (length, 0.0)]


def gentle_curve() -> list[tuple[float, float]]:
    """A long right-hand sweep - enough curvature to expose poor lookahead."""
    points: list[tuple[float, float]] = []
    radius = 90.0
    for i in range(25):
        angle = -0.55 * (i / 24.0)
        points.append((radius * math.sin(angle), radius * (1.0 - math.cos(angle))))
    return points


def s_curve() -> list[tuple[float, float]]:
    """An S-bend: left, then right. The classic lane-keeping stress test."""
    points: list[tuple[float, float]] = []
    for i in range(41):
        t = i / 40.0
        x = t * 180.0
        y = 14.0 * math.sin(t * 2.0 * math.pi)
        points.append((x, y))
    return points


def chicane() -> list[tuple[float, float]]:
    """A tightening pair of opposite lock. Wide enough to be drivable."""
    return [
        (0.0, 0.0),
        (30.0, 0.0),
        (42.0, 3.5),
        (56.0, -3.5),
        (68.0, 0.0),
        (115.0, 0.0),
    ]


def builtin_scenarios() -> list[Scenario]:
    """The scenario set seeded into a fresh database.

    Four tracks x a few traffic configurations. Deliberately small: each one
    exists to demonstrate a specific failure mode of a driving policy rather
    than to be a library.
    """
    return [
        Scenario(
            name="Straight - open road",
            track_points=straight_road(140.0),
            ego_speed=8.0,
            weather="dry",
            difficulty="easy",
            duration=18.0,
            description=(
                "The control case. A straight, empty road, short enough to finish. "
                "Any policy that cannot hold a lane here has a bug, not a tuning "
                "problem."
            ),
        ),
        Scenario(
            name="Straight - slow lead vehicle",
            track_points=straight_road(),
            ego_speed=14.0,
            weather="dry",
            difficulty="normal",
            duration=20.0,
            npcs=[NPCConfig(s=55.0, lateral=0.0, speed=6.0, desired_speed=6.5, label="lead-car")],
            description=(
                "A lead car doing 6 m/s while the ego closes at 14. The textbook "
                "car-following test: brake early, settle at a sane gap, and watch "
                "min TTC stay comfortably above a second."
            ),
        ),
        Scenario(
            name="Straight - stalled car, wet road",
            track_points=straight_road(),
            ego_speed=16.0,
            weather="wet",
            difficulty="hard",
            duration=22.0,
            obstacles=[ObstacleConfig(s=90.0, lateral=0.0, radius=1.1, label="stalled-car")],
            description=(
                "A stationary obstacle 90 m out, on wet tarmac where braking is "
                "38% weaker. The interesting number is the stopping distance the "
                "controller actually needed versus the 90 m it had."
            ),
        ),
        Scenario(
            name="Gentle curve - lane keeping",
            track_points=gentle_curve(),
            ego_speed=12.0,
            weather="dry",
            difficulty="normal",
            duration=20.0,
            description=(
                "A long right-hand sweep. Tests whether the lookahead controller "
                "scales with speed instead of cutting the inside of the corner."
            ),
        ),
        Scenario(
            name="S-curve - two lane changes",
            track_points=s_curve(),
            ego_speed=11.0,
            weather="dry",
            difficulty="normal",
            duration=24.0,
            description=(
                "Left then right. Steering has to reverse cleanly, which is where "
                "sluggish controllers show up as lateral-error spikes."
            ),
        ),
        Scenario(
            name="S-curve - traffic in fog",
            track_points=s_curve(),
            ego_speed=13.0,
            weather="fog",
            difficulty="hard",
            duration=24.0,
            npcs=[
                NPCConfig(s=40.0, lateral=0.0, speed=5.0, desired_speed=5.5, label="lead-car"),
                NPCConfig(s=75.0, lateral=1.7, speed=9.0, desired_speed=9.5, label="adjacent-car"),
            ],
            description=(
                "Two cars in reduced visibility with the speed ceiling cut to "
                "16 m/s. The interesting output is not whether it survives - it's "
                "how much margin it keeps."
            ),
        ),
        Scenario(
            name="Chicane - rapid reversals",
            track_points=chicane(),
            lane_width=4.0,
            ego_speed=10.0,
            weather="dry",
            difficulty="hard",
            duration=14.0,
            description=(
                "Opposite lock, twice. Steering-rate limits and jerk are the whole "
                "story here; a comfortable score means the controller smoothed the "
                "input instead of chasing the path vertex by vertex."
            ),
        ),
    ]
