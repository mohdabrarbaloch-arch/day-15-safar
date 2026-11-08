"""Driving policies.

Two are shipped, both deterministic:

- ``PurePursuitController`` - the baseline. Follows the centreline with a
  speed-dependent lookahead, brakes for whatever is ahead.
- ``TabularPolicy`` - the learned one, backed by the Q-table from ``agent.py``.

Keeping them behind the same tiny interface (``act`` -> ``Action``) is what lets
the engine stay ignorant of which one it is running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sim.track import Projection, Track
from sim.vehicle import VehicleState


@dataclass(frozen=True)
class Action:
    """A command pair for one timestep."""

    accel: float
    steer: float


# Discrete action grid shared by the agent and the registry, so a learned
# policy and the engine can never disagree about what action index means what.
ACCEL_CHOICES: tuple[float, ...] = (-2.0, 0.0, 2.0)
STEER_CHOICES: tuple[float, ...] = (-0.25, 0.0, 0.25)
ACTION_COUNT = len(ACCEL_CHOICES) * len(STEER_CHOICES)


def decode_action(index: int) -> Action:
    """Turn a flat action index into an ``Action``."""
    if not 0 <= index < ACTION_COUNT:
        raise ValueError(f"action index {index} out of range 0..{ACTION_COUNT - 1}")
    accel = ACCEL_CHOICES[index // len(STEER_CHOICES)]
    steer = STEER_CHOICES[index % len(STEER_CHOICES)]
    return Action(accel=accel, steer=steer)


def encode_action(accel: float, steer: float) -> int:
    """Turn an ``Action`` back into its index (nearest grid cell)."""
    a_index = min(range(len(ACCEL_CHOICES)), key=lambda i: abs(ACCEL_CHOICES[i] - accel))
    s_index = min(range(len(STEER_CHOICES)), key=lambda i: abs(STEER_CHOICES[i] - steer))
    return a_index * len(STEER_CHOICES) + s_index


class PurePursuitController:
    """Path-following baseline with simple longitudinal speed control."""

    def __init__(
        self,
        *,
        lookahead_gain: float = 0.75,
        min_lookahead: float = 6.0,
        max_lookahead: float = 16.0,
        target_speed: float = 12.0,
        obstacle_margin: float = 1.8,
        follow_headway: float = 0.95,
        accel_gain: float = 2.6,
    ) -> None:
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.target_speed = target_speed
        self.obstacle_margin = obstacle_margin
        self.follow_headway = follow_headway
        self.accel_gain = accel_gain

    def _lookahead(self, speed: float) -> float:
        return max(self.min_lookahead, min(self.max_lookahead, speed * self.lookahead_gain))

    def act(
        self,
        *,
        vehicle_state: VehicleState,
        projection: Projection,
        track: Track,
        wheelbase: float,
        hazards: list[dict[str, float]],
        speed_cap: float,
        brake_capability: float,
    ) -> Action:
        """Produce one command pair.

        Lateral: pure pursuit toward a point ``lookahead`` metres ahead along
        the track. Longitudinal: cruise at the target speed, reduced to
        whatever still lets the vehicle decelerate to the nearest hazard's
        speed before it arrives.

        The braking constraint is ``v <= v_lead + 2-a-d``, not ``v <= 2-a-d``.
        That distinction matters: you do not have to stop for a moving car, you
        only have to slow down to its speed. Getting this wrong is the classic
        way an ACC implementation ends up braking far too late.
        """
        speed = vehicle_state.speed
        lookahead = self._lookahead(max(speed, 1.0))

        target_x, target_y, _ = track.point_at(projection.s + lookahead)
        dx = target_x - vehicle_state.x
        dy = target_y - vehicle_state.y

        # Rotate the target into vehicle frame, then apply the pure-pursuit law.
        cos_h = math.cos(vehicle_state.heading)
        sin_h = math.sin(vehicle_state.heading)
        local_x = dx * cos_h + dy * sin_h
        local_y = -dx * sin_h + dy * cos_h

        distance_sq = local_x * local_x + local_y * local_y
        if distance_sq <= 1e-6:
            steer = 0.0
        else:
            curvature = 2.0 * local_y / distance_sq
            steer = math.atan(wheelbase * curvature)

        desired_speed = min(self.target_speed, speed_cap)
        for hazard in hazards:
            gap = hazard["gap"]
            if gap <= 0.0:
                desired_speed = 0.0
                break

            leader_speed = hazard.get("speed", 0.0)
            # The gap we insist on keeping grows with speed - a standstill
            # margin plus a headway proportional to how fast we're going.
            # Without this the controller is happy to close to zero gap at
            # speed and only then brake, which is exactly how the baseline
            # used to crash.
            required_clearance = self.obstacle_margin + self.follow_headway * max(
                speed, leader_speed
            )
            usable = gap - required_clearance

            if usable <= 0.0:
                # Already inside the margin: match the leader, don't accelerate.
                allowed = leader_speed
            else:
                allowed = math.sqrt(
                    max(0.0, leader_speed * leader_speed + 2.0 * brake_capability * usable)
                )
            desired_speed = min(desired_speed, allowed)

        error = desired_speed - speed
        accel = max(-6.0, min(3.0, error * self.accel_gain))
        return Action(accel=accel, steer=steer)


class TabularPolicy:
    """A greedy policy over a learned Q-table."""

    def __init__(self, q_table: dict[str, list[float]]) -> None:
        if not q_table:
            raise ValueError("a TabularPolicy needs a non-empty Q-table")
        self.q_table = q_table

    def act(self, state_key: str, *, fallback_accel: float = 0.0) -> Action:
        """Pick the highest-valued action for ``state_key``.

        Unknown states fall back to a neutral action rather than raising: a
        policy that crashes on an unseen state is not a policy.
        """
        values = self.q_table.get(state_key)
        if values is None:
            return Action(accel=fallback_accel, steer=0.0)
        best_index = max(range(len(values)), key=lambda i: values[i])
        return decode_action(best_index)
