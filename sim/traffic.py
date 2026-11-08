"""NPC traffic using the Intelligent Driver Model.

IDM is chosen because it is the smallest model that produces both free-flow
and realistic congestion from the same three lines of maths. No hand-authored
behaviour trees, no per-vehicle scripting.
"""

from __future__ import annotations

import math

from sim.geometry import clamp


class TrafficVehicle:
    """A non-player vehicle following its predecessor with IDM."""

    def __init__(
        self,
        s: float,
        lateral: float,
        speed: float,
        desired_speed: float = 12.0,
        length: float = 4.4,
        width: float = 1.85,
        label: str = "traffic",
    ) -> None:
        self.s = s
        self.lateral = lateral
        self.speed = speed
        self.desired_speed = desired_speed
        self.length = length
        self.width = width
        self.label = label
        self.accel = 0.0

        # IDM parameters.
        self.time_headway = 1.4
        self.min_gap = 2.0
        self.comfort_accel = 1.8
        self.comfort_brake = 2.0

    # --- IDM ------------------------------------------------------

    def idm_acceleration(self, gap_to_leader: float, leader_speed: float) -> float:
        """Compute IDM acceleration for the given gap and leader speed.

        ``gap_to_leader`` is bumper-to-bumper in metres and may be ``inf``
        when there is nothing ahead.
        """
        v = max(0.0, self.speed)
        v0 = max(0.1, self.desired_speed)
        free_flow = (v / v0) ** 4

        if not math.isfinite(gap_to_leader):
            return self.comfort_accel * (1.0 - free_flow)

        gap = max(0.1, gap_to_leader)
        closing = v - leader_speed
        desired_gap = self.min_gap + max(
            0.0,
            v * self.time_headway
            + (v * closing) / (2.0 * math.sqrt(self.comfort_accel * self.comfort_brake)),
        )
        interaction = (desired_gap / gap) ** 2
        return self.comfort_accel * (1.0 - free_flow - interaction)

    def step(self, dt: float, gap_to_leader: float, leader_speed: float) -> None:
        """Advance one timestep."""
        self.accel = self.idm_acceleration(gap_to_leader, leader_speed)
        self.speed = clamp(self.speed + self.accel * dt, 0.0, self.desired_speed * 1.35)
        self.s += self.speed * dt

    # --- serialisation --------------------------------------------

    def state_at(self, s_override: float | None = None) -> dict[str, float]:
        """Compact description for frames."""
        return {
            "s": round(s_override if s_override is not None else self.s, 4),
            "lateral": round(self.lateral, 4),
            "speed": round(self.speed, 4),
        }


def bumper_gap(
    leader_s: float,
    leader_length: float,
    follower_s: float,
    follower_length: float,
) -> float:
    """Bumper-to-bumper gap between two vehicles on the same centreline."""
    centre_gap = leader_s - follower_s
    return centre_gap - 0.5 * (leader_length + follower_length)
