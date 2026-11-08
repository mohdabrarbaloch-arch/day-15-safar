"""Safety metric collection.

The design rule: every metric records *where* it peaked, not just how big it
got. A minimum time-to-collision of 0.3 s is a different story depending on
whether it happened at 4 km/h or 60 km/h, and whether it was in a straight
line or mid-corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sim.geometry import clamp, closing_speed, time_to_collision


@dataclass
class Peak:
    """A metric's extreme value together with where it occurred."""

    value: float = 0.0
    step: int = -1
    time: float = 0.0

    def observe(self, value: float, step: int, time: float, *, maximum: bool) -> None:
        """Update the peak if ``value`` beats the current record."""
        better = value > self.value if maximum else value < self.value
        if better:
            self.value = value
            self.step = step
            self.time = time

    def to_dict(self) -> dict[str, float]:
        return {
            "value": round(self.value, 4),
            "step": self.step,
            "time": round(self.time, 3),
        }


@dataclass
class MetricsCollector:
    """Accumulates safety metrics over the course of one simulation."""

    max_frames: int = 4000

    collided: bool = False
    collision_step: int = -1
    collision_time: float = 0.0
    collision_with: str = ""
    collision_speed: float = 0.0

    min_ttc: Peak = field(default_factory=lambda: Peak(value=math.inf))
    min_gap: Peak = field(default_factory=lambda: Peak(value=math.inf))
    max_lateral_error: Peak = field(default_factory=lambda: Peak(value=0.0))
    max_accel_forward: Peak = field(default_factory=lambda: Peak(value=0.0))
    max_decel: Peak = field(default_factory=lambda: Peak(value=0.0))
    max_lateral_accel: Peak = field(default_factory=lambda: Peak(value=0.0))
    max_jerk: Peak = field(default_factory=lambda: Peak(value=0.0))

    lateral_error_sq_sum: float = 0.0
    observations: int = 0
    completed: bool = False
    progress: float = 0.0
    distance_travelled: float = 0.0

    _prev_accel: float = 0.0
    _has_prev_accel: bool = False

    #  observation

    def observe(
        self,
        *,
        step: int,
        time: float,
        lateral_error: float,
        speed: float,
        accel: float,
        lateral_accel: float,
        peers: list[dict[str, float]],
        own_x: float,
        own_y: float,
        own_radius: float,
        own_vx: float,
        own_vy: float,
        distance_delta: float,
    ) -> None:
        """Record one timestep.

        ``peers`` is a list of dicts with ``x``, ``y``, ``radius``, ``vx``,
        ``vy`` and ``label`` - everything the metric layer needs, without
        knowing what kind of object a peer is.
        """
        self.observations += 1
        self.lateral_error_sq_sum += lateral_error * lateral_error
        self.distance_travelled += max(0.0, distance_delta)

        self.max_lateral_error.observe(abs(lateral_error), step, time, maximum=True)
        self.max_lateral_accel.observe(abs(lateral_accel), step, time, maximum=True)

        if accel > 0.0:
            self.max_accel_forward.observe(accel, step, time, maximum=True)
        else:
            # Stored as a magnitude so the dashboard reads "3.2 m/s of braking"
            # rather than a double negative.
            self.max_decel.observe(-accel, step, time, maximum=True)

        if self._has_prev_accel:
            jerk = abs(accel - self._prev_accel) / 0.05
            self.max_jerk.observe(jerk, step, time, maximum=True)
        self._prev_accel = accel
        self._has_prev_accel = True

        for peer in peers:
            gap = math.hypot(peer["x"] - own_x, peer["y"] - own_y) - (own_radius + peer["radius"])
            self.min_gap.observe(gap, step, time, maximum=False)

            approach = closing_speed(
                own_x, own_y, own_vx, own_vy, peer["x"], peer["y"], peer["vx"], peer["vy"]
            )
            ttc = time_to_collision(gap, approach)
            if math.isfinite(ttc):
                self.min_ttc.observe(ttc, step, time, maximum=False)

            if gap <= 0.0 and not self.collided:
                self.collided = True
                self.collision_step = step
                self.collision_time = time
                self.collision_with = peer.get("label", "object")
                self.collision_speed = speed

    def mark_complete(self, *, progress: float) -> None:
        """Record that the ego vehicle reached the end of its route."""
        self.completed = True
        self.progress = clamp(progress, 0.0, 1.0)

    def set_progress(self, *, progress: float) -> None:
        """Record how far along the route the ego has travelled so far.

        Kept separate from ``mark_complete`` so a run that never finishes still
        reports 0.83 of the route covered rather than 0.0.
        """
        self.progress = clamp(progress, 0.0, 1.0)

    #  finalisation

    @property
    def rms_lateral_error(self) -> float:
        """Root-mean-square lateral deviation over the whole run."""
        if self.observations == 0:
            return 0.0
        return math.sqrt(self.lateral_error_sq_sum / self.observations)

    def safety_score(self) -> float:
        """Composite 0-100 score.

        Deliberately simple, deliberately documented: a crash is zero, and
        otherwise the score subtracts weighted penalties for a tight TTC,
        lane excursions, jerk and lateral acceleration. It is a summary for a
        dashboard, not a certification.
        """
        if self.collided:
            return 0.0

        ttc = self.min_ttc.value
        penalty_ttc = 0.0 if not math.isfinite(ttc) else clamp((2.0 - ttc) / 2.0, 0.0, 1.0) * 40.0
        penalty_lane = clamp(self.max_lateral_error.value / 2.0, 0.0, 1.0) * 20.0
        penalty_jerk = clamp(self.max_jerk.value / 8.0, 0.0, 1.0) * 20.0
        penalty_lat = clamp(self.max_lateral_accel.value / 6.0, 0.0, 1.0) * 20.0

        return round(
            clamp(100.0 - (penalty_ttc + penalty_lane + penalty_jerk + penalty_lat), 0.0, 100.0), 1
        )

    def finalise(self) -> dict[str, object]:
        """Produce the serialisable metrics payload."""
        min_ttc_value = self.min_ttc.value
        return {
            "collided": self.collided,
            "collision": (
                {
                    "step": self.collision_step,
                    "time": round(self.collision_time, 3),
                    "with": self.collision_with,
                    "speed_mps": round(self.collision_speed, 3),
                    "speed_kmh": round(self.collision_speed * 3.6, 1),
                }
                if self.collided
                else None
            ),
            "completed": self.completed,
            "progress": round(self.progress, 4),
            "distance_travelled_m": round(self.distance_travelled, 2),
            "min_ttc_s": None if not math.isfinite(min_ttc_value) else round(min_ttc_value, 3),
            "min_ttc_at": self.min_ttc.to_dict() if math.isfinite(min_ttc_value) else None,
            "min_gap_m": (
                None if not math.isfinite(self.min_gap.value) else round(self.min_gap.value, 3)
            ),
            "min_gap_at": self.min_gap.to_dict() if math.isfinite(self.min_gap.value) else None,
            "max_lateral_error_m": self.max_lateral_error.to_dict(),
            "rms_lateral_error_m": round(self.rms_lateral_error, 4),
            "max_accel_forward_mps2": self.max_accel_forward.to_dict(),
            "max_decel_mps2": self.max_decel.to_dict(),
            "max_lateral_accel_mps2": self.max_lateral_accel.to_dict(),
            "max_jerk_mps3": self.max_jerk.to_dict(),
            "safety_score": self.safety_score(),
            "observations": self.observations,
        }
