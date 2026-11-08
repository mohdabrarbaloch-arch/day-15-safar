"""Track representation: an arc-length parameterised polyline.

Working in arc length rather than nearest-point distance is what makes the
lookahead controller stable through curves. A nearest-point search jumps
forward in discrete steps; arc length moves continuously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sim.geometry import project_on_segment, signed_side, wrap_angle


@dataclass(frozen=True)
class Projection:
    """Where a point sits relative to the track centreline."""

    s: float
    """Arc length along the track at the closest point, metres."""

    lateral: float
    """Signed offset from the centreline, metres. Positive is to the left."""

    heading: float
    """Track heading at that arc length, radians."""

    curvature: float
    """Local curvature, 1/metres. Positive curves left."""


class Track:
    """An open or closed path defined by a list of points."""

    def __init__(
        self,
        points: list[tuple[float, float]],
        closed: bool = False,
        default_lane_width: float = 3.5,
    ) -> None:
        if len(points) < 2:
            raise ValueError("a track needs at least two points")

        self.points = [(float(x), float(y)) for x, y in points]
        self.closed = closed
        self.lane_width = default_lane_width

        if closed:
            self.points.append(self.points[0])

        self._segments = list(zip(self.points, self.points[1:], strict=False))
        if not self._segments:
            raise ValueError("a track needs at least one segment")

        self._cumulative: list[float] = [0.0]
        for (ax, ay), (bx, by) in self._segments:
            self._cumulative.append(self._cumulative[-1] + math.hypot(bx - ax, by - ay))

        self.length = self._cumulative[-1]
        if self.length <= 1e-9:
            raise ValueError("a track must have non-zero length")

        self._headings = [math.atan2(by - ay, bx - ax) for (ax, ay), (bx, by) in self._segments]
        self._curvatures = self._compute_curvatures()

    #  construction helpers

    def _compute_curvatures(self) -> list[float]:
        """Finite-difference curvature per segment."""
        result: list[float] = []
        count = len(self._headings)
        for i in range(count):
            if self.closed:
                prev_i = (i - 1) % count
                next_i = (i + 1) % count
            else:
                prev_i = max(0, i - 1)
                next_i = min(count - 1, i + 1)

            turn = wrap_angle(self._headings[next_i] - self._headings[prev_i])
            arc = self._cumulative[next_i + 1] - self._cumulative[prev_i]
            result.append(turn / arc if arc > 1e-9 else 0.0)
        return result

    @classmethod
    def from_centerline(
        cls,
        points: list[tuple[float, float]],
        closed: bool = False,
        lane_width: float = 3.5,
    ) -> Track:
        """Convenience constructor mirroring the API's naming."""
        return cls(points, closed=closed, default_lane_width=lane_width)

    #  projection

    def project(self, x: float, y: float) -> Projection:
        """Find the closest point on the track to ``(x, y)``.

        Scans all segments. For the tracks this project ships (tens of
        segments) that is cheaper than building a spatial index, and it removes
        a whole class of subtle bugs.
        """
        best_dist = math.inf
        best_s = 0.0
        best_lateral = 0.0
        best_index = 0

        for index, ((ax, ay), (bx, by)) in enumerate(self._segments):
            t, qx, qy, dist = project_on_segment(x, y, ax, ay, bx, by)
            if dist < best_dist - 1e-12:
                best_dist = dist
                best_s = self._cumulative[index] + t * (
                    self._cumulative[index + 1] - self._cumulative[index]
                )
                side = signed_side(x, y, ax, ay, bx, by)
                # Normalise the cross product into a true perpendicular distance.
                seg_len = math.hypot(bx - ax, by - ay)
                best_lateral = math.copysign(dist, side) if seg_len > 1e-9 else 0.0
                best_index = index

        return Projection(
            s=best_s,
            lateral=best_lateral,
            heading=self._headings[best_index],
            curvature=self._curvatures[best_index],
        )

    def point_at(self, s: float) -> tuple[float, float, float]:
        """Return ``(x, y, heading)`` at arc length ``s``."""
        if self.closed:
            s = s % self.length
        s = max(0.0, min(s, self.length))

        for index, ((ax, ay), (bx, by)) in enumerate(self._segments):
            seg_start = self._cumulative[index]
            seg_end = self._cumulative[index + 1]
            if s <= seg_end or index == len(self._segments) - 1:
                span = seg_end - seg_start
                t = (s - seg_start) / span if span > 1e-9 else 0.0
                x = ax + (bx - ax) * t
                y = ay + (by - ay) * t
                return x, y, self._headings[index]

        # Unreachable, but keeps the type checker honest.
        ax, ay = self.points[-1]
        return ax, ay, self._headings[-1]

    def point_at_offset(self, s: float, lateral: float) -> tuple[float, float, float]:
        """Return a point offset laterally from the centreline at arc length ``s``."""
        x, y, heading = self.point_at(s)
        nx, ny = -math.sin(heading), math.cos(heading)
        return x + nx * lateral, y + ny * lateral, heading

    def heading_at(self, s: float) -> float:
        """Track heading at arc length ``s``."""
        return self.point_at(s)[2]

    def remaining(self, s: float) -> float:
        """Distance left to the end of the track."""
        return max(0.0, self.length - s)

    def is_inside(self, lateral: float, margin: float = 0.0) -> bool:
        """Whether a lateral offset is still within the drivable corridor."""
        half = self.lane_width * 0.5 + margin
        return abs(lateral) <= half
