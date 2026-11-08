"""Pure-math geometry helpers used across the simulation core.

No third-party imports here on purpose: this module is the foundation the
tests lean on, so it stays dependency-free and deterministic.
"""

from __future__ import annotations

import math

TAU = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians into the half-open interval (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, TAU)
    if wrapped < 0.0:
        wrapped += TAU
    return wrapped - math.pi


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from ``a`` to ``b``."""
    return a + (b - a) * t


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points."""
    return math.hypot(bx - ax, by - ay)


def project_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float, float]:
    """Project a point onto a line segment.

    Returns ``(t, qx, qy, dist)`` where ``t`` is the normalised position along
    the segment (0 at A, 1 at B, clamped), ``(qx, qy)`` is the projected
    point, and ``dist`` is the distance from the original point to it.
    """
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy

    if length_sq <= 1e-12:
        t = 0.0
    else:
        t = clamp(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0, 1.0)

    qx = ax + t * dx
    qy = ay + t * dy
    return t, qx, qy, math.hypot(px - qx, py - qy)


def signed_side(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Cross product of the segment direction with the vector A->P.

    Positive means the point lies to the left of the direction of travel,
    negative means right. Magnitude is twice the triangle area, so callers
    should normalise if they need a real distance.
    """
    dx = bx - ax
    dy = by - ay
    return dx * (py - ay) - dy * (px - ax)


def circles_overlap(
    x1: float,
    y1: float,
    r1: float,
    x2: float,
    y2: float,
    r2: float,
) -> bool:
    """True when two circles touch or intersect."""
    return distance(x1, y1, x2, y2) <= (r1 + r2)


def circle_gap(
    x1: float,
    y1: float,
    r1: float,
    x2: float,
    y2: float,
    r2: float,
) -> float:
    """Signed clearance between two circles: negative means overlap."""
    return distance(x1, y1, x2, y2) - (r1 + r2)


def closing_speed(
    x1: float,
    y1: float,
    vx1: float,
    vy1: float,
    x2: float,
    y2: float,
    vx2: float,
    vy2: float,
) -> float:
    """Rate at which the gap between two moving points is shrinking.

    Positive means the objects are approaching each other along their
    connecting line; negative means they are separating.
    """
    dx = x2 - x1
    dy = y2 - y1
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return 0.0
    ux = dx / norm
    uy = dy / norm
    relative_x = vx2 - vx1
    relative_y = vy2 - vy1
    return -(relative_x * ux + relative_y * uy)


def time_to_collision(gap: float, approach: float) -> float:
    """Time until contact, given a clearance and a closing rate.

    Returns ``inf`` when the objects are not closing on each other.
    """
    if approach <= 1e-6:
        return math.inf
    if gap <= 0.0:
        return 0.0
    return gap / approach


def heading_vector(heading: float) -> tuple[float, float]:
    """Unit vector pointing along ``heading``."""
    return math.cos(heading), math.sin(heading)


def left_normal(heading: float) -> tuple[float, float]:
    """Unit vector 90 degrees to the left of ``heading``."""
    return -math.sin(heading), math.cos(heading)


def vehicle_radius(length: float, width: float) -> float:
    """Circumscribed radius used for collision checks.

    Using the half-diagonal means the collision circle fully contains the
    rectangle, which makes collisions slightly conservative - the right
    direction to err for a safety tool.
    """
    return 0.5 * math.hypot(length, width)


def polyline_length(points: list[tuple[float, float]]) -> float:
    """Total length of an open polyline."""
    total = 0.0
    for (ax, ay), (bx, by) in zip(points, points[1:], strict=False):
        total += math.hypot(bx - ax, by - ay)
    return total


def is_finite(*values: float) -> bool:
    """True when every supplied value is a finite real number."""
    return all(math.isfinite(v) for v in values)
