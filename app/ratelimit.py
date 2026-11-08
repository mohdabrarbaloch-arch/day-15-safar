"""Sliding-window rate limiting, per client IP and per bucket.

In-memory and therefore per-process. The README and ARCHITECTURE both say so:
for more than one worker you want Redis behind the same interface.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class SlidingWindowLimiter:
    """A count-based sliding window keyed by an arbitrary string."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """Consume one slot for ``key``.

        Returns ``(allowed, remaining)``.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                return False, 0

            bucket.append(now)
            return True, max(0, limit - len(bucket))

    def reset(self) -> None:
        """Drop all recorded hits. Used by the test suite."""
        with self._lock:
            self._hits.clear()


def client_key(request: Request, bucket: str) -> str:
    """Build a stable limiter key from the request."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return f"{bucket}:{ip}"


# One bucket per concern so a busy simulation endpoint cannot starve login.
auth_limiter = SlidingWindowLimiter()
sim_limiter = SlidingWindowLimiter()
default_limiter = SlidingWindowLimiter()


def enforce(request: Request, bucket: str, limit: int) -> None:
    """Raise 429 when the caller has exhausted ``bucket``."""
    limiter = {
        "auth": auth_limiter,
        "sim": sim_limiter,
        "default": default_limiter,
    }.get(bucket, default_limiter)

    allowed, remaining = limiter.allow(client_key(request, bucket), limit)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )

    # Attached so the response middleware can echo it back.
    request.state.rate_limit_remaining = remaining


def reset_all() -> None:
    """Clear every bucket. Test-suite helper."""
    auth_limiter.reset()
    sim_limiter.reset()
    default_limiter.reset()
