"""Safar application package."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.config import get_settings

__all__ = ["rate_limit_dep", "get_settings"]

_settings = get_settings()


def rate_limit_dep(request: Request, bucket: str, limit: int) -> None:
    """Module-level indirection so routers can import it without a cycle.

    ``app.ratelimit`` imports nothing from the routers, but keeping this
    wrapper here means routers never need to know whether limiting is
    in-memory or backed by Redis.
    """
    from app.ratelimit import enforce

    enforce(request, bucket, limit)


def raise_not_configured() -> None:
    """Placeholder guard used when a feature is intentionally disabled."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="This feature is not configured on this deployment.",
    )
