"""Shared pytest fixtures.

Environment variables are set *before* the app is imported, because
``app.config`` reads them at import time and caches the result.
"""

from __future__ import annotations

import os
import pathlib

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./test_safar.db"
os.environ["SEED_DEMO"] = "true"
os.environ["JWT_SECRET"] = "test-secret-that-is-long-enough-for-the-suite-0000"

# Start from a clean database so the seeded demo account is predictable.
_DB = pathlib.Path("test_safar.db")
if _DB.exists():
    _DB.unlink()

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's lifespan (tables + demo seed) running."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Clear the in-memory limiters around every test.

    Without this, a suite that signs in a dozen times would trip the 10/min
    auth budget and start failing for the wrong reason.
    """
    from app import ratelimit

    ratelimit.reset_all()
    yield
    ratelimit.reset_all()


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    """Bearer headers for the seeded demo account."""
    response = client.post(
        "/api/auth/login",
        json={"email": "demo@safar.pk", "password": "safar-demo-2026"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
