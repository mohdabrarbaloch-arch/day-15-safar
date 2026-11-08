"""FastAPI application factory, middleware and lifespan.

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.models import Scenario, User
from app.routers import agents, auth, runs, scenarios
from app.schemas import HealthResponse
from app.security import hash_password
from sim.scenario import builtin_scenarios

settings = get_settings()


# --- startup / shutdown ----------------------------------------------


def seed_demo_data() -> None:
    """Create the demo account and the built-in scenario set.

    Idempotent: safe to run on every boot, and it only ever touches rows it
    created itself.
    """
    if not settings.seed_demo or settings.is_production:
        return

    db = SessionLocal()
    try:
        demo = db.scalar(select(User).where(User.email == settings.demo_email))
        if demo is None:
            demo = User(
                email=settings.demo_email,
                full_name="Safar Demo",
                hashed_password=hash_password(settings.demo_password),
            )
            db.add(demo)
            db.commit()
            db.refresh(demo)

        existing = db.scalar(
            select(Scenario).where(Scenario.owner_id == demo.id, Scenario.is_builtin.is_(True))
        )
        if existing is None:
            for template in builtin_scenarios():
                db.add(
                    Scenario(
                        owner_id=demo.id,
                        name=template.name,
                        description=template.description,
                        track_points=[[float(x), float(y)] for x, y in template.track_points],
                        closed_track=template.closed_track,
                        lane_width=template.lane_width,
                        ego_lateral=template.ego_lateral,
                        ego_speed=template.ego_speed,
                        weather=template.weather,
                        difficulty=template.difficulty,
                        duration=template.duration,
                        dt=template.dt,
                        npcs=[
                            {
                                "s": n.s,
                                "lateral": n.lateral,
                                "speed": n.speed,
                                "desired_speed": n.desired_speed,
                                "label": n.label,
                                "length": n.length,
                                "width": n.width,
                            }
                            for n in template.npcs
                        ],
                        obstacles=[
                            {
                                "s": o.s,
                                "lateral": o.lateral,
                                "radius": o.radius,
                                "label": o.label,
                            }
                            for o in template.obstacles
                        ],
                        is_builtin=True,
                    )
                )
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate configuration, create tables, seed demo content."""
    settings.validate_for_startup()
    init_db()
    seed_demo_data()
    yield


# --- application -----------------------------------------------------


def create_app() -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Safar simulates an autonomous vehicle and reports its safety "
            "envelope: time to collision, lane deviation, acceleration and jerk, "
            "each traceable to the timestep where it peaked. It also trains a "
            "tabular Q-learning policy you can read a row at a time."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-RateLimit-Remaining"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Attach hardening headers to every response."""
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=()")
        remaining = getattr(request.state, "rate_limit_remaining", None)
        if remaining is not None:
            response.headers.setdefault("X-RateLimit-Remaining", str(remaining))
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """Turn stray domain errors into 422s rather than 500s."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    app.include_router(auth.router)
    app.include_router(scenarios.router)
    app.include_router(runs.router)
    app.include_router(agents.router)

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness and version probe."""
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            environment=settings.environment,
            simulation_core="deterministic",
        )

    @app.get("/api/meta", tags=["meta"])
    def meta() -> dict[str, object]:
        """Static description of the platform, used by the About view."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "agents": [
                {
                    "id": "pursuit",
                    "label": "Pure pursuit",
                    "description": "Baseline path follower with a speed-aware following margin.",
                },
                {
                    "id": "qlearning",
                    "label": "Q-learning",
                    "description": "Tabular policy trained in-process on the scenario's track.",
                },
            ],
            "weather": ["dry", "wet", "fog"],
            "limits": {
                "max_duration_s": settings.sim_max_duration,
                "max_frames": settings.sim_max_frames,
                "rate_limit_sim_per_minute": settings.rate_limit_sim_per_minute,
            },
            "disclaimer": (
                "A kinematic bicycle model, not a vehicle dynamics package. "
                "The composite safety score is a summary heuristic, not a certification."
            ),
        }

    # Static frontend last, so it cannot shadow the API routes.
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse("static/index.html")

    return app


app = create_app()
