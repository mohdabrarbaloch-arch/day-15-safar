"""Simulation run routes."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import rate_limit_dep
from app.config import get_settings
from app.database import get_db
from app.models import Agent, Run, Scenario, User
from app.schemas import RunDetail, RunRequest, RunSummary
from app.security import get_current_user
from sim.engine import SimulationEngine
from sim.scenario import NPCConfig, ObstacleConfig
from sim.scenario import Scenario as SimScenario

router = APIRouter(prefix="/api/runs", tags=["runs"])
settings = get_settings()

SIM_LIMIT = settings.rate_limit_sim_per_minute
DEFAULT_LIMIT = settings.rate_limit_default_per_minute


def to_sim_scenario(row: Scenario) -> SimScenario:
    """Convert a stored scenario row into a runnable simulation scenario."""
    return SimScenario(
        name=row.name,
        track_points=[(float(p[0]), float(p[1])) for p in row.track_points],
        closed_track=row.closed_track,
        lane_width=row.lane_width,
        ego_lateral=row.ego_lateral,
        ego_speed=row.ego_speed,
        weather=row.weather,
        difficulty=row.difficulty,
        duration=row.duration,
        dt=row.dt,
        npcs=[NPCConfig(**npc) for npc in (row.npcs or [])],
        obstacles=[ObstacleConfig(**obstacle) for obstacle in (row.obstacles or [])],
        description=row.description,
    )


def _readable_scenario(db: Session, scenario_id: int, user: User) -> Scenario:
    row = db.get(Scenario, scenario_id)
    if row is None or (row.owner_id != user.id and not row.is_builtin):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    return row


@router.post("", response_model=RunDetail, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RunDetail:
    """Simulate a scenario and persist the result.

    The run is stored insert-only. Frames and metrics are both returned so the
    UI can play the replay without a second request.
    """
    rate_limit_dep(request, "sim", SIM_LIMIT)

    scenario_row = _readable_scenario(db, payload.scenario_id, current_user)
    scenario = to_sim_scenario(scenario_row)

    errors = scenario.validate()
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Scenario failed validation", "errors": errors},
        )

    q_table = None
    if payload.agent == "qlearning":
        if payload.agent_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="agent_id is required when agent is 'qlearning'",
            )
        agent_row = db.get(Agent, payload.agent_id)
        if agent_row is None or agent_row.owner_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        q_table = agent_row.q_table

    engine = SimulationEngine(
        max_frames=settings.sim_max_frames,
        max_duration=settings.sim_max_duration,
    )

    try:
        result = engine.run(scenario, agent=payload.agent, q_table=q_table)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    run = Run(
        owner_id=current_user.id,
        scenario_id=scenario_row.id,
        agent=payload.agent,
        seed=payload.seed,
        frames=result.frames,
        metrics=result.metrics,
        duration=result.duration,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return RunDetail(
        id=run.id,
        scenario_id=run.scenario_id,
        agent=run.agent,
        seed=run.seed,
        duration=run.duration,
        created_at=run.created_at,
        metrics=run.metrics,
        frames=run.frames,
        track_points=result.track_points,
        lane_width=result.lane_width,
    )


@router.get("", response_model=list[RunSummary])
def list_runs(
    request: Request,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Run]:
    """List the caller's recent runs, newest first."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    limit = max(1, min(limit, 100))
    stmt = (
        select(Run)
        .where(Run.owner_id == current_user.id)
        .order_by(Run.created_at.desc(), Run.id.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt))


@router.get("/{run_id}", response_model=RunDetail)
def get_run(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RunDetail:
    """Fetch one run, with its frames."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    run = db.get(Run, run_id)
    if run is None or run.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    scenario_row = db.get(Scenario, run.scenario_id)
    return RunDetail(
        id=run.id,
        scenario_id=run.scenario_id,
        agent=run.agent,
        seed=run.seed,
        duration=run.duration,
        created_at=run.created_at,
        metrics=run.metrics,
        frames=run.frames,
        track_points=scenario_row.track_points if scenario_row else [],
        lane_width=scenario_row.lane_width if scenario_row else 3.5,
    )


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_run(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete one of the caller's runs."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    run = db.get(Run, run_id)
    if run is None or run.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    db.delete(run)
    db.commit()


@router.get("/{run_id}/summary", response_model=RunSummary)
def run_summary(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Run:
    """Metrics only - no frames. Cheap enough to poll."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    run = db.get(Run, run_id)
    if run is None or run.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


_ = math  # keeps the import list honest if the math below is trimmed
