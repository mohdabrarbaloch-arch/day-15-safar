"""Scenario CRUD routes.

Built-in scenarios belong to whoever first seeded them but are readable by
everyone - they're demo content, not private data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import rate_limit_dep
from app.config import get_settings
from app.database import get_db
from app.models import Scenario, User
from app.schemas import ScenarioCreate, ScenarioOut
from app.security import get_current_user

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])
settings = get_settings()

DEFAULT_LIMIT = settings.rate_limit_default_per_minute


def _owned_or_builtin(db: Session, scenario_id: int, user: User) -> Scenario:
    """Fetch a scenario the caller may read."""
    scenario = db.get(Scenario, scenario_id)
    if scenario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if scenario.owner_id != user.id and not scenario.is_builtin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    return scenario


@router.get("", response_model=list[ScenarioOut])
def list_scenarios(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Scenario]:
    """List the caller's scenarios plus every built-in one."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)
    stmt = (
        select(Scenario)
        .where((Scenario.owner_id == current_user.id) | (Scenario.is_builtin.is_(True)))
        .order_by(Scenario.is_builtin.desc(), Scenario.id.asc())
    )
    return list(db.scalars(stmt))


@router.get("/{scenario_id}", response_model=ScenarioOut)
def get_scenario(
    scenario_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Scenario:
    """Fetch one scenario by id."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)
    return _owned_or_builtin(db, scenario_id, current_user)


@router.post("", response_model=ScenarioOut, status_code=status.HTTP_201_CREATED)
def create_scenario(
    payload: ScenarioCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Scenario:
    """Create a new scenario."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    duplicate = db.scalar(
        select(Scenario).where(
            Scenario.owner_id == current_user.id,
            Scenario.name == payload.name.strip(),
        )
    )
    if duplicate is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You already have a scenario with that name",
        )

    scenario = Scenario(
        owner_id=current_user.id,
        name=payload.name.strip(),
        description=payload.description.strip(),
        track_points=[[float(x), float(y)] for x, y in payload.track_points],
        closed_track=payload.closed_track,
        lane_width=payload.lane_width,
        ego_lateral=payload.ego_lateral,
        ego_speed=payload.ego_speed,
        weather=payload.weather,
        difficulty=payload.difficulty,
        duration=payload.duration,
        dt=payload.dt,
        npcs=[npc.model_dump() for npc in payload.npcs],
        obstacles=[obstacle.model_dump() for obstacle in payload.obstacles],
        is_builtin=False,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return scenario


@router.delete(
    "/{scenario_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_scenario(
    scenario_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete one of the caller's own scenarios. Built-ins are protected."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    scenario = db.get(Scenario, scenario_id)
    if scenario is None or scenario.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if scenario.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Built-in scenarios cannot be deleted"
        )

    db.delete(scenario)
    db.commit()
