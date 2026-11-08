"""Agent training routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import rate_limit_dep
from app.config import get_settings
from app.database import get_db
from app.models import Agent, Scenario, User
from app.routers.runs import to_sim_scenario
from app.schemas import AgentDetail, AgentSummary, TrainingRequest
from app.security import get_current_user
from sim.agent import TrainingConfig, train

router = APIRouter(prefix="/api/agents", tags=["agents"])
settings = get_settings()

SIM_LIMIT = settings.rate_limit_sim_per_minute
DEFAULT_LIMIT = settings.rate_limit_default_per_minute


@router.get("", response_model=list[AgentSummary])
def list_agents(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Agent]:
    """List the caller's trained agents."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)
    stmt = (
        select(Agent)
        .where(Agent.owner_id == current_user.id)
        .order_by(Agent.created_at.desc(), Agent.id.desc())
    )
    return list(db.scalars(stmt))


@router.post("/train", response_model=AgentDetail, status_code=status.HTTP_201_CREATED)
def train_agent(
    payload: TrainingRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgentDetail:
    """Train a tabular Q-learning policy against a scenario's track.

    Training runs in-process and is fully seeded, so the same request always
    produces the same Q-table and the same reward curve.
    """
    rate_limit_dep(request, "sim", SIM_LIMIT)

    scenario_row = db.get(Scenario, payload.scenario_id)
    if scenario_row is None or (
        scenario_row.owner_id != current_user.id and not scenario_row.is_builtin
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")

    scenario = to_sim_scenario(scenario_row)
    errors = scenario.validate()
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Scenario failed validation", "errors": errors},
        )

    config = TrainingConfig(
        episodes=payload.episodes,
        steps_per_episode=payload.steps_per_episode,
        alpha=payload.alpha,
        gamma=payload.gamma,
        seed=payload.seed,
        dt=scenario.dt,
    )

    try:
        track = scenario.build_track()
        result = train(config, track, lane_width=scenario.lane_width, speed_cap=scenario.speed_cap)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    summary = result.summary()
    agent = Agent(
        owner_id=current_user.id,
        name=payload.name.strip(),
        q_table=result.q_table,
        summary=summary,
        episodes=result.episodes,
        mean_reward=result.mean_reward_last_50,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    return AgentDetail(
        id=agent.id,
        name=agent.name,
        episodes=agent.episodes,
        mean_reward=agent.mean_reward,
        created_at=agent.created_at,
        summary=agent.summary,
        policy=result.policy_grid(),
    )


@router.get("/{agent_id}", response_model=AgentDetail)
def get_agent(
    agent_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgentDetail:
    """Fetch one agent, including its full greedy policy grid."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    agent = db.get(Agent, agent_id)
    if agent is None or agent.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    from sim.agent import TrainingResult

    stand_in = TrainingResult(q_table=agent.q_table or {}, episodes=agent.episodes)
    return AgentDetail(
        id=agent.id,
        name=agent.name,
        episodes=agent.episodes,
        mean_reward=agent.mean_reward,
        created_at=agent.created_at,
        summary=agent.summary,
        policy=stand_in.policy_grid(),
    )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_agent(
    agent_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete one of the caller's agents."""
    rate_limit_dep(request, "default", DEFAULT_LIMIT)

    agent = db.get(Agent, agent_id)
    if agent is None or agent.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    db.delete(agent)
    db.commit()
