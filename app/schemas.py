"""Pydantic v2 request/response contracts.

Every field is bounded. This module is the input-sanitisation layer - nothing
reaches the simulation engine without passing through here first.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# --- auth ------------------------------------------------------------


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=120)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, value: str) -> str:
        if value.isdigit() or value.isalpha():
            raise ValueError("password must mix letters and at least one other character")
        return value


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


# --- scenarios -------------------------------------------------------


class NPCPayload(BaseModel):
    s: float = Field(ge=0.0, le=5000.0)
    lateral: float = Field(default=0.0, ge=-10.0, le=10.0)
    speed: float = Field(default=10.0, ge=0.0, le=60.0)
    desired_speed: float = Field(default=12.0, gt=0.0, le=60.0)
    label: str = Field(default="traffic", max_length=40)
    length: float = Field(default=4.4, gt=0.5, le=30.0)
    width: float = Field(default=1.85, gt=0.5, le=5.0)


class ObstaclePayload(BaseModel):
    s: float = Field(ge=0.0, le=5000.0)
    lateral: float = Field(default=0.0, ge=-10.0, le=10.0)
    radius: float = Field(default=0.6, gt=0.05, le=5.0)
    label: str = Field(default="obstacle", max_length=40)


class ScenarioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    track_points: list[tuple[float, float]] = Field(min_length=2, max_length=500)
    closed_track: bool = False
    lane_width: float = Field(default=3.5, gt=0.5, le=20.0)
    ego_lateral: float = Field(default=0.0, ge=-10.0, le=10.0)
    ego_speed: float = Field(default=0.0, ge=0.0, le=60.0)
    weather: Literal["dry", "wet", "fog"] = "dry"
    difficulty: Literal["easy", "normal", "hard"] = "normal"
    duration: float = Field(default=20.0, ge=1.0, le=120.0)
    dt: float = Field(default=0.05, ge=0.01, le=0.2)
    npcs: list[NPCPayload] = Field(default_factory=list, max_length=20)
    obstacles: list[ObstaclePayload] = Field(default_factory=list, max_length=40)

    @field_validator("track_points")
    @classmethod
    def _finite_points(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        import math

        for x, y in value:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("track points must be finite")
            if abs(x) > 100_000 or abs(y) > 100_000:
                raise ValueError("track points are out of range")
        return value


class ScenarioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    track_points: list[Any]
    closed_track: bool
    lane_width: float
    ego_lateral: float
    ego_speed: float
    weather: str
    difficulty: str
    duration: float
    dt: float
    npcs: list[Any]
    obstacles: list[Any]
    is_builtin: bool
    created_at: datetime


# --- runs ------------------------------------------------------------


class RunRequest(BaseModel):
    scenario_id: int = Field(gt=0)
    agent: Literal["pursuit", "qlearning"] = "pursuit"
    agent_id: int | None = Field(default=None, gt=0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scenario_id: int
    agent: str
    seed: int
    duration: float
    created_at: datetime
    metrics: dict[str, Any]


class RunDetail(RunSummary):
    frames: list[Any]
    track_points: list[Any] = Field(default_factory=list)
    lane_width: float = 3.5


# --- agents ----------------------------------------------------------


class TrainingRequest(BaseModel):
    name: str = Field(default="safar-agent", min_length=1, max_length=120)
    scenario_id: int = Field(gt=0)
    episodes: int = Field(default=300, ge=10, le=2000)
    steps_per_episode: int = Field(default=120, ge=20, le=600)
    alpha: float = Field(default=0.22, gt=0.0, le=1.0)
    gamma: float = Field(default=0.92, gt=0.0, le=1.0)
    seed: int = Field(default=7, ge=0, le=2**31 - 1)


class AgentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    episodes: int
    mean_reward: float
    created_at: datetime
    summary: dict[str, Any]


class AgentDetail(AgentSummary):
    policy: list[dict[str, Any]] = Field(default_factory=list)


# --- misc ------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    simulation_core: str
