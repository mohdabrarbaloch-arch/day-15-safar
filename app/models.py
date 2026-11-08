"""SQLAlchemy models.

Runs are append-only: a completed simulation is never mutated. That single
constraint is what makes the analytics layer cheap to add later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    """Timezone-aware UTC now, so timestamps never drift."""
    return datetime.now(UTC)


class User(Base):
    """A platform user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    scenarios: Mapped[list[Scenario]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    agents: Mapped[list[Agent]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Scenario(Base):
    """A saved world definition."""

    __tablename__ = "scenarios"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_scenario_owner_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    track_points: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    closed_track: Mapped[bool] = mapped_column(Boolean, default=False)
    lane_width: Mapped[float] = mapped_column(Float, default=3.5)
    ego_lateral: Mapped[float] = mapped_column(Float, default=0.0)
    ego_speed: Mapped[float] = mapped_column(Float, default=0.0)
    weather: Mapped[str] = mapped_column(String(16), default="dry")
    difficulty: Mapped[str] = mapped_column(String(16), default="normal")
    duration: Mapped[float] = mapped_column(Float, default=20.0)
    dt: Mapped[float] = mapped_column(Float, default=0.05)
    npcs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    obstacles: Mapped[list[Any]] = mapped_column(JSON, default=list)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped[User] = relationship(back_populates="scenarios")
    runs: Mapped[list[Run]] = relationship(back_populates="scenario", cascade="all, delete-orphan")


class Run(Base):
    """One completed simulation. Insert-only."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), index=True
    )
    agent: Mapped[str] = mapped_column(String(32), default="pursuit")
    seed: Mapped[int] = mapped_column(Integer, default=0)
    frames: Mapped[list[Any]] = mapped_column(JSON, default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    owner: Mapped[User] = relationship(back_populates="runs")
    scenario: Mapped[Scenario] = relationship(back_populates="runs")


class Agent(Base):
    """A trained Q-learning policy."""

    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    q_table: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    episodes: Mapped[int] = mapped_column(Integer, default=0)
    mean_reward: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped[User] = relationship(back_populates="agents")
