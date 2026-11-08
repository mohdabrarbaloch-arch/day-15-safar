"""Application settings, loaded from the environment.

The production guard here is deliberate: a weak JWT secret is a startup
failure, not a warning in a log nobody reads.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_JWT_SECRET = "dev-only-insecure-secret-change-me-please-0000000000"


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Application
    environment: str = Field(default="development")
    app_name: str = "Safar"
    app_version: str = "1.0.0"

    # Security
    jwt_secret: str = Field(default=DEV_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=1440, ge=5, le=60 * 24 * 30)

    # Database
    database_url: str = "sqlite:///./safar.db"

    # CORS
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # Rate limiting
    rate_limit_auth_per_minute: int = Field(default=10, ge=1)
    rate_limit_sim_per_minute: int = Field(default=30, ge=1)
    rate_limit_default_per_minute: int = Field(default=120, ge=1)

    # Simulation
    sim_dt: float = Field(default=0.05, gt=0.0, le=0.2)
    sim_max_duration: float = Field(default=120.0, gt=0.0, le=600.0)
    sim_max_frames: int = Field(default=4000, ge=100, le=100_000)

    # Demo data
    seed_demo: bool = True
    demo_email: str = "demo@safar.pk"
    demo_password: str = "safar-demo-2026"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("environment")
    @classmethod
    def _normalise_environment(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in ("development", "staging", "production", "test"):
            raise ValueError("environment must be development, staging, production or test")
        return normalised

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def validate_for_startup(self) -> None:
        """Fail fast on misconfiguration that would be unsafe in production."""
        if not self.is_production:
            return

        if self.jwt_secret == DEV_JWT_SECRET:
            raise RuntimeError(
                "JWT_SECRET is still the development default. Refusing to start in production."
            )
        if len(self.jwt_secret) < 32:
            raise RuntimeError("JWT_SECRET must be at least 32 characters in production.")
        if "*" in self.allowed_origins:
            raise RuntimeError("CORS_ORIGINS cannot contain '*' in production.")
        if self.seed_demo:
            # Not fatal, but surprising enough to say out loud.
            raise RuntimeError("SEED_DEMO must be false in production.")


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
