"""Application settings (12-factor compliant) — single source of truth.

Loaded from environment variables (or a `.env` file in development). All other
modules read configuration through `get_settings()` — never directly from
`os.environ`. This keeps the contract explicit and testable.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, RedisDsn, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file: <root>/src/config/settings.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    """Deployment environment — controls behaviour like log format and DB pool size."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


def _shared_config(env_prefix: str = "") -> SettingsConfigDict:
    """Common pydantic-settings config — single source so behaviour stays consistent."""
    return SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix=env_prefix,
        extra="ignore",
        case_sensitive=False,
        validate_default=True,  # catch invalid defaults at startup, not at runtime
    )


class AppSettings(BaseSettings):
    """Top-level application metadata."""

    model_config = _shared_config("APP_")

    name: str = "stock-prediction"
    env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    version: str = "0.1.0"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION


class APISettings(BaseSettings):
    """FastAPI / Uvicorn runtime configuration."""

    model_config = _shared_config("API_")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_per_minute: int = Field(default=60, ge=1)


class DatabaseSettings(BaseSettings):
    """PostgreSQL configuration. Exposes both async and sync URLs from a single DSN.

    The `url` field accepts a plain string in env vars; Pydantic validates it
    against `PostgresDsn` at instantiation. We never call `PostgresDsn(...)` as a
    constructor — that pattern is brittle across Pydantic 2.7-2.13 because the
    type was an `Annotated` alias in 2.7-2.9 and only became a real class in 2.10+.
    """

    model_config = _shared_config("DATABASE_")

    url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@localhost:5432/stock_prediction",  # type: ignore[assignment]
    )
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout: int = Field(default=30, ge=1)
    echo: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def async_url(self) -> str:
        """Async driver URL (asyncpg) — for FastAPI request handlers and async ETL."""
        return str(self.url).replace("postgresql://", "postgresql+asyncpg://", 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_url(self) -> str:
        """Sync driver URL (psycopg2) — for Alembic migrations and CLI scripts."""
        return str(self.url).replace("postgresql://", "postgresql+psycopg2://", 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def safe_url(self) -> str:
        """Password-masked URL — safe to log."""
        url_str = str(self.url)
        if "@" not in url_str:
            return url_str
        scheme, rest = url_str.split("://", 1)
        creds, host_part = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0] if ":" in creds else creds
        return f"{scheme}://{user}:***@{host_part}"


class RedisSettings(BaseSettings):
    """Redis configuration — cache layer + (future) message broker."""

    model_config = _shared_config("REDIS_")

    url: RedisDsn = Field(default="redis://localhost:6379/0")  # type: ignore[assignment]
    cache_ttl_seconds: int = Field(default=300, ge=1)


class MLflowSettings(BaseSettings):
    """MLflow tracking server configuration."""

    model_config = _shared_config("MLFLOW_")

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "stock-prediction"
    artifact_location: str = "./mlruns"


# Path that gets resolved against PROJECT_ROOT and ensured to exist.
ResolvedPath = Annotated[Path, Field(description="Path resolved against project root")]


class PathSettings(BaseSettings):
    """Filesystem layout — all paths resolved relative to PROJECT_ROOT."""

    model_config = _shared_config()

    data_dir: ResolvedPath = Path("data")
    raw_data_dir: ResolvedPath = Path("data/raw")
    interim_data_dir: ResolvedPath = Path("data/interim")
    processed_data_dir: ResolvedPath = Path("data/processed")
    models_dir: ResolvedPath = Path("models")
    logs_dir: ResolvedPath = Path("logs")

    @field_validator(
        "data_dir",
        "raw_data_dir",
        "interim_data_dir",
        "processed_data_dir",
        "models_dir",
        "logs_dir",
        mode="after",
    )
    @classmethod
    def _resolve_against_root(cls, value: Path) -> Path:
        """Make every path absolute against PROJECT_ROOT and ensure it exists."""
        resolved = value if value.is_absolute() else (PROJECT_ROOT / value).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


class LoggingSettings(BaseSettings):
    """Loguru sink configuration."""

    model_config = _shared_config("LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"
    rotation: str = "100 MB"
    retention: str = "30 days"


class MLSettings(BaseSettings):
    """Cross-cutting ML defaults."""

    model_config = _shared_config()

    random_seed: int = 42
    default_market: Literal["jp", "us"] = "jp"
    default_horizon_days: int = Field(default=5, ge=1, le=60)


class Settings:
    """Aggregate settings object — instantiate once per process via `get_settings()`.

    Sub-configs are constructed eagerly so any missing/invalid env var fails at
    startup (fast, loud) rather than the first time the relevant section is
    touched (slow, hard to debug).
    """

    def __init__(self) -> None:
        self.app = AppSettings()
        self.api = APISettings()
        self.db = DatabaseSettings()
        self.redis = RedisSettings()
        self.mlflow = MLflowSettings()
        self.paths = PathSettings()
        self.logging = LoggingSettings()
        self.ml = MLSettings()
        self.project_root: Path = PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton Settings.

    Cached so that env vars are read exactly once. Tests can clear the cache
    with `get_settings.cache_clear()` if they need to reload.
    """
    return Settings()
