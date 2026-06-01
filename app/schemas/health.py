"""Health-check response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded", "down"]


class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health of the API and its dependencies."""

    status: HealthStatus
    version: str
    environment: str
    components: list[ComponentHealth] = Field(default_factory=list)
