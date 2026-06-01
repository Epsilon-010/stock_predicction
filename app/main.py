"""FastAPI application factory.

Lifespan handles **process-wide** setup/teardown — logging, opening DB+Redis
pools, closing them on shutdown. Per-request resources (DB sessions, the
Redis client) are scoped through the DI providers in `app/api/dependencies.py`.

`create_app()` is exported as a factory rather than a module-level
`app = FastAPI(...)` so tests can create isolated instances without
hammering the singleton settings cache.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1 import api_router as v1_router
from app.middleware import install_rate_limiting
from app.schemas.health import HealthResponse
from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.utils.db import dispose_engines
from src.utils.redis_client import close_redis_pool


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: configure logging. Shutdown: dispose engines + pools."""
    setup_logging()
    settings = get_settings()
    logger.info(
        "API starting | env={} | version={} | port={}",
        settings.app.env.value,
        settings.app.version,
        settings.api.port,
    )
    try:
        yield
    finally:
        logger.info("API shutting down")
        await dispose_engines()
        await close_redis_pool()


def create_app() -> FastAPI:
    """Build a configured FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Production-grade ML pipeline for TSE stock-direction prediction.",
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ───────────────────────────────────────────────────────
    install_rate_limiting(app)

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(v1_router, prefix=settings.api.prefix)

    # Minimal root-level health for load balancers — doesn't hit deps.
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Cheap liveness probe (no DB / Redis call)",
    )
    async def liveness() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=settings.app.version,
            environment=settings.app.env.value,
            components=[],
        )

    return app


# Uvicorn entry point: `uvicorn app.main:app ...`
app = create_app()
