"""API package — versioned routers + cross-cutting dependencies."""

from app.api.v1 import api_router as v1_router

__all__ = ["v1_router"]
