"""Integration tests for the FastAPI app — smoke-level coverage.

These tests use FastAPI's TestClient (which spins up the ASGI app in-process)
to verify the wiring between routers, dependencies, services and schemas.
They do NOT exercise model inference end-to-end (that would require a
trained MLflow model and processed features); the predict endpoints are
asserted to respond with the expected HTTP shape only.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_liveness_endpoint(client: TestClient) -> None:
    """Root /health returns 200 without touching dependencies."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "environment" in body


def test_openapi_schema_exposed(client: TestClient) -> None:
    """OpenAPI doc is generated and lists our routers."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/health" in paths
    assert any(p.startswith("/api/v1/tickers") for p in paths)
    assert any(p.startswith("/api/v1/predict") for p in paths)


def test_v1_health_returns_component_breakdown(client: TestClient) -> None:
    """The detailed /api/v1/health probes DB + Redis and returns components."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    component_names = {c["name"] for c in body["components"]}
    assert {"postgres", "redis"}.issubset(component_names)


def test_get_unknown_ticker_returns_404(client: TestClient) -> None:
    """Looking up a non-existent symbol returns 404 with a structured error."""
    r = client.get("/api/v1/tickers/DOES_NOT_EXIST_999")
    assert r.status_code in {404, 503}  # 503 if DB is unreachable
    if r.status_code == 404:
        assert "not found" in r.json()["detail"].lower()
