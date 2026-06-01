# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build for the FastAPI inference service.
#   stage 1 (builder): install deps with uv into a virtual env
#   stage 2 (runtime): slim image with only the venv and source
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.13

# ── Stage 1 ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Build-time deps for compiling C extensions (xgboost, lightgbm, asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager). Pinned to a known-good 0.5.x release.
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy only manifests first to leverage layer cache
COPY pyproject.toml ./
COPY uv.lock* ./

# Resolve & install dependencies into /app/.venv
RUN uv venv /app/.venv \
    && . /app/.venv/bin/activate \
    && uv pip install --no-cache .

# ── Stage 2 ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

# Runtime libs only (no compilers in the final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime safety
RUN groupadd --system --gid 1001 app \
    && useradd  --system --uid 1001 --gid app --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app app/ /app/app/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
