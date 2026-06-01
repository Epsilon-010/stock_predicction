# ─────────────────────────────────────────────────────────────────────────────
# Minimal MLflow server image with Postgres backend support.
#
# The official `ghcr.io/mlflow/mlflow` image ships without `psycopg2`, so we
# build our own slim variant that can talk to the project's Postgres instance.
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# psycopg2 needs the postgres client lib at runtime; curl is for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG MLFLOW_VERSION=2.19.0
RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg2-binary>=2.9" \
        "boto3>=1.34"

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:5000/health || exit 1

# `command:` from docker-compose.yml supplies the actual server flags so that
# backend/artifact paths can be changed without rebuilding the image.
ENTRYPOINT ["mlflow"]
CMD ["server", "--host", "0.0.0.0", "--port", "5000"]
