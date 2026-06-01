# ─────────────────────────────────────────────────────────────────────────────
# Convenience targets — works on Linux/macOS/WSL; on Windows use `make` via Git Bash.
# Run `make help` for the list.
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
.PHONY: help install dev-install lint format type test test-fast cov \
        up down logs ps clean db-migrate db-revision db-reset \
        ingest train predict serve mlflow-ui pre-commit

PYTHON      := uv run python
UV          := uv
COMPOSE     := docker compose

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────────────
install: ## Install runtime dependencies.
	$(UV) sync

dev-install: ## Install runtime + dev dependencies and pre-commit hooks.
	$(UV) sync --extra dev
	$(UV) run pre-commit install

# ── Code quality ─────────────────────────────────────────────────────────────
lint: ## Run ruff (lint) + mypy (types).
	$(UV) run ruff check src app tests
	$(UV) run mypy src app

format: ## Auto-format code (ruff format + ruff --fix).
	$(UV) run ruff format src app tests
	$(UV) run ruff check --fix src app tests

type: ## Type-check only.
	$(UV) run mypy src app

pre-commit: ## Run all pre-commit hooks on all files.
	$(UV) run pre-commit run --all-files

# ── Tests ────────────────────────────────────────────────────────────────────
test: ## Run the full test suite.
	$(UV) run pytest

test-fast: ## Run only unit tests (skip integration).
	$(UV) run pytest -m "not integration and not slow"

cov: ## Open the HTML coverage report.
	$(UV) run pytest && python -m webbrowser htmlcov/index.html

# ── Infrastructure (docker) ──────────────────────────────────────────────────
up: ## Start postgres + redis + mlflow.
	$(COMPOSE) up -d postgres redis mlflow

down: ## Stop the stack (keeps volumes).
	$(COMPOSE) down

logs: ## Tail logs from all services.
	$(COMPOSE) logs -f --tail=100

ps: ## Show container status.
	$(COMPOSE) ps

clean: ## Stop and DESTROY volumes (you will lose data).
	$(COMPOSE) down -v

# ── Database migrations ──────────────────────────────────────────────────────
db-migrate: ## Apply all pending Alembic migrations.
	$(UV) run alembic upgrade head

db-revision: ## Auto-generate a new migration. Usage: make db-revision m="add prices table"
	$(UV) run alembic revision --autogenerate -m "$(m)"

db-reset: ## Drop and recreate the dev database. DESTRUCTIVE.
	$(COMPOSE) exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS stock_prediction;"
	$(COMPOSE) exec postgres psql -U postgres -c "CREATE DATABASE stock_prediction;"
	$(MAKE) db-migrate

# ── Pipeline entry points ────────────────────────────────────────────────────
ingest: ## Run the ETL flow (raw .txt → parquet → postgres).
	$(PYTHON) -m src.orchestration.flows ingest

features: ## Build the feature matrix (silver → processed).
	$(PYTHON) -m src.orchestration.flows features

train: ## Train all enabled models from model_config.yaml.
	$(PYTHON) -m src.orchestration.flows train

predict: ## Run batch predictions for the latest available date.
	$(PYTHON) -m src.orchestration.flows predict

# ── Application ──────────────────────────────────────────────────────────────
serve: ## Start the FastAPI server with hot reload (development).
	$(UV) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui: ## Open the local MLflow UI in a browser.
	@python -m webbrowser http://localhost:5000
