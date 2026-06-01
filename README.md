# Stock Prediction — TSE (Tokyo Stock Exchange)

Production-grade ML pipeline that ingests ~3,900 Japanese equities + ETFs +
indices, engineers technical features, trains direction-classifier models with
rigorous walk-forward validation, and serves predictions through a FastAPI
inference layer.

> **Scope honesty.** This is a portfolio / learning project. The objective is to
> showcase a complete, professional ML engineering pipeline — not to claim alpha
> over public markets. Predicting raw OHLCV-only direction at 5 days is hard;
> expect AUC in the 0.52–0.58 range, not 0.90.

---

## Architecture

```
                          ┌──────────────────────────┐
                          │   Raw .txt files (bronze)│
                          │   data/raw/jp/...        │
                          └────────────┬─────────────┘
                                       │ Prefect ETL
                                       ▼
                          ┌──────────────────────────┐
                          │   Parquet (silver)       │
                          │   partitioned by year    │
                          │   data/interim/...       │
                          └────────────┬─────────────┘
                                       │ COPY / SQLAlchemy
                                       ▼
        ┌─────────────────────────────────────────────────────────┐
        │       PostgreSQL + TimescaleDB (gold)                   │
        │   ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
        │   │  ohlcv      │  │  features    │  │  predictions  │  │
        │   │ (hypertable)│  │ (hypertable) │  │  (hypertable) │  │
        │   └─────────────┘  └──────────────┘  └───────────────┘  │
        └────────────┬─────────────────────┬──────────────────────┘
                     │                     │
            training │                     │ queries
                     ▼                     ▼
        ┌─────────────────────┐   ┌────────────────────────┐
        │  Training pipeline  │   │  FastAPI inference     │
        │  - LogisticReg      │   │  /api/v1/predict       │
        │  - XGBoost          │   │  /api/v1/backtest      │
        │  - Walk-forward CV  │   │  /api/v1/tickers       │
        │  - MLflow tracking  │   └───────────┬────────────┘
        └─────────────────────┘               │
                                              ▼
                                      ┌──────────────────┐
                                      │   Redis cache    │
                                      │ (latest prices,  │
                                      │  predictions)    │
                                      └──────────────────┘
```

**Medallion data layout** (bronze / silver / gold) — same pattern Databricks,
Snowflake and most modern data platforms use.

---

## Tech stack

| Layer            | Tooling                                                    |
| ---------------- | ---------------------------------------------------------- |
| Language         | Python 3.13, `uv` package manager                          |
| API              | FastAPI, Uvicorn, slowapi (rate limiting)                  |
| Validation       | Pydantic v2, pydantic-settings                             |
| Database         | PostgreSQL 16 + TimescaleDB extension                      |
| ORM / migrations | SQLAlchemy 2.0 (hybrid async + sync), Alembic              |
| Cache            | Redis 7                                                    |
| ETL / Orchestr.  | Prefect 2, Polars, PyArrow                                 |
| ML               | scikit-learn, XGBoost, LightGBM                            |
| Experiment track | MLflow                                                     |
| Logging          | Loguru (JSON in staging/prod)                              |
| Tests            | pytest, pytest-asyncio, httpx                              |
| Quality          | Ruff, Black, Mypy, pre-commit                              |
| Container        | Docker, docker-compose                                     |

---

## Project layout

```
stock_predicction/
├── pyproject.toml          # deps, ruff/black/mypy/pytest config
├── docker-compose.yml      # postgres + redis + mlflow + api
├── Dockerfile              # multi-stage build for the API
├── Makefile                # one-line dev commands
├── .env.example            # env-var contract
│
├── src/                    # ML / data engineering code
│   ├── config/             # settings.py, logging, model_config
│   ├── utils/              # db.py, redis_client.py
│   ├── etl/                # extract / transform / load
│   ├── features/           # feature engineering
│   ├── models/             # training, inference
│   ├── evaluation/         # metrics, backtest, walk-forward
│   ├── tracking/           # MLflow helpers
│   └── orchestration/      # Prefect flows
│
├── app/                    # FastAPI inference layer
│   ├── main.py
│   ├── api/v1/             # endpoint modules
│   ├── schemas/            # pydantic request/response models
│   ├── services/           # business logic
│   └── middleware/
│
├── data/
│   ├── raw/jp/             # bronze: original .txt files
│   ├── interim/            # silver: parquet
│   └── processed/          # gold: feature matrices
│
├── alembic/                # DB migrations
├── tests/                  # unit + integration
├── notebooks/              # EDA, prototyping
└── scripts/                # one-shot CLI utilities
```

---

## Quick start

```bash
# 1. Install dependencies (creates .venv automatically)
make dev-install

# 2. Copy and edit environment variables
cp .env.example .env

# 3. Bring up the infrastructure (postgres + redis + mlflow)
make up

# 4. Run database migrations
make db-migrate

# 5. Ingest raw .txt files → parquet → postgres
make ingest

# 6. Train all enabled models (logreg + xgboost)
make train

# 7. Start the API
make serve
# → http://localhost:8000/docs
```

Other useful targets — run `make help` for the full list.

---

## What's intentionally rigorous

- **Walk-forward validation only.** No k-fold on time series — that's instant
  data leakage. Train/val/test are time-ordered with an embargo gap.
- **Survivorship bias acknowledged.** The dataset only contains currently-listed
  tickers; backtest results are biased upward. Noted in the model card.
- **No look-ahead features.** Features at time `t` use only data observable at
  `t` — verified by unit tests on the feature pipeline.
- **Transaction costs in backtest.** 5 bps round-trip cost; small but realistic.
- **Honest baseline.** "Always predict up" is computed as the floor every model
  must beat before any claim of skill.

---

## Status

🚧 **Foundation phase.** Configuration, infrastructure and architecture are in
place. ETL → features → training → API endpoints are the next milestones.

---

## License

MIT — see [LICENSE](LICENSE) (TODO).
