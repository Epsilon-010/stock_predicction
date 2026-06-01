-- ─────────────────────────────────────────────────────────────────────────────
-- Initial Postgres extensions + companion databases.
-- Runs once when the postgres volume is first created.
-- ─────────────────────────────────────────────────────────────────────────────

-- Time-series superpowers (hypertables, continuous aggregates, compression).
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Used by Alembic for case-insensitive ticker lookups and ranking features.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Separate database for MLflow's backend store — keeps experiment metadata
-- isolated from the application schema.
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')
\gexec
