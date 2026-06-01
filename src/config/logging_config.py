"""Loguru-based logging setup — call `setup_logging()` once at process start.

Behaviour by environment:
  - development : pretty colourised console + rotating file (DEBUG default)
  - staging/prod: structured JSON to stderr + rotating file (INFO default)
  - test        : suppressed (WARNING+) to keep pytest output readable

Implementation notes:
  - JSON output uses a **custom sink** (function), not a `format=` callable.
    Loguru's `format=<callable>` returns a TEMPLATE string that loguru then
    `.format_map(record)`s — meaning raw `{` braces in JSON would explode.
    A sink, on the other hand, owns the entire output, so we can serialise
    safely with `json.dumps(...)` and write the line ourselves.
  - stdlib loggers (uvicorn, sqlalchemy, mlflow, ...) are intercepted so the
    whole process logs through a single pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import Environment, get_settings

if TYPE_CHECKING:
    from loguru import Message


_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _build_json_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Serialise a loguru record into a structured log payload."""
    payload: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record.get("exception") is not None:
        exc = record["exception"]
        payload["exception"] = {
            "type": exc.type.__name__ if exc.type else None,
            "value": str(exc.value) if exc.value else None,
        }
    extras = {k: v for k, v in record.get("extra", {}).items() if not k.startswith("_")}
    if extras:
        payload["extra"] = extras
    return payload


def _json_stderr_sink(message: Message) -> None:
    """Custom sink: write each record as a single JSON line to stderr."""
    line = json.dumps(_build_json_payload(message.record), default=str, ensure_ascii=False)
    sys.stderr.write(line + "\n")


def _json_file_sink_factory(file_path: Path) -> Callable[[Message], None]:
    """Build a custom file sink that writes JSON lines to *file_path*."""

    def sink(message: Message) -> None:
        line = json.dumps(_build_json_payload(message.record), default=str, ensure_ascii=False)
        with open(file_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return sink


class _InterceptHandler(logging.Handler):
    """Bridge stdlib logging (uvicorn, sqlalchemy, mlflow) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """Configure loguru sinks based on current settings. Idempotent."""
    settings = get_settings()
    log_settings = settings.logging
    env = settings.app.env

    logger.remove()

    # ── Test environment: keep pytest output clean ───────────────────────────
    if env == Environment.TEST:
        logger.add(sys.stderr, level="WARNING", format=_CONSOLE_FORMAT)
        return

    use_json = log_settings.format == "json" and env != Environment.DEVELOPMENT

    # ── stderr sink ──────────────────────────────────────────────────────────
    if use_json:
        logger.add(_json_stderr_sink, level=log_settings.level)
    else:
        logger.add(
            sys.stderr,
            level=log_settings.level,
            format=_CONSOLE_FORMAT,
            colorize=True,
            backtrace=True,
            diagnose=env == Environment.DEVELOPMENT,  # never leak local vars in prod
        )

    # ── File sink (rotating) ─────────────────────────────────────────────────
    log_file = settings.paths.logs_dir / f"{settings.app.name}.log"
    if use_json:
        # Custom JSON sink doesn't support rotation/retention out of the box,
        # so we delegate to loguru's built-in file sink with `serialize=True`
        # which writes one JSON-per-line in loguru's standard envelope. For
        # structured-log aggregators that's actually preferable (stable schema).
        logger.add(
            log_file,
            level=log_settings.level,
            serialize=True,
            rotation=log_settings.rotation,
            retention=log_settings.retention,
            compression="zip",
            backtrace=True,
            diagnose=False,
            enqueue=True,
        )
    else:
        logger.add(
            log_file,
            level=log_settings.level,
            format=_CONSOLE_FORMAT,
            rotation=log_settings.rotation,
            retention=log_settings.retention,
            compression="zip",
            backtrace=True,
            diagnose=env == Environment.DEVELOPMENT,
            enqueue=True,
        )

    # ── Redirect stdlib loggers (uvicorn, sqlalchemy, …) into loguru ─────────
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "sqlalchemy.engine"):
        logging.getLogger(noisy).handlers = [_InterceptHandler()]
        logging.getLogger(noisy).propagate = False

    logger.info(
        "Logging initialised | env={} | level={} | format={}",
        env.value,
        log_settings.level,
        log_settings.format,
    )
