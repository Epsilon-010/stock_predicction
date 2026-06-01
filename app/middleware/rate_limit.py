"""Per-IP rate limiting via slowapi.

slowapi wraps FastAPI's request lifecycle with `Limiter`, which maps each
request to a key (`get_remote_address` by default) and rejects calls that
exceed the configured rate.

In production behind a reverse proxy (nginx, Cloud Run, ALB), the *real*
client IP is in `X-Forwarded-For`. We pull from that header first and fall
back to the socket address — without this, every request would look like
it came from the proxy and rate limiting wouldn't work.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.config.settings import get_settings


def _client_ip(request: Request) -> str:
    """X-Forwarded-For if present, else the direct peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "anonymous"


def build_limiter() -> Limiter:
    settings = get_settings()
    return Limiter(
        key_func=_client_ip,
        default_limits=[f"{settings.api.rate_limit_per_minute}/minute"],
    )


def install_rate_limiting(app: FastAPI) -> Limiter:
    """Attach the limiter + exception handler + middleware to `app`."""
    limiter = build_limiter()
    app.state.limiter = limiter
    # slowapi's handler signature is `(Request, RateLimitExceeded) -> Response`,
    # which is a strict subtype of Starlette's `(Request, Exception)` contract
    # but mypy can't narrow that automatically.
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)
    return limiter
