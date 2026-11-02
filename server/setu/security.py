"""Request-level security: API token, session scoping, rate limiting.

* **API token** (``SETU_API_TOKEN``). When set, every mutating route and every route
  that spends a paid third-party quota (LLM, ElevenLabs) requires
  ``Authorization: Bearer <token>`` or ``X-SETU-Token``. Compared in constant time.
  Unset means open - acceptable on localhost, and ``/api/v1/settings`` says so.
* **Session scoping** (``X-SETU-Session``). The browser generates a random id once and
  sends it on every request. Conversations and research traces are stored under it and
  only returned to the same session. It is an ownership boundary for a demo, not
  authentication: anyone holding the id can read that session's history.
* **Rate limiting.** An in-process token bucket per client IP and route group. It is
  per-process, so behind several workers the effective limit multiplies; a shared store
  (Redis) is the production answer.
"""

from __future__ import annotations

import hmac
import re
import threading
import time

from fastapi import Header, HTTPException, Request

from .config import settings

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def require_token(authorization: str | None = Header(default=None),
                  x_setu_token: str | None = Header(default=None)) -> None:
    want = settings.api_token
    if not want:
        return
    got = x_setu_token or ""
    if authorization and authorization.lower().startswith("bearer "):
        got = authorization[7:].strip()
    if not got or not hmac.compare_digest(got.encode(), want.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid API token")


def session_id(x_setu_session: str | None = Header(default=None)) -> str:
    if x_setu_session and _SESSION_RE.fullmatch(x_setu_session):
        return x_setu_session
    return "anonymous"


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, group: str, per_minute: int) -> bool:
        now = time.monotonic()
        rate = per_minute / 60.0
        with self._lock:
            tokens, last = self._buckets.get((key, group), (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - last) * rate)
            if tokens < 1.0:
                self._buckets[(key, group)] = (tokens, now)
                return False
            self._buckets[(key, group)] = (tokens - 1.0, now)
            if len(self._buckets) > 20_000:
                self._buckets.clear()
            return True


limiter = RateLimiter()


def rate_limit(group: str, per_minute: int | None = None):
    def dep(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        if not limiter.hit(ip, group, per_minute or settings.rate_limit_per_min):
            raise HTTPException(status_code=429, detail=f"rate limit exceeded for {group}; slow down")

    return dep


__all__ = ["RateLimiter", "limiter", "rate_limit", "require_token", "session_id"]
