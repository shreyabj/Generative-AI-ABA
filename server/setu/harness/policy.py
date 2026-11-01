"""Resilience policy: retries with jitter, circuit breakers, deadlines, fallbacks.

These are the parts of a harness that stop a flaky dependency (an STT endpoint,
an LLM endpoint) from taking the whole turn down. Nothing here knows anything
about RAG - it is deliberately generic infrastructure.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class HarnessError(RuntimeError):
    """Base class for harness-level failures."""


class StageTimeout(HarnessError):
    pass


class CircuitOpen(HarnessError):
    pass


class StageFailed(HarnessError):
    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"stage '{stage}' failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


@dataclass(slots=True)
class RetryPolicy:
    attempts: int = 2
    base_delay_ms: float = 12.0
    max_delay_ms: float = 240.0
    multiplier: float = 2.4
    jitter: float = 0.35
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def delay_for(self, attempt: int) -> float:
        raw = min(self.base_delay_ms * (self.multiplier ** max(0, attempt - 1)), self.max_delay_ms)
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread)) / 1000.0

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        return attempt < self.attempts and isinstance(exc, self.retry_on)


NO_RETRY = RetryPolicy(attempts=1)


@dataclass
class CircuitBreaker:
    """Trips after ``fail_threshold`` consecutive failures, half-opens after cooldown."""

    name: str = "default"
    fail_threshold: int = 4
    cooldown_s: float = 20.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return "half-open"
        return "open"

    def guard(self) -> None:
        if self.state == "open":
            left = self.cooldown_s - (time.monotonic() - (self._opened_at or 0.0))
            raise CircuitOpen(f"circuit '{self.name}' open for another {left:.1f}s")

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold:
            self._opened_at = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "failures": self._failures}


@dataclass
class Outcome(Generic[T]):
    """Structured result of a policy-wrapped call - never raises at the call site."""

    ok: bool
    value: T | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0
    fell_back: bool = False

    def unwrap(self, default: T) -> T:
        return self.value if (self.ok and self.value is not None) else default


async def with_deadline(coro: Awaitable[T], timeout_s: float, *, stage: str = "stage") -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise StageTimeout(f"{stage} exceeded {timeout_s * 1000:.0f}ms deadline") from exc


async def run_resilient(
    fn: Callable[[], Awaitable[T]],
    *,
    stage: str,
    timeout_s: float | None = None,
    retry: RetryPolicy | None = None,
    breaker: CircuitBreaker | None = None,
    fallbacks: Sequence[Callable[[], Awaitable[T]]] = (),
) -> Outcome[T]:
    """Run ``fn`` under retry + timeout + breaker, then try ``fallbacks`` in order."""
    retry = retry or NO_RETRY
    started = time.perf_counter()
    attempts = 0
    last_error: BaseException | None = None

    if breaker is not None:
        try:
            breaker.guard()
        except CircuitOpen as exc:
            last_error = exc
            retry = NO_RETRY
            attempts = 0

    if last_error is None:
        for attempt in range(1, retry.attempts + 1):
            attempts = attempt
            try:
                value = await (with_deadline(fn(), timeout_s, stage=stage) if timeout_s else fn())
                if breaker is not None:
                    breaker.record_success()
                return Outcome(
                    ok=True, value=value, attempts=attempt,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if breaker is not None:
                    breaker.record_failure()
                if not retry.should_retry(attempt, exc):
                    break
                await asyncio.sleep(retry.delay_for(attempt))

    for fb in fallbacks:
        try:
            value = await fb()
            return Outcome(
                ok=True, value=value, attempts=attempts, fell_back=True,
                error=f"{type(last_error).__name__}: {last_error}" if last_error else None,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    return Outcome(
        ok=False,
        error=f"{type(last_error).__name__}: {last_error}" if last_error else "unknown failure",
        attempts=attempts,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


__all__ = [
    "NO_RETRY",
    "CircuitBreaker",
    "CircuitOpen",
    "HarnessError",
    "Outcome",
    "RetryPolicy",
    "StageFailed",
    "StageTimeout",
    "run_resilient",
    "with_deadline",
]
