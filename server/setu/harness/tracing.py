"""Span tracing and the latency budget allocator.

Two ideas do most of the work in SETU's latency story:

1. **Spans.** Every stage opens a span. Spans nest, carry attributes and are
   returned to the client, so the UI can draw a real waterfall instead of a
   single opaque "took 143 ms".

2. **The budget.** A turn is handed a wall-clock deadline. Before any *optional*
   stage runs, the orchestrator asks the budget whether it can afford it. If not,
   the stage is skipped and recorded in ``skipped``. The pipeline is therefore
   *anytime*: quality degrades gracefully but the deadline is respected.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .contracts import BudgetReport, SpanModel


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class Span:
    name: str
    depth: int
    start_ms: float
    duration_ms: float = 0.0
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_model(self, origin: float) -> SpanModel:
        return SpanModel(
            name=self.name,
            depth=self.depth,
            start_ms=round(self.start_ms - origin, 3),
            duration_ms=round(self.duration_ms, 3),
            status=self.status,
            attrs=self.attrs,
        )


class LatencyBudget:
    """Wall-clock deadline with affordability queries."""

    def __init__(self, total_ms: float, origin_ms: float | None = None) -> None:
        self.total_ms = float(total_ms)
        self.origin_ms = origin_ms if origin_ms is not None else now_ms()
        self.skipped: list[str] = []
        self.degraded = False

    @property
    def used_ms(self) -> float:
        return now_ms() - self.origin_ms

    @property
    def remaining_ms(self) -> float:
        return self.total_ms - self.used_ms

    def expired(self) -> bool:
        return self.remaining_ms <= 0.0

    def can_afford(self, cost_ms: float, *, reserve_ms: float = 0.0) -> bool:
        return self.remaining_ms - reserve_ms >= cost_ms

    def skip(self, stage: str, cost_ms: float) -> None:
        self.skipped.append(stage)
        self.degraded = True

    def deadline_s(self, cap_s: float | None = None) -> float:
        """Remaining budget as seconds, for use with asyncio timeouts."""
        left = max(self.remaining_ms, 1.0) / 1000.0
        return min(left, cap_s) if cap_s else left

    def report(self) -> BudgetReport:
        used = self.used_ms
        return BudgetReport(
            total_ms=round(self.total_ms, 3),
            used_ms=round(used, 3),
            remaining_ms=round(self.total_ms - used, 3),
            slo_met=used <= self.total_ms,
            skipped=list(self.skipped),
            degraded=self.degraded,
        )


class Trace:
    """Collects a nested span tree for one turn."""

    def __init__(self, turn_id: str | None = None) -> None:
        self.turn_id = turn_id or uuid.uuid4().hex[:12]
        self.origin_ms = now_ms()
        self.spans: list[Span] = []
        self._depth = 0
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        sp = Span(name=name, depth=self._depth, start_ms=now_ms(), attrs=dict(attrs))
        self.spans.append(sp)
        self._depth += 1
        try:
            yield sp
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            sp.status = "error"
            sp.attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._depth -= 1
            sp.duration_ms = now_ms() - sp.start_ms

    def mark(self, name: str, cost_ms: float = 0.0, status: str = "skipped", **attrs: Any) -> None:
        """Record a zero-duration span (used for skipped / cached stages)."""
        sp = Span(name=name, depth=self._depth, start_ms=now_ms(), duration_ms=cost_ms, status=status)
        sp.attrs.update(attrs)
        self.spans.append(sp)

    def event(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "at_ms": round(now_ms() - self.origin_ms, 3), **payload})

    @property
    def elapsed_ms(self) -> float:
        return now_ms() - self.origin_ms

    def models(self) -> list[SpanModel]:
        return [s.to_model(self.origin_ms) for s in self.spans]

    def by_name(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.spans:
            out[s.name] = out.get(s.name, 0.0) + s.duration_ms
        return out


__all__ = ["LatencyBudget", "Span", "Trace", "now_ms"]
