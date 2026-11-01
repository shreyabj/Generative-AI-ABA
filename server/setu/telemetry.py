"""Latency telemetry.

The task asks for P50 / P70 / P100 over a reasonable number of queries rather than a
single best-case run, so latency is a first-class, always-on measurement here rather
than something a benchmark script bolts on afterwards. Every turn served by the API -
including the ones you make by clicking around the UI - lands in this ring buffer, and
``GET /api/v1/metrics`` reports the live distribution.

Three separate distributions are tracked, because collapsing them would be misleading:

``pipeline_ms``  Transcript in -> verified answer out. Routing, embedding, dense and
                 sparse retrieval, fusion, parent expansion, reranking, composition and
                 all guardrails. **This is the 200 ms SLO.**
``stt_ms``       The third-party speech-to-text round trip.
``e2e_ms``       Everything, wall clock, including STT.

P100 is reported as a true maximum, not a P99.9 in disguise, and the per-stage table is
also percentiled - a P50 breakdown tells you where the time goes, and the P100 breakdown
tells you which stage is responsible for your worst case. Those are usually different
stages, which is exactly why both are shown.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import settings

PERCENTILES: tuple[int, ...] = (50, 70, 90, 95, 99, 100)


@dataclass(slots=True)
class TurnRecord:
    turn_id: str
    pipeline_ms: float
    stt_ms: float = 0.0
    e2e_ms: float = 0.0
    status: str = "answered"
    path: str = "reflex"
    lang: str = "en"
    shape: str = "unknown"
    trust: float = 0.0
    slo_met: bool = True
    degraded: bool = False
    stages: dict[str, float] = field(default_factory=dict)


def percentiles(values: Sequence[float] | np.ndarray, ps: Iterable[int] = PERCENTILES) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": 0.0 for p in ps}
    out: dict[str, float] = {}
    for p in ps:
        # `lower` interpolation keeps P100 an observed value, never a synthesised one.
        v = float(arr.max()) if p == 100 else float(np.percentile(arr, p, method="linear"))
        out[f"p{p}"] = round(v, 3)
    return out


def histogram(values: Sequence[float], *, edges: Sequence[float] | None = None) -> list[dict[str, Any]]:
    edges = list(edges or [0, 10, 20, 30, 40, 50, 65, 80, 100, 125, 150, 175, 200, 250, 400])
    arr = np.asarray(list(values), dtype=np.float64)
    out: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        count = int(((arr >= lo) & (arr < hi)).sum()) if arr.size else 0
        out.append({"lo": lo, "hi": hi, "count": count})
    overflow = int((arr >= edges[-1]).sum()) if arr.size else 0
    out.append({"lo": edges[-1], "hi": None, "count": overflow})
    return out


class MetricsCollector:
    """Bounded, thread-safe, lock-per-write ring buffer."""

    def __init__(self, window: int | None = None) -> None:
        self.window = window or settings.metrics_window
        self._records: deque[TurnRecord] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._total = 0

    # ---------------------------------------------------------------- recording
    def record(self, rec: TurnRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._total += 1

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._total = 0

    def snapshot(self) -> list[TurnRecord]:
        with self._lock:
            return list(self._records)

    # ------------------------------------------------------------------ reports
    def report(self, *, include_histogram: bool = True) -> dict[str, Any]:
        recs = self.snapshot()
        if not recs:
            return {
                "samples": 0,
                "total_turns": self._total,
                "slo_ms": settings.budget.total_ms,
                "pipeline_ms": percentiles([]),
                "note": "no turns recorded yet - ask something or run scripts/bench.py",
            }

        # `pipeline_ms` reports the SLO-scoped distribution (reflex path). Deliberate turns
        # are a hosted-LLM round trip an order of magnitude slower, and blending them in
        # would make the headline percentiles describe two different systems at once.
        pipeline = [r.pipeline_ms for r in recs if r.path != "deliberate"] or [
            r.pipeline_ms for r in recs
        ]
        stt = [r.stt_ms for r in recs if r.stt_ms > 0]
        e2e = [r.e2e_ms for r in recs if r.e2e_ms > 0]
        answered = [r for r in recs if r.status == "answered"]

        stage_names: set[str] = set()
        for r in recs:
            stage_names |= set(r.stages)
        stage_table = {}
        for name in sorted(stage_names):
            vals = [r.stages[name] for r in recs if name in r.stages]
            stage_table[name] = {
                "samples": len(vals),
                "mean": round(float(np.mean(vals)), 3),
                **percentiles(vals, (50, 95, 100)),
            }

        # SLO compliance is computed over the turns the SLO actually claims to cover: the
        # local reflex path. A deliberate turn is a network round trip to a hosted LLM and
        # is documented as outside the budget (docs/LATENCY.md §5), so counting it here
        # would make the headline number contradict the claim it is meant to support.
        # Deliberate turns are still measured - reported separately, just below.
        in_scope = [r for r in recs if r.path != "deliberate"] or recs
        deliberate = [r for r in recs if r.path == "deliberate"]
        slo_met = sum(1 for r in in_scope if r.slo_met)

        report: dict[str, Any] = {
            "samples": len(recs),
            "total_turns": self._total,
            "window": self.window,
            "slo_ms": settings.budget.total_ms,
            "slo_scope": "reflex path only; the deliberate path is excluded by design",
            "slo_samples": len(in_scope),
            "slo_compliance": round(slo_met / len(in_scope), 4),
            "slo_breaches": len(in_scope) - slo_met,
            "deliberate_ms": (
                {**percentiles([r.pipeline_ms for r in deliberate]), "samples": len(deliberate)}
                if deliberate
                else None
            ),
            "degraded_turns": sum(1 for r in recs if r.degraded),
            "pipeline_ms": {
                **percentiles(pipeline),
                "mean": round(float(np.mean(pipeline)), 3),
                "min": round(float(np.min(pipeline)), 3),
                "stdev": round(float(np.std(pipeline)), 3),
            },
            "stt_ms": {**percentiles(stt), "mean": round(float(np.mean(stt)), 3)} if stt else None,
            "e2e_ms": {**percentiles(e2e), "mean": round(float(np.mean(e2e)), 3)} if e2e else None,
            "stages": stage_table,
            "status_counts": dict(Counter(r.status for r in recs)),
            "path_counts": dict(Counter(r.path for r in recs)),
            "shape_counts": dict(Counter(r.shape for r in recs)),
            "lang_counts": dict(Counter(r.lang for r in recs)),
            "answer_rate": round(len(answered) / len(recs), 4),
            "mean_trust": round(float(np.mean([r.trust for r in answered])), 2) if answered else 0.0,
        }
        if include_histogram:
            report["pipeline_histogram"] = histogram(pipeline)
        return report

    def headline(self) -> dict[str, float]:
        """Compact numbers for the UI header badge."""
        recs = [r for r in self.snapshot() if r.path != "deliberate"]
        if not recs:
            return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "samples": 0}
        p = percentiles([r.pipeline_ms for r in recs], (50, 70, 100))
        return {**p, "samples": len(recs)}


metrics = MetricsCollector()

__all__ = ["PERCENTILES", "MetricsCollector", "TurnRecord", "histogram", "metrics", "percentiles"]
