"""Guardrail primitives.

Every gate returns a :class:`GuardVerdict` - a decision plus the score and the
threshold that produced it. Nothing returns a bare boolean, because "blocked" with
no number attached is not something you can tune, test or show a user. The whole
verdict list is returned to the client and rendered in the UI's trust panel.

Three stages, chosen so that each one can fail fast:

``input``      Runs before retrieval. Rejects things that should never reach the
               index at all: malformed input, prompt injection, unsafe requests.
``retrieval``  Runs after retrieval, before answering. Decides whether the corpus
               actually contains an answer.
``output``     Runs after answering. Verifies the answer against its own evidence
               and redacts anything that should not leave the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..harness.contracts import (
    AnswerDraft,
    ChunkHit,
    Decision,
    GuardVerdict,
    RetrievalResult,
    RoutePlan,
)
from ..harness.tracing import now_ms


@dataclass
class GateContext:
    """Everything any gate might need, assembled once per turn."""

    query: str
    lang: str = "en"
    qvec: np.ndarray | None = None
    plan: RoutePlan | None = None
    retrieval: RetrievalResult | None = None
    answer: AnswerDraft | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def hits(self) -> list[ChunkHit]:
        return list(self.retrieval.hits) if self.retrieval else []


class Gate(Protocol):
    name: str
    stage: str

    def check(self, ctx: GateContext) -> GuardVerdict: ...


class BaseGate:
    name = "gate"
    stage = "input"

    def check(self, ctx: GateContext) -> GuardVerdict:  # pragma: no cover - interface
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def _verdict(
        self,
        decision: Decision,
        score: float,
        *,
        reason: str,
        threshold: float | None = None,
        started: float | None = None,
        **detail: Any,
    ) -> GuardVerdict:
        return GuardVerdict(
            gate=self.name,
            stage=self.stage,  # type: ignore[arg-type]
            decision=decision,
            score=round(float(score), 4),
            threshold=threshold,
            reason=reason,
            latency_ms=round(now_ms() - started, 3) if started is not None else 0.0,
            detail=detail,
        )

    def allow(self, score: float = 1.0, *, reason: str = "clear", started: float | None = None, **d: Any):
        return self._verdict(Decision.ALLOW, score, reason=reason, started=started, **d)

    def flag(self, score: float, *, reason: str, threshold: float | None = None,
             started: float | None = None, **d: Any):
        return self._verdict(Decision.FLAG, score, reason=reason, threshold=threshold, started=started, **d)

    def block(self, score: float, *, reason: str, threshold: float | None = None,
              started: float | None = None, **d: Any):
        return self._verdict(Decision.BLOCK, score, reason=reason, threshold=threshold, started=started, **d)


__all__ = ["BaseGate", "Gate", "GateContext"]
