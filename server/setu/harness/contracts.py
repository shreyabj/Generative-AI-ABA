"""Typed contracts for every hop in the pipeline.

Nothing crosses a stage boundary as a loose dict. Each stage declares a
Pydantic input model and a Pydantic output model, and the tool registry
validates both directions. That is what turns "call a model" into a harness.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------- enums
class Layer(str, Enum):
    """Chunk lattice layers, coarse to fine."""

    ATOMIC = "atomic"  # L0 - proposition / sentence granularity
    SEMANTIC = "semantic"  # L1 - embedding-drift coherence blocks
    WINDOW = "window"  # L2 - fixed budget windows, adaptive overlap
    DOCUMENT = "document"  # L3 - whole passage + metadata header


class QueryShape(str, Enum):
    FACTOID = "factoid"
    NUMERIC = "numeric"
    DEFINITION = "definition"
    DESCRIPTION = "description"
    ENTITY = "entity"
    YESNO = "yesno"
    LIST = "list"
    UNKNOWN = "unknown"


class AnswerMode(str, Enum):
    REFLEX = "reflex"
    DELIBERATE = "deliberate"
    AUTO = "auto"


class Decision(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class TurnStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    REFUSED = "refused"
    DEGRADED = "degraded"
    ERROR = "error"


# ------------------------------------------------------------------ requests
class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str | None = Field(default=None, description="ISO-639-1 hint, e.g. hi, ta, en")
    mode: AnswerMode = AnswerMode.REFLEX
    budget_ms: float | None = Field(default=None, ge=20, le=20000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    explain: bool = True

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty query")
        return v


class TranscriptResult(BaseModel):
    text: str = ""
    lang: str | None = None
    lang_confidence: float | None = None
    provider: str = "none"
    model: str = ""
    duration_ms: float = 0.0
    words: list[dict[str, Any]] = Field(default_factory=list)


# ------------------------------------------------------------------- routing
class LayerPlan(BaseModel):
    layer: Layer
    k: int
    weight: float


class RoutePlan(BaseModel):
    shape: QueryShape
    script: str = "latin"
    lang: str = "en"
    layers: list[LayerPlan]
    dense_weight: float
    sparse_weight: float
    rerank: bool = True
    expand_parents: bool = True
    token_count: int = 0
    reason: str = ""


# ----------------------------------------------------------------- retrieval
class ChunkHit(BaseModel):
    chunk_id: str
    doc_id: str
    layer: Layer
    text: str
    lang: str
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    rrf: float | None = None
    rank_dense: int | None = None
    rank_sparse: int | None = None
    parent_id: str | None = None
    char_start: int = 0
    char_end: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)
    # SETU 2.0 - score fidelity. ``score`` keeps its v1 meaning (rerank score when the
    # chunk was reranked, else the fused score); these keep the parts that it overwrites.
    fused_score: float | None = None
    rank_fused: int | None = None
    rerank_signals: dict[str, float] | None = None
    # SETU 2.0 - provenance for chunks that came from a knowledge collection.
    collection_id: str | None = None
    document_id: str | None = None
    title: str | None = None
    source_url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    published_at: str | None = None


class RetrievalResult(BaseModel):
    hits: list[ChunkHit] = Field(default_factory=list)
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    layers_hit: list[Layer] = Field(default_factory=list)
    top_score: float = 0.0
    margin: float = 0.0
    truncated_by_budget: bool = False


# ----------------------------------------------------------------- answering
class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    layer: Layer
    snippet: str
    score: float


class Claim(BaseModel):
    text: str
    support: float
    cited: list[int] = Field(default_factory=list)
    numeric_ok: bool = True


class AnswerDraft(BaseModel):
    text: str = ""
    lang: str = "en"
    citations: list[Citation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    path: Literal["reflex", "deliberate", "none"] = "reflex"
    model: str = "extractive-span-fusion"
    confidence: float = 0.0


# ---------------------------------------------------------------- guardrails
class GuardVerdict(BaseModel):
    gate: str
    stage: Literal["input", "retrieval", "output"]
    decision: Decision
    score: float = 1.0
    threshold: float | None = None
    reason: str = ""
    latency_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class GuardReport(BaseModel):
    verdicts: list[GuardVerdict] = Field(default_factory=list)
    blocked: bool = False
    blocked_by: str | None = None
    trust_score: float = 0.0
    groundedness: float = 0.0
    safe_message: str | None = None

    def add(self, v: GuardVerdict) -> GuardVerdict:
        self.verdicts.append(v)
        if v.decision is Decision.BLOCK and not self.blocked:
            self.blocked = True
            self.blocked_by = v.gate
        return v


# --------------------------------------------------------------------- spans
class SpanModel(BaseModel):
    name: str
    depth: int
    start_ms: float
    duration_ms: float
    status: str = "ok"
    attrs: dict[str, Any] = Field(default_factory=dict)


class BudgetReport(BaseModel):
    total_ms: float
    used_ms: float
    remaining_ms: float
    slo_met: bool
    skipped: list[str] = Field(default_factory=list)
    degraded: bool = False


# --------------------------------------------------------------------- turns
class TurnResult(BaseModel):
    turn_id: str
    status: TurnStatus
    query: str
    normalized_query: str = ""
    answer: str = ""
    answer_lang: str = "en"
    path: str = "reflex"
    model: str = ""
    citations: list[Citation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[ChunkHit] = Field(default_factory=list)
    route: RoutePlan | None = None
    retrieval: RetrievalResult | None = None
    guards: GuardReport = Field(default_factory=GuardReport)
    trust_score: float = 0.0
    transcript: TranscriptResult | None = None
    spans: list[SpanModel] = Field(default_factory=list)
    budget: BudgetReport | None = None
    pipeline_ms: float = 0.0
    stt_ms: float = 0.0
    e2e_ms: float = 0.0
    slo_met: bool = False
    warnings: list[str] = Field(default_factory=list)
    speculative_hit: bool = False


__all__ = [
    "AnswerDraft",
    "AnswerMode",
    "AskRequest",
    "BudgetReport",
    "ChunkHit",
    "Citation",
    "Claim",
    "Decision",
    "GuardReport",
    "GuardVerdict",
    "Layer",
    "LayerPlan",
    "QueryShape",
    "RetrievalResult",
    "RoutePlan",
    "SpanModel",
    "TranscriptResult",
    "TurnResult",
    "TurnStatus",
]
