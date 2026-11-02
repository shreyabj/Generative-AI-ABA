"""Contracts for the SETU 2.0 research pipeline (``POST /api/v1/research/ask``).

Kept separate from ``harness/contracts.py`` so the v1 ``/ask`` contract - and the
200 ms Quick path that the latency benchmark measures - is untouched.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..harness.contracts import GuardVerdict


class ResearchStyle(str, Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    COMPARATIVE = "comparative"
    CRITICAL = "critical"
    EXECUTIVE = "executive"


class Framework(str, Enum):
    SWOT = "swot"
    PESTLE = "pestle"
    PORTER = "porter"
    FINANCIAL = "financial"
    RISK = "risk"
    COMPETITOR = "competitor"
    RECOMMENDATIONS = "recommendations"
    SCENARIO = "scenario"


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=8, ge=1, le=30)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    rerank: bool | None = None
    expand: bool | None = None
    query_expansion: bool = False
    document_ids: list[str] | None = None
    langs: list[str] | None = None


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    collection_ids: list[str] = Field(default_factory=list, max_length=20)
    style: ResearchStyle = ResearchStyle.QUICK
    framework: Framework | None = None
    lang: str | None = None
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    assumptions: str | None = Field(default=None, max_length=1000)
    persist: bool = True

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v


class StageRecord(BaseModel):
    key: str
    name: str
    status: Literal["ok", "skipped", "failed", "blocked", "flagged"] = "ok"
    started_ms: float = 0.0
    duration_ms: float = 0.0
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvidenceItem(BaseModel):
    marker: int
    chunk_id: str
    collection_id: str
    collection_name: str = ""
    document_id: str | None = None
    doc_id: str
    title: str | None = None
    source_url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    published_at: str | None = None
    layer: str
    lang: str
    text: str
    context_text: str | None = None
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rerank_signals: dict[str, float] | None = None
    rank_dense: int | None = None
    rank_sparse: int | None = None
    rank_fused: int | None = None
    sub_query: int = 0
    injection_flags: list[str] = Field(default_factory=list)
    synthetic: bool = False
    cited: bool = False


class ClaimOut(BaseModel):
    text: str
    kind: Literal["fact", "calculation", "interpretation", "assumption"] = "fact"
    citations: list[int] = Field(default_factory=list)
    dropped_citations: list[int] = Field(default_factory=list)
    support: float = 0.0
    supported: bool = True
    numeric_ok: bool = True
    evidence_sentence: str | None = None
    removed_reason: str | None = None


class AnswerSection(BaseModel):
    heading: str = ""
    claims: list[ClaimOut] = Field(default_factory=list)


class AnswerTable(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class Statement(BaseModel):
    marker: int
    quote: str
    title: str | None = None
    published_at: str | None = None
    value: str | None = None


class Contradiction(BaseModel):
    topic: str
    statements: list[Statement]
    explanation: str = ""
    detector: Literal["deterministic", "model"] = "deterministic"


class CalcInput(BaseModel):
    label: str
    value: float
    unit: str = ""
    marker: int | None = None
    source: Literal["evidence", "user", "assumption"] = "evidence"


class Calculation(BaseModel):
    label: str
    operation: str
    expression: str
    inputs: list[CalcInput]
    result: float
    unit: str = ""
    verified_inputs: bool = True
    note: str = ""


class ResearchResult(BaseModel):
    turn_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    status: Literal["answered", "partial", "abstained", "refused", "error"]
    style: ResearchStyle
    framework: Framework | None = None
    question: str
    resolved_query: str
    rewrite_method: Literal["none", "rules", "llm"] = "none"
    sub_queries: list[str] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    lang: str = "en"
    summary: str = ""
    sections: list[AnswerSection] = Field(default_factory=list)
    table: AnswerTable | None = None
    removed_claims: list[ClaimOut] = Field(default_factory=list)
    answer_markdown: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    missing_information: str = ""
    guards: list[GuardVerdict] = Field(default_factory=list)
    trust_score: float = 0.0
    groundedness: float = 0.0
    stages: list[StageRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    path: Literal["extractive", "llm", "extractive-fallback", "none"] = "none"
    model: str = ""
    collections: list[dict[str, Any]] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    created_at: str = ""


__all__ = [
    "AnswerSection", "AnswerTable", "CalcInput", "Calculation", "ClaimOut", "Contradiction",
    "EvidenceItem", "Framework", "ResearchRequest", "ResearchResult", "ResearchStyle",
    "RetrievalConfig", "StageRecord", "Statement",
]
