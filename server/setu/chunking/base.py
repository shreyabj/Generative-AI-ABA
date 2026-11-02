"""Core types for the chunk lattice."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..harness.contracts import Layer
from ..text import Sentence


def stable_id(*parts: Any) -> str:
    raw = "\u241f".join(str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(slots=True)
class Document:
    """One retrievable passage plus everything MS MARCO-XI tells us about it."""

    doc_id: str
    text: str
    lang: str
    meta: dict[str, Any] = field(default_factory=dict)
    sentences: list[Sentence] | None = None
    sent_vecs: np.ndarray | None = None  # (n_sentences, dim), L2-normalised

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.meta.get("query") or "")

    @property
    def query_type(self) -> str:
        return str(self.meta.get("query_type") or "UNKNOWN")


@dataclass(slots=True)
class Chunk:
    """A node in the lattice.

    ``text`` is what a human reads and what gets cited.
    ``embed_text`` is what the encoder sees - it may carry a compact metadata
    header (see :mod:`setu.chunking.metadata`), which measurably helps
    disambiguate near-duplicate passages without polluting the displayed answer.
    """

    chunk_id: str
    doc_id: str
    layer: Layer
    text: str
    embed_text: str
    lang: str
    char_start: int
    char_end: int
    parent_id: str | None = None
    sent_span: tuple[int, int] = (0, 0)  # [first, last) sentence index in the doc
    overlap_ratio: float = 0.0
    cohesion: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "layer": self.layer.value,
            "text": self.text,
            "lang": self.lang,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "parent_id": self.parent_id,
            "sent_span": list(self.sent_span),
            "overlap_ratio": round(self.overlap_ratio, 4),
            "cohesion": round(self.cohesion, 4),
            **self.meta,
        }


class Chunker(Protocol):
    """A single view over a document."""

    layer: Layer
    name: str

    def split(self, doc: Document) -> list[Chunk]: ...


@dataclass(slots=True)
class LatticeStats:
    documents: int = 0
    chunks: int = 0
    per_layer: dict[str, int] = field(default_factory=dict)
    per_lang: dict[str, int] = field(default_factory=dict)
    avg_chars: dict[str, float] = field(default_factory=dict)
    avg_overlap: dict[str, float] = field(default_factory=dict)
    parent_links: int = 0
    # Nodes dropped because a coarser layer already carried identical text. Reported
    # rather than hidden: on a very short passage the four views legitimately converge,
    # and the collapse rate tells you how much of the corpus is too short to stratify.
    deduped: dict[str, int] = field(default_factory=dict)
    layers_per_doc: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total_emitted = self.chunks + sum(self.deduped.values())
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "per_layer": self.per_layer,
            "per_lang": self.per_lang,
            "avg_chars": {k: round(v, 1) for k, v in self.avg_chars.items()},
            "avg_overlap": {k: round(v, 3) for k, v in self.avg_overlap.items()},
            "parent_links": self.parent_links,
            "deduped": self.deduped,
            "dedup_rate": round(sum(self.deduped.values()) / total_emitted, 4) if total_emitted else 0.0,
            "docs_by_layer_count": {str(k): v for k, v in sorted(self.layers_per_doc.items())},
        }


def adjacent_cohesion(vecs: np.ndarray | None) -> float:
    """Mean cosine between neighbouring sentences - a cheap coherence proxy."""
    if vecs is None or len(vecs) < 2:
        return 1.0
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    return float(np.clip(sims.mean(), -1.0, 1.0))


def span_of(sentences: Sequence[Sentence], lo: int, hi: int) -> tuple[int, int]:
    if not sentences or lo >= hi:
        return (0, 0)
    return (sentences[lo].start, sentences[min(hi, len(sentences)) - 1].end)


__all__ = [
    "Chunk",
    "Chunker",
    "Document",
    "LatticeStats",
    "adjacent_cohesion",
    "span_of",
    "stable_id",
]
