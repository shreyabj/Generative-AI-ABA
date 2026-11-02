"""Rank fusion.

A hybrid retriever has to combine a cosine similarity (bounded, roughly 0-1, and
fairly well calibrated) with a BM25 score (unbounded, corpus-dependent, and
absolutely not comparable across queries). Adding them directly is a common bug:
whichever scale happens to be larger wins.

SETU fuses on **rank** first and uses scores only as a tie-breaker:

* **Reciprocal Rank Fusion** contributes ``w / (rrf_k + rank)`` per retriever.
  Rank-based, so scale-free and robust to a retriever having a bad day.
* **Normalised score blending** contributes a small amount of min-max normalised
  score signal so that a *decisively* top-1 hit outranks a merely first-place one.
* **Layer priors** apply the router's per-layer weights, so a query routed to the
  atomic layer prefers atoms without being unable to surface a document node.
* **Cohesion prior** gives a small bonus to chunks that measured as internally
  coherent at ingest - a semantically clean block is a better citation than one
  that happens to straddle a topic change.

The final ``score`` on each hit is the fused value; the per-retriever contributions
are preserved on the hit so the UI can show *why* something ranked where it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import settings


def minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


@dataclass
class Candidate:
    idx: int
    rrf: float = 0.0
    blend: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    rank_dense: int | None = None
    rank_sparse: int | None = None
    layer_weight: float = 1.0
    cohesion: float = 0.0
    score: float = 0.0
    sources: set[str] = field(default_factory=set)


class RankFuser:
    def __init__(
        self,
        rrf_k: float | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        blend_weight: float = 0.28,
        cohesion_weight: float = 0.05,
    ) -> None:
        self.rrf_k = settings.rrf_k if rrf_k is None else rrf_k
        self.dense_weight = settings.dense_weight if dense_weight is None else dense_weight
        self.sparse_weight = settings.sparse_weight if sparse_weight is None else sparse_weight
        self.blend_weight = blend_weight
        self.cohesion_weight = cohesion_weight

    def fuse(
        self,
        dense: list[tuple[np.ndarray, np.ndarray, float]],
        sparse: tuple[np.ndarray, np.ndarray] | None,
        *,
        cohesion: np.ndarray | None = None,
        top_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[Candidate]:
        """Fuse per-layer dense result sets with one sparse result set.

        ``dense`` is a list of ``(indices, scores, layer_weight)`` - one entry per
        lattice layer the router selected.
        """
        dw = self.dense_weight if dense_weight is None else dense_weight
        sw = self.sparse_weight if sparse_weight is None else sparse_weight
        pool: dict[int, Candidate] = {}

        for indices, scores, layer_weight in dense:
            if indices.size == 0:
                continue
            norm = minmax(scores)
            for rank, (i, raw, nrm) in enumerate(zip(indices, scores, norm, strict=True)):
                c = pool.setdefault(int(i), Candidate(idx=int(i)))
                c.rrf += dw * layer_weight / (self.rrf_k + rank + 1)
                c.blend += dw * layer_weight * float(nrm)
                if c.dense_score is None or float(raw) > c.dense_score:
                    c.dense_score = float(raw)
                    c.rank_dense = rank + 1
                c.layer_weight = max(c.layer_weight, layer_weight)
                c.sources.add("dense")

        if sparse is not None and sparse[0].size:
            s_idx, s_scores = sparse
            norm = minmax(s_scores)
            for rank, (i, raw, nrm) in enumerate(zip(s_idx, s_scores, norm, strict=True)):
                c = pool.setdefault(int(i), Candidate(idx=int(i)))
                c.rrf += sw / (self.rrf_k + rank + 1)
                c.blend += sw * float(nrm)
                c.sparse_score = float(raw)
                c.rank_sparse = rank + 1
                c.sources.add("sparse")

        if not pool:
            return []

        cands = list(pool.values())
        rrf_vals = minmax(np.asarray([c.rrf for c in cands], dtype=np.float32))
        blend_vals = np.asarray([c.blend for c in cands], dtype=np.float32)
        blend_vals = minmax(blend_vals)

        for c, r, b in zip(cands, rrf_vals, blend_vals, strict=True):
            coh = 0.0
            if cohesion is not None and 0 <= c.idx < cohesion.shape[0]:
                c.cohesion = float(cohesion[c.idx])
                coh = max(0.0, c.cohesion)
            c.score = float(
                (1.0 - self.blend_weight) * r
                + self.blend_weight * b
                + self.cohesion_weight * coh
            )

        cands.sort(key=lambda c: -c.score)
        return cands[: top_k or settings.fuse_k]


__all__ = ["Candidate", "RankFuser", "minmax"]
