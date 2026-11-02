"""Retrieval and answer metrics.

Retrieval metrics are computed at **document level**: a ranked list of document ids
(deduplicated in rank order, because five chunks of one report are one document) is
scored against the labelled relevant set. All are standard IR definitions with binary
relevance.

Answer metrics are automated heuristics and are labelled that way everywhere they are
shown:

* ``faithfulness`` - groundedness coverage from the citation verifier (support of kept
  claims against their cited passages). A heuristic, not an entailment proof.
* ``citation_precision`` - share of cited documents that are in the labelled relevant set.
* ``context_relevance`` - share of assembled passages that come from relevant documents.
* ``answer_relevance`` - cosine between the question and the answer text in the shared
  embedding space. Cheap and crude; it rewards on-topic answers, not correct ones.
* ``reference_recall`` - share of the reference answer's content tokens present in the
  answer (the v1 ``evaluate.py`` measure).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..text import content_tokens


def dedup(ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(i for i in ids if i))


def recall_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def hit_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(ranked: Sequence[str], relevant: set[str], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    r = dedup(ranked)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at(r, relevant, k), 4)
        out[f"precision@{k}"] = round(precision_at(r, relevant, k), 4)
        out[f"hit@{k}"] = hit_at(r, relevant, k)
        out[f"ndcg@{k}"] = round(ndcg_at(r, relevant, k), 4)
    out["mrr@10"] = round(mrr_at(r, relevant, 10), 4)
    return out


def reference_recall(answer: str, reference: str) -> float:
    ref = set(content_tokens(reference))
    if not ref:
        return 0.0
    return round(len(ref & set(content_tokens(answer))) / len(ref), 4)


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 3)


__all__ = ["dedup", "hit_at", "mean", "mrr_at", "ndcg_at", "percentile", "precision_at", "recall_at",
           "reference_recall", "retrieval_metrics"]
