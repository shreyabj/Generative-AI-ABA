"""Anytime hybrid retrieval engine.

"Anytime" is the load-bearing word. Every optional stage is preceded by a question
to the latency budget: *can I still afford this?* If the answer is no, the stage is
skipped, recorded in ``budget.skipped``, and the pipeline continues with what it
already has. The engine therefore has no failure mode where it blows the 200 ms SLO
to finish a nice-to-have refinement - it degrades quality instead of degrading
latency, and it says so in the response.

Flow
----
1. **Embed** the query once (LRU-cached; a repeated or speculatively prefetched
   query costs ~0 ms).
2. **Dense**, once per layer the router selected, using precomputed boolean layer
   masks so a layer restriction is a matrix view rather than a post-filter.
3. **Sparse** BM25 across the whole lattice - posting-list cost only.
4. **Fuse** with RRF + normalised blending + layer and cohesion priors.
5. **Expand parents** (optional) so a winning atom is read in the context of its
   coherence block.
6. **Diversify** - collapse overlapping spans from the same document so five
   citations are five *different* pieces of evidence, not one passage sliced four ways.
7. **Rerank** (optional) by late interaction over cached sentence vectors.

The engine also computes ``top_score`` and ``margin``, which the topicality guardrail
consumes: a flat score distribution means the corpus does not actually contain the answer,
and that is a signal to abstain rather than to answer confidently from the least-bad match.
``margin`` is the gap to the best hit from a *different source passage* rather than to hit
2, because in a parallel multilingual corpus hit 2 is usually the same passage in another
language - see the note at the emit step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import settings
from ..harness.contracts import ChunkHit, Layer, RetrievalResult, RoutePlan
from ..harness.tracing import LatencyBudget, Trace
from ..index.corpus import ChunkRecord, CorpusIndex
from ..index.fusion import Candidate, RankFuser
from ..text import truncate
from .rerank import LateInteractionReranker


def _passage_group(doc_id: str) -> str:
    """The source question a passage belongs to: "1102432:7:kn" -> "1102432".

    Every passage sharing this prefix was collected for the same original query, in every
    indexed language, so they are one body of evidence rather than competing candidates.
    """
    return doc_id.rsplit(":", 2)[0] if doc_id.count(":") >= 2 else doc_id


@dataclass
class RetrievalOptions:
    """Per-request retrieval knobs (SETU 2.0). Defaults reproduce v1 behaviour exactly.

    mode       hybrid | dense | sparse - used by the evaluation lab's ablations
    rerank     None follows the router's plan; True/False forces it
    expand     None follows the plan; True/False forces parent expansion
    row_mask   boolean mask over index rows (document / language filters). It is ANDed
               with the layer mask for dense search and passed to BM25 as its
               ``candidate_mask``, so a filter is a view, never a post-filter loop.
    sparse_query  optional expanded query text for BM25 only (pseudo-relevance feedback)
    """

    mode: str = "hybrid"
    rerank: bool | None = None
    expand: bool | None = None
    row_mask: np.ndarray | None = None
    dense_k: int | None = None
    sparse_k: int | None = None
    sparse_query: str | None = None


def _rows(idx: np.ndarray, scores: np.ndarray, index: CorpusIndex, limit: int) -> list[dict[str, Any]]:
    out = []
    for rank, (i, s) in enumerate(zip(idx[:limit], scores[:limit], strict=False), start=1):
        rec = index.record(int(i))
        out.append({
            "rank": rank, "row": int(i), "chunk_id": rec.chunk_id, "doc_id": rec.doc_id,
            "layer": rec.layer.value, "score": round(float(s), 6),
            "preview": truncate(rec.text, 140),
        })
    return out


class RetrievalEngine:
    def __init__(
        self,
        index: CorpusIndex,
        fuser: RankFuser | None = None,
        reranker: LateInteractionReranker | None = None,
    ) -> None:
        self.index = index
        self.fuser = fuser or RankFuser()
        self.reranker = reranker or LateInteractionReranker()

    # ==================================================================== search
    def search(
        self,
        query: str,
        qvec: np.ndarray,
        plan: RoutePlan,
        *,
        budget: LatencyBudget,
        trace: Trace,
        top_k: int | None = None,
        options: RetrievalOptions | None = None,
        capture: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Run hybrid retrieval.

        ``capture``, when given, is filled with the candidate lists of every stage (dense
        per layer, BM25, fused, parent expansions, rerank signals). It is what the RAG
        explorer renders; leaving it ``None`` keeps the hot path allocation-free.
        """
        top_k = top_k or settings.final_k
        b = settings.budget
        opt = options or RetrievalOptions()
        use_dense = opt.mode in ("hybrid", "dense")
        use_sparse = opt.mode in ("hybrid", "sparse")
        dense_w = plan.dense_weight if opt.mode == "hybrid" else (1.0 if use_dense else 0.0)
        sparse_w = plan.sparse_weight if opt.mode == "hybrid" else (1.0 if use_sparse else 0.0)
        row_mask = opt.row_mask
        if capture is not None:
            capture.update({"mode": opt.mode, "dense": [], "sparse": [], "fused": [],
                            "expanded": [], "reranked": [], "filtered_rows":
                            int(row_mask.sum()) if row_mask is not None else None})

        # ---------------------------------------------------------- dense, per layer
        dense_sets: list[tuple[np.ndarray, np.ndarray, float]] = []
        dense_total = 0
        if use_dense:
            with trace.span("retrieve.dense", layers=[lp.layer.value for lp in plan.layers]) as sp:
                for lp in plan.layers:
                    mask = self.index.mask_for(lp.layer)
                    if mask is None:
                        continue
                    if row_mask is not None:
                        mask = mask & row_mask
                    if not mask.any():
                        continue
                    k = lp.k if opt.dense_k is None else max(4, round(opt.dense_k * lp.weight))
                    idx, scores = self.index.vector_store.search(qvec, k, mask=mask)
                    if idx.size:
                        dense_sets.append((idx, scores, lp.weight))
                        dense_total += int(idx.size)
                        if capture is not None:
                            capture["dense"].append({
                                "layer": lp.layer.value, "weight": round(lp.weight, 4), "k": k,
                                "results": _rows(idx, scores, self.index, 10),
                            })
                    if budget.expired():
                        sp.attrs["truncated"] = True
                        break
                sp.attrs["candidates"] = dense_total
        else:
            trace.mark("retrieve.dense", status="skipped", reason=f"mode={opt.mode}")

        # -------------------------------------------------------------------- sparse
        sparse: tuple[np.ndarray, np.ndarray] | None = None
        if not use_sparse:
            trace.mark("retrieve.sparse", status="skipped", reason=f"mode={opt.mode}")
        elif opt.mode == "sparse" or budget.can_afford(
            b.retrieve_sparse_ms, reserve_ms=b.compose_ms + b.guard_output_ms
        ):
            with trace.span("retrieve.sparse") as sp:
                sparse = self.index.bm25.search(
                    opt.sparse_query or query, opt.sparse_k or settings.sparse_k,
                    candidate_mask=row_mask,
                )
                sp.attrs["candidates"] = int(sparse[0].size)
                if capture is not None:
                    capture["sparse"] = _rows(sparse[0], sparse[1], self.index, 20)
                    capture["sparse_query"] = opt.sparse_query or query
        else:
            budget.skip("retrieve.sparse", b.retrieve_sparse_ms)
            trace.mark("retrieve.sparse", status="skipped", reason="budget")

        # ---------------------------------------------------------------------- fuse
        with trace.span("fuse") as sp:
            cands = self.fuser.fuse(
                dense_sets,
                sparse,
                cohesion=self.index.cohesion,
                top_k=settings.fuse_k,
                dense_weight=dense_w,
                sparse_weight=sparse_w,
            )
            sp.attrs["fused"] = len(cands)
        fused_rank = {c.idx: i + 1 for i, c in enumerate(cands)}
        fused_score = {c.idx: c.score for c in cands}
        if capture is not None:
            capture["fusion"] = {"rrf_k": self.fuser.rrf_k, "dense_weight": round(dense_w, 4),
                                 "sparse_weight": round(sparse_w, 4),
                                 "formula": "0.72*norm(rrf) + 0.28*norm(blend) + 0.05*cohesion"}
            capture["fused"] = [
                {"rank": i + 1, "row": c.idx, "chunk_id": self.index.record(c.idx).chunk_id,
                 "doc_id": self.index.record(c.idx).doc_id,
                 "layer": self.index.record(c.idx).layer.value,
                 "rrf": round(c.rrf, 6), "blend": round(c.blend, 6), "score": round(c.score, 6),
                 "dense_score": None if c.dense_score is None else round(c.dense_score, 6),
                 "sparse_score": None if c.sparse_score is None else round(c.sparse_score, 6),
                 "sources": sorted(c.sources)}
                for i, c in enumerate(cands)
            ]

        if not cands:
            return RetrievalResult(
                hits=[], dense_candidates=dense_total,
                sparse_candidates=int(sparse[0].size) if sparse else 0, fused_candidates=0,
            )

        # ------------------------------------------------------------ parent expand
        want_expand = plan.expand_parents if opt.expand is None else opt.expand
        if want_expand and (opt.expand or budget.can_afford(b.expand_ms, reserve_ms=b.compose_ms)):
            with trace.span("expand.parents") as sp:
                before = {c.idx for c in cands}
                added = self._expand_parents(cands)
                sp.attrs["added"] = added
                if capture is not None:
                    capture["expanded"] = [
                        {"row": c.idx, "chunk_id": self.index.record(c.idx).chunk_id,
                         "layer": self.index.record(c.idx).layer.value,
                         "score": round(c.score, 6)}
                        for c in cands if c.idx not in before
                    ]
        elif want_expand:
            budget.skip("expand.parents", b.expand_ms)
            trace.mark("expand.parents", status="skipped", reason="budget")

        # ----------------------------------------------------------- diversify spans
        with trace.span("diversify") as sp:
            n_before = len(cands)
            cands = self._diversify(cands, limit=max(settings.rerank_k, top_k))
            sp.attrs["kept"] = len(cands)
            if capture is not None:
                capture["diversify"] = {"before": n_before, "kept": len(cands)}

        # -------------------------------------------------------------------- rerank
        rerank_map: dict[int, float] = {}
        best_sent: dict[int, int] = {}
        signals: dict[int, dict[str, float]] = {}
        want_rerank = plan.rerank if opt.rerank is None else opt.rerank
        do_rerank = want_rerank and (opt.rerank or budget.can_afford(
            b.rerank_ms, reserve_ms=b.compose_ms + b.guard_output_ms
        ))
        if do_rerank:
            with trace.span("rerank.late-interaction") as sp:
                rows = [c.idx for c in cands[: settings.rerank_k]]
                scored = self.reranker.rerank(
                    self.index, rows, query, qvec,
                    numeric_query=plan.shape.value == "numeric",
                )
                for row, sig in scored:
                    rerank_map[row] = sig.score
                    best_sent[row] = sig.best_sentence
                    signals[row] = {
                        "maxsim": round(sig.maxsim, 4), "meansim": round(sig.meansim, 4),
                        "lexical": round(sig.lexical, 4), "numeric": round(sig.numeric, 4),
                        "brevity": round(sig.brevity, 4), "score": round(sig.score, 6),
                    }
                order = {row: i for i, (row, _) in enumerate(scored)}
                # Reranked candidates move to the front; the tail keeps fusion order.
                cands.sort(key=lambda c: (order.get(c.idx, 10_000), -c.score))
                sp.attrs["reranked"] = len(scored)
                if capture is not None:
                    capture["reranked"] = [
                        {"rank": i + 1, "row": row, "chunk_id": self.index.record(row).chunk_id,
                         "fused_rank": fused_rank.get(row), **signals[row]}
                        for i, (row, _) in enumerate(scored)
                    ]
        elif want_rerank:
            budget.skip("rerank.late-interaction", b.rerank_ms)
            trace.mark("rerank.late-interaction", status="skipped", reason="budget")
        elif capture is not None:
            capture["rerank_skipped"] = "disabled"

        # ---------------------------------------------------------------------- emit
        hits = [
            self._to_hit(
                c, rerank_map.get(c.idx), best_sent.get(c.idx),
                fused=fused_score.get(c.idx, c.score), rank_fused=fused_rank.get(c.idx),
                signals=signals.get(c.idx),
            )
            for c in cands[:top_k]
        ]
        primary = [rerank_map.get(c.idx, c.score) for c in cands[:top_k]]
        top = float(primary[0]) if primary else 0.0

        # Margin is measured against the best hit from a *different source passage*, not
        # against hit 2.
        #
        # In a parallel corpus the runner-up is routinely the same passage in another
        # language, or the same passage at another lattice granularity. Neither is a
        # competing answer, so subtracting it measured the corpus's own redundancy and
        # called the result "a flat distribution". A correct Tamil retrieval abstained at
        # margin 0.012 for exactly this reason: its top hits were nine translations and
        # granularities of the one passage that answers the question.
        #
        # Out-of-corpus queries are unaffected, which is the point. Their top hits are
        # unrelated passages from different groups scoring alike, so the margin stays small
        # and the gate still blocks. Unanimous evidence from a single group now reports the
        # full top score instead - the strongest signal available, not the weakest.
        margin = top
        head = self._group(hits[0].doc_id) if hits else ""
        for h, score in zip(hits[1:], primary[1:], strict=False):
            if self._group(h.doc_id) != head:
                margin = float(top - score)
                break

        return RetrievalResult(
            hits=hits,
            dense_candidates=dense_total,
            sparse_candidates=int(sparse[0].size) if sparse else 0,
            fused_candidates=len(cands),
            layers_hit=sorted({h.layer for h in hits}, key=lambda layer: layer.value),
            top_score=round(top, 6),
            margin=round(margin, 6),
            truncated_by_budget=budget.degraded,
        )

    # ================================================================= internals
    def _group(self, doc_id: str) -> str:
        """Evidence group for the margin check.

        MS MARCO passages group by source question (``qid:rank:lang``). Sections of an
        ingested document group by their parent document, so two sections of the same
        report are one body of evidence rather than competing answers.
        """
        doc = self.index.docs.get(doc_id)
        if doc is not None and doc.meta.get("document_id"):
            return str(doc.meta["document_id"])
        return _passage_group(doc_id)

    def provenance(self, rec: ChunkRecord) -> dict[str, Any]:
        """Collection / document / page / section of a chunk, from its section's metadata."""
        doc = self.index.docs.get(rec.doc_id)
        if doc is None or not doc.meta.get("document_id"):
            return {}
        m = doc.meta
        pages = [p for p in (m.get("page_map") or [])
                 if p[2] > rec.char_start and p[1] < rec.char_end]
        return {
            "collection_id": m.get("collection_id"),
            "document_id": m.get("document_id"),
            "title": m.get("doc_title") or m.get("title"),
            "source_url": m.get("source_url"),
            "page_start": min(p[0] for p in pages) if pages else m.get("page_start"),
            "page_end": max(p[0] for p in pages) if pages else m.get("page_end"),
            "section": m.get("section"),
            "published_at": m.get("published_at"),
        }

    def _expand_parents(self, cands: list[Candidate]) -> int:
        """Attach each fine hit's parent as an additional, slightly discounted candidate."""
        existing = {c.idx for c in cands}
        added = 0
        for c in list(cands[: settings.rerank_k]):
            rec = self.index.record(c.idx)
            if rec.layer not in (Layer.ATOMIC, Layer.WINDOW):
                continue
            parent = self.index.parent_of(rec)
            if parent is None or parent.row in existing:
                continue
            clone = Candidate(
                idx=parent.row,
                score=c.score * 0.88,
                rrf=c.rrf * 0.88,
                blend=c.blend * 0.88,
                dense_score=c.dense_score,
                layer_weight=c.layer_weight,
                cohesion=parent.cohesion,
            )
            clone.sources.add("parent-expansion")
            cands.append(clone)
            existing.add(parent.row)
            added += 1
        if added:
            cands.sort(key=lambda c: -c.score)
        return added

    def _diversify(self, cands: list[Candidate], *, limit: int, max_per_doc: int = 2) -> list[Candidate]:
        """Drop candidates whose text span mostly duplicates a better-ranked one."""
        kept: list[Candidate] = []
        per_doc: dict[str, int] = {}
        spans: dict[str, list[tuple[int, int]]] = {}

        for c in cands:
            rec = self.index.record(c.idx)
            if per_doc.get(rec.doc_id, 0) >= max_per_doc:
                continue
            a0, a1 = rec.char_start, rec.char_end
            redundant = False
            for b0, b1 in spans.get(rec.doc_id, ()):
                inter = min(a1, b1) - max(a0, b0)
                if inter <= 0:
                    continue
                if inter / max(1, min(a1 - a0, b1 - b0)) >= 0.62:
                    redundant = True
                    break
            if redundant:
                continue
            kept.append(c)
            per_doc[rec.doc_id] = per_doc.get(rec.doc_id, 0) + 1
            spans.setdefault(rec.doc_id, []).append((a0, a1))
            if len(kept) >= limit:
                break
        return kept

    def _to_hit(
        self,
        c: Candidate,
        rerank_score: float | None,
        best_sent: int | None,
        *,
        fused: float | None = None,
        rank_fused: int | None = None,
        signals: dict[str, float] | None = None,
    ) -> ChunkHit:
        rec: ChunkRecord = self.index.record(c.idx)
        parent = self.index.parent_of(rec)
        meta = {
            "chunker": rec.meta.get("chunker"),
            "overlap_ratio": round(rec.overlap_ratio, 3),
            "cohesion": round(rec.cohesion, 3),
            "sources": sorted(c.sources),
            "row": c.idx,
        }
        if best_sent is not None:
            meta["best_sentence_index"] = best_sent
        if parent is not None:
            meta["parent_layer"] = parent.layer.value
            meta["parent_preview"] = truncate(parent.text, 260)
        if rec.meta.get("is_selected") is not None:
            meta["is_selected"] = rec.meta.get("is_selected")
        return ChunkHit(
            chunk_id=rec.chunk_id,
            doc_id=rec.doc_id,
            layer=rec.layer,
            text=rec.text,
            lang=rec.lang,
            score=round(rerank_score if rerank_score is not None else c.score, 6),
            dense_score=round(c.dense_score, 6) if c.dense_score is not None else None,
            sparse_score=round(c.sparse_score, 6) if c.sparse_score is not None else None,
            rerank_score=round(rerank_score, 6) if rerank_score is not None else None,
            rrf=round(c.rrf, 6),
            rank_dense=c.rank_dense,
            rank_sparse=c.rank_sparse,
            parent_id=rec.parent_id,
            char_start=rec.char_start,
            char_end=rec.char_end,
            meta=meta,
            fused_score=round(fused, 6) if fused is not None else None,
            rank_fused=rank_fused,
            rerank_signals=signals,
            **self.provenance(rec),
        )


__all__ = ["RetrievalEngine"]
