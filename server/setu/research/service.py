"""The SETU 2.0 research pipeline.

Stages, each recorded with its duration, inputs, outputs and status (the RAG explorer
renders exactly this list - nothing in it is reconstructed after the fact)::

    query -> normalise -> language -> admission guard -> understand (classify, resolve
    follow-up, decompose) -> embed -> safety guard -> dense -> sparse -> fusion ->
    parent expansion -> rerank -> topicality guard -> context assembly -> generation ->
    citation validation -> groundedness & PII -> contradictions -> final answer

It reuses every v1 component rather than replacing it: the router, the chunk lattice
indexes, the hybrid engine (now with filters and full score capture), the late-interaction
reranker, the extractive answerer, the seven guardrail gates and the trust composite.
New are multi-collection retrieval, conversation-aware query resolution, the five
generation styles, claim-level citation verification, contradiction detection and the
deterministic calculator.

The Quick style on a single collection is served by the extractive path, so it needs no
LLM and keeps v1's latency profile. ``POST /api/v1/ask`` (v1) is untouched and remains
the path the 200 ms benchmark measures.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..config import settings
from ..guardrails.base import GateContext
from ..guardrails.redaction import PIIGate
from ..harness.contracts import (
    AnswerDraft,
    ChunkHit,
    Decision,
    GuardReport,
    GuardVerdict,
    RetrievalResult,
    RoutePlan,
)
from ..harness.tracing import LatencyBudget, Trace, now_ms
from ..knowledge.registry import BUILTIN_ID, CollectionRegistry, LoadedCollection
from ..knowledge.service import KnowledgeError
from ..retrieval.engine import RetrievalOptions
from ..retrieval.router import QueryRouter
from ..storage.embed_cache import CachedEncoder
from ..storage.repos import ConversationRepo, TraceRepo
from ..text import LANG_NAMES, content_tokens, detect_lang, normalize, numbers_in, truncate
from . import contradictions as contra
from .generate import (
    build_calculations,
    build_prompt,
    extractive_sections,
    parse_answer,
    to_markdown,
)
from .llm import LLMClient, LLMError
from .models import (
    AnswerSection,
    AnswerTable,
    ClaimOut,
    EvidenceItem,
    Framework,
    ResearchRequest,
    ResearchResult,
    ResearchStyle,
    RetrievalConfig,
    StageRecord,
)
from .understanding import QueryUnderstanding, Understanding
from .verify import CitationVerifier

MAX_CONTEXT_PASSAGES = {ResearchStyle.QUICK: 6, ResearchStyle.DETAILED: 12, ResearchStyle.COMPARATIVE: 14,
                        ResearchStyle.CRITICAL: 12, ResearchStyle.EXECUTIVE: 12}


# ================================================================ stage recorder
class Stages:
    def __init__(self) -> None:
        self.t0 = now_ms()
        self.records: list[StageRecord] = []

    @contextmanager
    def run(self, key: str, name: str, **inputs: Any):
        rec = StageRecord(key=key, name=name, started_ms=round(now_ms() - self.t0, 3), input=inputs)
        t = now_ms()
        try:
            yield rec
        except Exception as exc:
            rec.status = "failed"
            rec.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec.duration_ms = round(now_ms() - t, 3)
            self.records.append(rec)

    def add(self, key: str, name: str, *, status: str = "ok", duration_ms: float = 0.0,
            started_ms: float | None = None, **parts: Any) -> StageRecord:
        rec = StageRecord(key=key, name=name, status=status,  # type: ignore[arg-type]
                          started_ms=round(started_ms if started_ms is not None else now_ms() - self.t0, 3),
                          duration_ms=round(duration_ms, 3), input=parts.get("input", {}),
                          output=parts.get("output", {}), detail=parts.get("detail", {}),
                          error=parts.get("error"))
        self.records.append(rec)
        return rec

    def total(self) -> float:
        return round(now_ms() - self.t0, 3)


# ================================================================== retrieval
class RetrievalRun:
    """One (sub-query, collection) retrieval with its capture and timings."""

    def __init__(self, qi: int, query: str, col: LoadedCollection, plan: RoutePlan, result: RetrievalResult,
                 capture: dict[str, Any] | None, spans: dict[str, float], ms: float, sparse_query: str | None):
        self.qi, self.query, self.col, self.plan, self.result = qi, query, col, plan, result
        self.capture, self.spans, self.ms, self.sparse_query = capture, spans, ms, sparse_query


def prf_terms(col: LoadedCollection, query: str, n_docs: int = 3, n_terms: int = 5) -> list[str]:
    """Pseudo-relevance feedback: frequent content terms from the top BM25 hits."""
    idx, _ = col.index.bm25.search(query, n_docs)
    q = set(content_tokens(query))
    counts: Counter[str] = Counter()
    for i in idx:
        counts.update(t for t in content_tokens(col.index.record(int(i)).text) if t not in q and len(t) > 3
                      and not t.isdigit())
    return [t for t, _ in counts.most_common(n_terms)]


class ResearchService:
    def __init__(self, *, pipeline: Any, registry: CollectionRegistry, llm: LLMClient | None = None) -> None:
        self.pipeline = pipeline
        self.registry = registry
        self.encoder = pipeline.encoder
        self.guards = pipeline.guards
        self.router = QueryRouter()
        self.llm = llm or LLMClient()
        self.understanding = QueryUnderstanding(self.llm)
        self.cached = CachedEncoder(self.encoder)
        self.verifier = CitationVerifier(self.cached)
        self.pii = PIIGate()
        self.conversations = ConversationRepo()
        self.traces = TraceRepo()

    # ------------------------------------------------------------ collections
    def resolve_collections(self, ids: list[str]) -> tuple[list[LoadedCollection], list[str]]:
        warnings: list[str] = []
        if not ids:
            user = [c for c in self.registry.items() if c.kind != "builtin"]
            cols = user or [c for c in self.registry.items() if c.id == BUILTIN_ID]
            return cols, warnings
        cols = []
        for cid in dict.fromkeys(ids):
            c = self.registry.get(cid)
            if c is None:
                from ..storage.repos import CollectionRepo

                row = CollectionRepo().get(cid)
                if row is None:
                    raise KnowledgeError(f"collection '{cid}' not found", 404)
                warnings.append(f"collection '{row['name']}' is not searchable yet (status: {row['status']})")
                continue
            cols.append(c)
        return cols, warnings

    @staticmethod
    def _doc_mask(col: LoadedCollection, document_ids: set[str] | None, langs: set[str] | None) -> np.ndarray | None:
        if not document_ids and not langs:
            return None
        idx = col.index
        mask = np.ones(len(idx.chunks), dtype=bool)
        if document_ids:
            docs_ok = {d for d, rec in idx.docs.items() if rec.meta.get("document_id") in document_ids or d in document_ids}
            mask &= np.fromiter((c.doc_id in docs_ok for c in idx.chunks), dtype=bool, count=len(idx.chunks))
        if langs:
            mask &= np.fromiter((c.lang in langs for c in idx.chunks), dtype=bool, count=len(idx.chunks))
        return mask

    def retrieve(self, queries: list[tuple[str, list[str] | None]], cols: list[LoadedCollection],
                 cfg: RetrievalConfig, *, lang: str | None, capture: bool, budget_ms: float) -> list[RetrievalRun]:
        runs: list[RetrievalRun] = []
        base_docs = set(cfg.document_ids or [])
        langs = set(cfg.langs or [])
        for qi, (text, doc_filter) in enumerate(queries):
            qvec = self.encoder.encode_query(text)
            for col in cols:
                plan = self.router.route(text, lang_hint=lang, available_layers=set(col.index.layer_masks))
                docs = (base_docs | set(doc_filter)) if doc_filter else (base_docs or None)
                mask = self._doc_mask(col, docs, langs or None)
                if mask is not None and not mask.any():
                    continue
                sparse_q = None
                if cfg.query_expansion:
                    extra = prf_terms(col, text)
                    sparse_q = f"{text} {' '.join(extra)}" if extra else None
                opts = RetrievalOptions(mode=cfg.mode, rerank=cfg.rerank, expand=cfg.expand, row_mask=mask,
                                        sparse_query=sparse_q)
                trace = Trace()
                cap: dict[str, Any] | None = {} if capture else None
                t = now_ms()
                res = col.engine.search(text, qvec, plan, budget=LatencyBudget(budget_ms), trace=trace,
                                        top_k=cfg.top_k, options=opts, capture=cap)
                runs.append(RetrievalRun(qi, text, col, plan, res, cap, trace.by_name(), now_ms() - t, sparse_q))
        return runs

    @staticmethod
    def merge(runs: list[RetrievalRun], top_k: int, n_queries: int) -> list[tuple[ChunkHit, RetrievalRun]]:
        best: dict[tuple[str, str], tuple[ChunkHit, RetrievalRun]] = {}
        for r in runs:
            for h in r.result.hits:
                key = (r.col.id, h.chunk_id)
                if key not in best or h.score > best[key][0].score:
                    best[key] = (h, r)
        ranked = sorted(best.values(), key=lambda t: -t[0].score)
        limit = max(top_k, min(30, n_queries * 3)) if n_queries > 1 else top_k
        if n_queries <= 1:
            return ranked[:limit]
        # Each sub-query is guaranteed a share of the evidence, so a comparison of A and B
        # cannot end up citing only A because A's passages embed slightly closer.
        quota = max(2, math.ceil(limit / n_queries))
        chosen: list[tuple[ChunkHit, RetrievalRun]] = []
        taken: set[tuple[str, str]] = set()
        for qi in range(n_queries):
            mine = [t for t in ranked if t[1].qi == qi and (t[1].col.id, t[0].chunk_id) not in taken][:quota]
            for t in mine:
                taken.add((t[1].col.id, t[0].chunk_id))
                chosen.append(t)
        for t in ranked:
            if len(chosen) >= limit:
                break
            k = (t[1].col.id, t[0].chunk_id)
            if k not in taken:
                taken.add(k)
                chosen.append(t)
        chosen.sort(key=lambda t: -t[0].score)
        return chosen[:limit]

    # ----------------------------------------------------------------- history
    def history(self, conversation_id: str | None) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        out = []
        for m in self.conversations.messages(conversation_id)[-8:]:
            if m["role"] != "assistant":
                continue
            p = m.get("payload") or {}
            out.append({"question": p.get("question") or "", "resolved": m.get("resolved_query") or p.get("resolved_query"),
                        "summary": p.get("summary") or truncate(m.get("content") or "", 300),
                        "documents": list({e.get("document_id") for e in p.get("evidence", []) if e.get("cited")})})
        return out[-3:]

    # ===================================================================== ask
    async def ask(self, req: ResearchRequest, *, session_id: str) -> ResearchResult:
        st = Stages()
        trace_id = Trace().turn_id
        warnings: list[str] = []
        report = GuardReport()
        style = req.style
        framework = req.framework

        # ---- conversation ----------------------------------------------------
        conv_id = req.conversation_id
        if conv_id:
            conv = self.conversations.get(conv_id)
            if conv is None or conv["session_id"] != session_id:
                raise KnowledgeError("conversation not found", 404)
        history = self.history(conv_id)

        st.add("query", "User query", output={"question": req.question, "style": style.value,
                                               "framework": framework.value if framework else None,
                                               "conversation_turns": len(history)})

        # ---- normalise + language ---------------------------------------------
        with st.run("normalize", "Query normalisation", text=req.question) as s:
            norm = normalize(req.question, casefold=False)
            s.output = {"text": norm, "changed": norm != req.question}
        with st.run("language", "Language detection", hint=req.lang) as s:
            lang = detect_lang(norm, req.lang)
            s.output = {"lang": lang, "name": LANG_NAMES.get(lang, lang),
                        "method": "explicit hint" if req.lang else "script detection"}

        resolution: dict[str, Any] = {"method": "none"}

        def result(status: str, **kw: Any) -> ResearchResult:
            return ResearchResult(
                turn_id=trace_id, conversation_id=conv_id, status=status, style=style,  # type: ignore[arg-type]
                framework=framework, question=req.question, resolved_query=kw.pop("resolved", norm), lang=lang,
                rewrite_method=resolution["method"], guards=report.verdicts, stages=st.records, warnings=warnings,
                created_at=datetime.now(timezone.utc).isoformat(), **kw,
            )

        # ---- admission guards (sanity + injection) ----------------------------
        gctx = GateContext(query=norm, lang=lang)
        with st.run("guard.admission", "Admission guardrails") as s:
            blocked = self.guards.run_admission(gctx, report)
            s.output = {"verdicts": [v.model_dump(mode="json") for v in report.verdicts]}
            if blocked:
                s.status = "blocked"
        if blocked:
            res = result("refused", missing_information=report.safe_message or "",
                         summary=report.safe_message or "")
            return await self._finish(res, req, session_id, st)

        # ---- collections -------------------------------------------------------
        cols, w = self.resolve_collections(req.collection_ids)
        warnings.extend(w)
        if not cols:
            res = result("abstained", missing_information="No searchable collection was selected. "
                         "Add documents to a collection or select the built-in sample.")
            return await self._finish(res, req, session_id, st)
        doc_titles: dict[str, str] = {}
        for c in cols:
            for rec in c.index.docs.values():
                did = rec.meta.get("document_id")
                if did:
                    doc_titles[did] = " ".join(str(rec.meta.get(k) or "") for k in ("doc_title", "filename", "domain"))

        # ---- understanding -------------------------------------------------------
        with st.run("understand", "Query classification & resolution", question=norm,
                    history=[h.get("resolved") for h in history]) as s:
            u: Understanding = await self.understanding.understand(norm, history, doc_titles=doc_titles)
            shape = self.router.route(u.resolved, lang_hint=lang)
            s.output = {**u.to_dict(), "shape": shape.shape.value, "route_reason": shape.reason}
            if u.is_followup and u.method == "rules":
                s.status = "flagged"
        resolution["method"] = u.method
        if u.intent == "comparison" and style is ResearchStyle.QUICK:
            warnings.append("comparison question - the Comparative Analysis style gives a structured comparison")
        if framework is None and u.intent == "scenario" and style is not ResearchStyle.QUICK:
            framework = Framework.SCENARIO

        queries: list[tuple[str, list[str] | None]] = [(u.resolved, None)]
        ent_docs = list(u.entity_documents.items())
        for i, sq in enumerate(u.sub_queries):
            filt = ent_docs[i][1] if u.intent in ("comparison", "contradiction") and i < len(ent_docs) \
                and not u.multi_part else None
            queries.append((sq, filt))

        # ---- embed + safety ------------------------------------------------------
        with st.run("embed", "Query embedding", queries=[q for q, _ in queries]) as s:
            before = self.encoder.stats()["cache"]["hits"]  # type: ignore[index]
            qvec = self.encoder.encode_query(u.resolved)
            for q, _ in queries[1:]:
                self.encoder.encode_query(q)
            after = self.encoder.stats()["cache"]["hits"]  # type: ignore[index]
            s.output = {"model": self.encoder.model_name, "dim": int(qvec.shape[0]), "vectors": len(queries),
                        "cache_hits": after - before, "norm": round(float(np.linalg.norm(qvec)), 4),
                        "preview": [round(float(x), 4) for x in qvec[:8]]}
        gctx = GateContext(query=u.resolved, lang=lang, qvec=qvec)
        with st.run("guard.safety", "Semantic safety guardrail") as s:
            blocked = self.guards.run_safety(gctx, report)
            s.output = {"verdict": report.verdicts[-1].model_dump(mode="json") if report.verdicts else None}
            if blocked:
                s.status = "blocked"
        if blocked:
            res = result("refused", resolved=u.resolved, missing_information=report.safe_message or "",
                         summary=report.safe_message or "", analysis=u.to_dict())
            return await self._finish(res, req, session_id, st)

        # ---- retrieval (CPU-bound: off the event loop) ---------------------------
        cfg = req.retrieval
        if style is not ResearchStyle.QUICK and cfg.top_k < 10:
            cfg = cfg.model_copy(update={"top_k": max(cfg.top_k, MAX_CONTEXT_PASSAGES[style] - 2)})
        budget_ms = settings.budget.total_ms if style is ResearchStyle.QUICK else 10_000.0
        t_ret = now_ms()
        runs = await asyncio.to_thread(self.retrieve, queries, cols, cfg, lang=lang, capture=True,
                                       budget_ms=budget_ms)
        ret_ms = now_ms() - t_ret
        self._retrieval_stages(st, runs, t_ret - st.t0, ret_ms)
        merged = self.merge(runs, cfg.top_k, len(queries))

        # ---- topicality -----------------------------------------------------------
        primary = [r for r in runs if r.qi == 0]
        primary_best = max(primary, key=lambda r: r.result.top_score, default=None)
        with st.run("guard.topicality", "Retrieval topicality guardrail") as s:
            verdicts = []
            passing = False
            for r in sorted(runs, key=lambda r: (r.qi, -r.result.top_score)):
                if not r.result.hits:
                    continue
                g = GateContext(query=r.query, lang=lang, retrieval=r.result,
                                extras={"dated_corpus": r.col.dated})
                v = self.guards.topicality.check(g)
                verdicts.append({"query": r.query, "collection": r.col.name, **v.model_dump(mode="json")})
                if v.decision is not Decision.BLOCK:
                    passing = True
                    if r is primary_best or r.qi == 0:
                        report.add(v)
                        break
            if not passing and verdicts:
                report.add(GuardVerdict(**{k: verdicts[0][k] for k in GuardVerdict.model_fields if k in verdicts[0]}))
            s.output = {"checked": verdicts[:6], "passed": passing}
            if not passing:
                s.status = "blocked"
        if not merged or not passing:
            closest = self._evidence(merged[:4], style)
            reason = verdicts[0]["reason"] if verdicts else "no passages were retrieved"
            missing = (
                f"The selected collection(s) ({', '.join(c.name for c in cols)}) do not contain passages that "
                f"distinctly answer this question ({reason}). The closest passages are listed as sources. "
                "Try naming the company, metric and period explicitly, or add a document that covers it."
            )
            res = result("abstained", resolved=u.resolved, sub_queries=u.sub_queries, analysis=u.to_dict(),
                         missing_information=missing, evidence=closest,
                         collections=[c.summary() for c in cols])
            return await self._finish(res, req, session_id, st)

        # ---- context assembly -------------------------------------------------------
        with st.run("context", "Context assembly", candidates=len(merged)) as s:
            evidence = self._evidence(merged[:MAX_CONTEXT_PASSAGES[style]], style)
            per_doc = Counter(e.document_id or e.doc_id for e in evidence)
            s.output = {
                "passages": len(evidence), "documents": len(per_doc),
                "chars": sum(len(e.context_text or e.text) for e in evidence),
                "injection_flagged": sum(1 for e in evidence if e.injection_flags),
                "synthetic": sum(1 for e in evidence if e.synthetic),
                "order": [{"marker": e.marker, "title": e.title or e.doc_id, "collection": e.collection_name,
                           "layer": e.layer, "score": round(e.score, 4), "sub_query": e.sub_query} for e in evidence],
            }
        if any(e.synthetic for e in evidence):
            warnings.append("some evidence comes from clearly labelled SYNTHETIC demonstration documents")

        # ---- generation + verification ----------------------------------------------
        use_llm = style is not ResearchStyle.QUICK or framework is not None
        if use_llm and not self.llm.enabled:
            warnings.append("no LLM key configured - served by the extractive fallback (verbatim spans, no synthesis)")
            use_llm = False
        if use_llm:
            res = await self._generate_llm(req, st, u, lang, style, framework, evidence, primary_best, warnings, report)
            if res is None:  # LLM failed: degrade, never crash
                res = self._generate_extractive(st, u, lang, style, merged, evidence, primary_best, qvec, warnings,
                                                report, fallback=True)
        else:
            res = self._generate_extractive(st, u, lang, style, merged, evidence, primary_best, qvec, warnings,
                                            report, fallback=False)
        res.turn_id, res.conversation_id = trace_id, conv_id
        res.question, res.lang, res.style, res.framework = req.question, lang, style, framework
        res.resolved_query, res.rewrite_method = u.resolved, u.method  # type: ignore[assignment]
        res.sub_queries, res.analysis = u.sub_queries, u.to_dict()
        res.collections = [c.summary() for c in cols]
        res.guards, res.stages, res.warnings = report.verdicts, st.records, warnings
        res.created_at = datetime.now(timezone.utc).isoformat()
        res.timings = {"retrieval_ms": round(ret_ms, 2)}
        return await self._finish(res, req, session_id, st)

    # ============================================================ retrieval stages
    def _retrieval_stages(self, st: Stages, runs: list[RetrievalRun], started: float, total_ms: float) -> None:
        def agg(name: str) -> float:
            return round(sum(r.spans.get(name, 0.0) for r in runs), 3)

        def per(key: str) -> list[dict[str, Any]]:
            out = []
            for r in runs:
                cap = r.capture or {}
                out.append({"query": r.query, "sub_query": r.qi, "collection": r.col.name,
                            "collection_id": r.col.id, "result": cap.get(key)})
            return out

        st.add("retrieve.dense", "Dense retrieval (per lattice layer)", started_ms=started,
               duration_ms=agg("retrieve.dense"),
               status="skipped" if all((r.capture or {}).get("mode") == "sparse" for r in runs) else "ok",
               input={"layers": sorted({lp.layer.value for r in runs for lp in r.plan.layers}),
                      "runs": len(runs)},
               output={"candidates": sum(r.result.dense_candidates for r in runs), "per_run": per("dense")})
        st.add("retrieve.sparse", "Sparse retrieval (BM25)", started_ms=started, duration_ms=agg("retrieve.sparse"),
               status="skipped" if all((r.capture or {}).get("mode") == "dense" for r in runs) else "ok",
               input={"expanded_queries": [r.sparse_query for r in runs if r.sparse_query]},
               output={"candidates": sum(r.result.sparse_candidates for r in runs), "per_run": per("sparse")})
        st.add("fusion", "Reciprocal Rank Fusion", started_ms=started, duration_ms=agg("fuse"),
               input={"config": next(((r.capture or {}).get("fusion") for r in runs), None)},
               output={"fused": sum(r.result.fused_candidates for r in runs), "per_run": per("fused")})
        exp = agg("expand.parents")
        st.add("expand", "Parent chunk expansion", started_ms=started, duration_ms=exp,
               status="ok" if any((r.capture or {}).get("expanded") for r in runs) or exp else "skipped",
               output={"added": sum(len((r.capture or {}).get("expanded") or []) for r in runs),
                       "per_run": per("expanded")})
        rr = agg("rerank.late-interaction")
        st.add("rerank", "Late-interaction reranking", started_ms=started, duration_ms=rr,
               status="ok" if rr else "skipped",
               input={"weights": "0.46 maxsim + 0.16 meansim + 0.22 lexical + 0.06 numeric + 0.10 brevity"},
               output={"per_run": per("reranked")})
        st.add("retrieve.summary", "Hybrid retrieval (all runs)", started_ms=started, duration_ms=total_ms,
               output={"runs": [{"query": r.query, "collection": r.col.name, "ms": round(r.ms, 2),
                                 "hits": len(r.result.hits), "top_score": r.result.top_score,
                                 "margin": r.result.margin, "shape": r.plan.shape.value} for r in runs]})

    # =================================================================== evidence
    def _evidence(self, merged: list[tuple[ChunkHit, RetrievalRun]], style: ResearchStyle) -> list[EvidenceItem]:
        out = []
        for i, (h, r) in enumerate(merged, start=1):
            idx = r.col.index
            row = idx.row_of(h.chunk_id)
            rec = idx.record(row) if row is not None else None
            dmeta = idx.docs[h.doc_id].meta if h.doc_id in idx.docs else {}
            context = None
            if style is not ResearchStyle.QUICK and rec is not None and h.doc_id in idx.docs:
                # Small-to-search, big-to-read: the generator gets a ~900-char window of the
                # section around the hit, so a figure in the next sentence is not lost.
                sec = idx.docs[h.doc_id].text
                if len(sec) <= 950:
                    context = sec
                else:
                    a = max(0, rec.char_start - 250)
                    b = min(len(sec), max(rec.char_end + 150, a + 900))
                    context = ("…" if a else "") + sec[a:b] + ("…" if b < len(sec) else "")
            title = h.title or dmeta.get("title") or dmeta.get("query")
            out.append(EvidenceItem(
                marker=i, chunk_id=h.chunk_id, collection_id=r.col.id, collection_name=r.col.name,
                document_id=h.document_id, doc_id=h.doc_id, title=title, source_url=h.source_url,
                page_start=h.page_start, page_end=h.page_end, section=h.section, published_at=h.published_at,
                layer=h.layer.value, lang=h.lang, text=h.text, context_text=context, score=h.score,
                dense_score=h.dense_score, sparse_score=h.sparse_score, fused_score=h.fused_score,
                rerank_score=h.rerank_score, rerank_signals=h.rerank_signals, rank_dense=h.rank_dense,
                rank_sparse=h.rank_sparse, rank_fused=h.rank_fused, sub_query=r.qi,
                injection_flags=list((rec.meta.get("injection_flags") if rec else None) or []),
                synthetic=bool(dmeta.get("synthetic")),
            ))
        return out

    # ================================================================ extractive
    def _generate_extractive(self, st: Stages, u: Understanding, lang: str, style: ResearchStyle,
                             merged: list[tuple[ChunkHit, RetrievalRun]], evidence: list[EvidenceItem],
                             primary: RetrievalRun | None, qvec: np.ndarray, warnings: list[str],
                             report: GuardReport, *, fallback: bool) -> ResearchResult:
        marker_of = {(e.collection_id, e.chunk_id): e.marker for e in evidence}
        extra: list[EvidenceItem] = []

        def remap(draft: AnswerDraft, col: LoadedCollection) -> AnswerDraft:
            mapping = {}
            for c in draft.citations:
                key = (col.id, c.chunk_id)
                if key not in marker_of:
                    row = col.index.row_of(c.chunk_id)
                    if row is None:
                        continue
                    rec = col.index.record(row)
                    m = len(evidence) + len(extra) + 1
                    marker_of[key] = m
                    dm = col.index.docs[rec.doc_id].meta
                    extra.append(EvidenceItem(
                        marker=m, chunk_id=rec.chunk_id, collection_id=col.id, collection_name=col.name,
                        document_id=dm.get("document_id"), doc_id=rec.doc_id,
                        title=dm.get("title") or dm.get("query"), layer=rec.layer.value, lang=rec.lang,
                        text=rec.text, score=c.score, synthetic=bool(dm.get("synthetic")),
                    ))
                mapping[c.marker] = marker_of[key]
            text = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping.get(int(m.group(1)), m.group(1))}]", draft.text)
            return draft.model_copy(update={"text": text})

        with st.run("generate", "Answer generation", path="extractive-fallback" if fallback else "extractive",
                    style=style.value) as s:
            groups: list[tuple[str, list[tuple[ChunkHit, RetrievalRun]]]] = []
            if style is ResearchStyle.QUICK or len({r.qi for _, r in merged}) <= 1:
                by_col: dict[str, list[tuple[ChunkHit, RetrievalRun]]] = {}
                for t in merged:
                    by_col.setdefault(t[1].col.id, []).append(t)
                groups = [("", v) for v in by_col.values()]
            else:
                by_q: dict[int, list[tuple[ChunkHit, RetrievalRun]]] = {}
                for t in merged:
                    by_q.setdefault(t[1].qi, []).append(t)
                labels = [u.resolved, *u.sub_queries]
                groups = [(labels[qi] if qi < len(labels) else "", v) for qi, v in sorted(by_q.items())]

            drafts: list[tuple[str, AnswerDraft, LoadedCollection, list[ChunkHit]]] = []
            for heading, items in groups:
                col = items[0][1].col
                hits = [h for h, r in items if r.col.id == col.id]
                plan = items[0][1].plan
                d = col.reflex.answer(u.resolved if not heading else heading, qvec if not heading
                                      else self.encoder.encode_query(heading), hits, plan)
                if d.text.strip():
                    drafts.append((heading, d, col, hits))
            s.output = {"drafts": len(drafts), "groups": len(groups),
                        "confidences": [round(d.confidence, 3) for _, d, _, _ in drafts]}

        if not drafts:
            return ResearchResult(turn_id="", status="abstained", style=style, question=u.original,
                                  resolved_query=u.resolved, evidence=evidence, path="extractive",
                                  missing_information="The retrieved passages contain no sentence that answers "
                                                      "the question closely enough to quote.")

        # Quick: one best draft, verified by the v1 groundedness gate of its collection.
        if style is ResearchStyle.QUICK or len(drafts) == 1:
            drafts.sort(key=lambda t: -t[1].confidence)
            drafts = drafts[:1]
        verified: list[tuple[str, AnswerDraft]] = []
        coverage_vals, weakest_vals = [], []
        with st.run("verify.citations", "Citation validation") as s:
            details = []
            for heading, d, col, hits in drafts:
                g = GateContext(query=u.resolved, lang=lang, retrieval=RetrievalResult(hits=hits), answer=d)
                v = col.groundedness.check(g)
                d.claims = list(col.groundedness.claims)
                details.append({"group": heading or "answer", **v.model_dump(mode="json")})
                if v.decision is Decision.BLOCK:
                    continue
                coverage_vals.append(v.score)
                weakest_vals.append(min((c.support for c in d.claims), default=v.score))
                verified.append((heading, remap(d, col)))
            s.output = {"method": "v1 groundedness gate (lexical containment + numeric check) on verbatim spans",
                        "checks": details}
            if not verified:
                s.status = "blocked"
        if not verified:
            return ResearchResult(turn_id="", status="abstained", style=style, question=u.original,
                                  resolved_query=u.resolved, evidence=evidence + extra, path="extractive",
                                  missing_information="The quotable passages did not pass verification.")
        evidence = evidence + extra
        sections = extractive_sections(verified)
        # Carry the v1 gate's per-claim support onto the claims it verified.
        support_by_text = {re.sub(r"\s*\[\d+\]", "", c.text).strip(): c.support
                           for _, d in verified for c in d.claims}
        for sec in sections:
            for c in sec.claims:
                c.support = round(float(support_by_text.get(c.text, coverage_vals[0] if coverage_vals else 0.0)), 4)
        cited = {m for s in sections for c in s.claims for m in c.citations}
        for e in evidence:
            e.cited = e.marker in cited
        table = None
        if style is ResearchStyle.COMPARATIVE and len(sections) > 1:
            table = AnswerTable(columns=["Sub-question", "Evidence (verbatim)"],
                                rows=[[s.heading or "-", " ".join(f"{c.text} [{','.join(map(str, c.citations))}]"
                                                                for c in s.claims[:2])] for s in sections])
        coverage = float(np.mean(coverage_vals)) if coverage_vals else 0.0
        report.groundedness = coverage
        self._pii(sections, report)
        trust = self._trust(coverage, min(weakest_vals) if weakest_vals else 0.0, primary,
                            len({e.document_id or e.doc_id for e in evidence if e.cited}), report)
        status = "answered"
        with st.run("guard.output", "Groundedness, PII & trust") as s:
            s.output = {"coverage": round(coverage, 4), "trust": trust,
                        "trust_floor": settings.trust_abstain_below}
            if style is ResearchStyle.QUICK and trust < settings.trust_abstain_below:
                s.status = "blocked"
                status = "abstained"
        contradictions = contra.detect_numeric(evidence)
        st.add("contradictions", "Contradiction detection", output={"found": len(contradictions),
                                                                   "detectors": ["deterministic"]})
        summary = sections[0].claims[0].text if sections and sections[0].claims else ""
        follow = self._fallback_followups(u, evidence)
        if status == "abstained":
            return ResearchResult(turn_id="", status="abstained", style=style, question=u.original,
                                  resolved_query=u.resolved, evidence=evidence, trust_score=trust,
                                  groundedness=coverage, path="extractive",
                                  missing_information="The best quotable answer scored below the trust floor "
                                                      f"({trust:.0f} < {settings.trust_abstain_below:.0f}).")
        return ResearchResult(
            turn_id="", status=status, style=style, question=u.original, resolved_query=u.resolved,  # type: ignore[arg-type]
            summary="" if style is ResearchStyle.QUICK else summary, sections=sections, table=table,
            answer_markdown=to_markdown("", sections, table, []), evidence=evidence,
            contradictions=contradictions, follow_ups=follow, trust_score=trust, groundedness=round(coverage, 4),
            path="extractive-fallback" if fallback else "extractive", model="extractive-span-fusion",
        )

    # ======================================================================= LLM
    async def _generate_llm(self, req: ResearchRequest, st: Stages, u: Understanding, lang: str,
                            style: ResearchStyle, framework: Framework | None, evidence: list[EvidenceItem],
                            primary: RetrievalRun | None, warnings: list[str],
                            report: GuardReport) -> ResearchResult | None:
        system, user, budget = build_prompt(question=req.question, resolved=u.resolved, lang=lang, style=style,
                                            framework=framework, items=evidence, assumptions=req.assumptions,
                                            sub_queries=u.sub_queries)
        with st.run("generate", "LLM answer generation", model=settings.research_model, style=style.value,
                    framework=framework.value if framework else None, prompt_chars=len(system) + len(user)) as s:
            try:
                data, meta = await self.llm.json(system=system, user=user, max_tokens=budget + 2500)
            except LLMError as exc:
                s.status = "failed"
                s.error = str(exc)
                warnings.append(f"LLM generation failed ({exc}) - served by the extractive fallback")
                return None
            parsed = parse_answer(data, len(evidence))
            s.output = {"model": meta["model"], "ms": meta["ms"], "attempts": meta["attempts"],
                        "usage": meta.get("usage", {}), "abstain": parsed["abstain"],
                        "sections": len(parsed["sections"]),
                        "claims": sum(len(x.claims) for x in parsed["sections"]),
                        "raw_preview": truncate(str(data), 1200)}

        if parsed["abstain"] and not any(x.claims for x in parsed["sections"]):
            return ResearchResult(turn_id="", status="abstained", style=style, question=req.question,
                                  resolved_query=u.resolved, evidence=evidence[:5], path="llm",
                                  model=settings.research_model,
                                  missing_information=parsed["missing"] or "The evidence does not answer this question.",
                                  follow_ups=parsed["follow_ups"])

        ev_text = {e.marker: (e.context_text or e.text) for e in evidence}
        calcs, cw = build_calculations(parsed["calculations"], ev_text, req.question, req.assumptions)
        warnings.extend(cw)
        allowed: set[str] = set()
        for c in calcs:
            for digits in (0, 1, 2):
                allowed.add(f"{round(c.result, digits):g}")
            allowed.add(f"{abs(round(c.result, 1)):g}")
        user_nums = set(numbers_in(req.question + " " + (req.assumptions or "")))

        with st.run("verify.citations", "Citation validation (claim level)") as s:
            all_claims = [c for x in parsed["sections"] for c in x.claims]
            summary_claim = ClaimOut(text=parsed["summary"], kind="interpretation",
                                     citations=parsed["summary_citations"]) if parsed["summary"] else None
            to_check = all_claims + ([summary_claim] if summary_claim else [])
            vr = await asyncio.to_thread(self.verifier.verify, to_check, ev_text, allowed_numbers=allowed,
                                         user_numbers=user_nums, pools=self._index_pools(evidence))
            kept_ids = {id(c) for c in vr.kept}
            del kept_ids
            kept_by_text = {(c.text, c.kind): c for c in vr.kept}
            sections = []
            for x in parsed["sections"]:
                ks = [kept_by_text[(c.text, c.kind)] for c in x.claims if (c.text, c.kind) in kept_by_text]
                sections.append(AnswerSection(heading=x.heading, claims=ks))
            sections = [x for x in sections if x.claims] or []
            summary = ""
            if summary_claim and (summary_claim.text, summary_claim.kind) in kept_by_text:
                sc = kept_by_text[(summary_claim.text, summary_claim.kind)]
                summary = sc.text + "".join(f" [{m}]" for m in sc.citations[:3])
            removed = [c for c in vr.removed if c is not summary_claim]
            s.output = {"checked": len(to_check), "kept": len(vr.kept), "removed": len(vr.removed),
                        "dropped_citations": vr.dropped_citations, "numeric_failures": vr.numeric_failures,
                        "thresholds": {"citation": 0.42, "fact": 0.50, "interpretation": 0.30},
                        "method": "max(lexical containment, semantic max-sim) against the cited passages only",
                        "removed_claims": [{"text": truncate(c.text, 160), "reason": c.removed_reason}
                                           for c in vr.removed][:12]}
            if vr.removed:
                s.status = "flagged"

        table = self._check_table(parsed["table"], ev_text, allowed)
        kept_claims = [c for x in sections for c in x.claims]
        if not kept_claims:
            return ResearchResult(turn_id="", status="abstained", style=style, question=req.question,
                                  resolved_query=u.resolved, evidence=evidence, removed_claims=removed,
                                  path="llm", model=settings.research_model,
                                  missing_information=(parsed["missing"] or "")
                                  + " Every generated claim failed citation verification, so no answer is shown.",
                                  follow_ups=parsed["follow_ups"])

        cited = {m for c in kept_claims for m in c.citations}
        cited |= {i.marker for c in calcs for i in c.inputs if i.marker}
        for e in evidence:
            e.cited = e.marker in cited
        self._pii(sections, report)
        report.groundedness = vr.coverage
        trust = self._trust(vr.coverage, vr.weakest, primary, len({e.document_id or e.doc_id for e in evidence
                                                                    if e.cited}), report)
        with st.run("guard.output", "Groundedness, PII & trust") as s:
            s.output = {"coverage": vr.coverage, "weakest_claim": vr.weakest, "trust": trust,
                        "note": "long answers are not withheld on the trust floor; failing claims are removed instead"}
            if trust < settings.trust_abstain_below:
                s.status = "flagged"
                warnings.append(f"composite trust {trust:.0f} is below {settings.trust_abstain_below:.0f} - "
                                "read the cited passages before relying on this answer")

        with st.run("contradictions", "Contradiction detection") as s:
            det = contra.detect_numeric(evidence)
            model_c, rejected = contra.verify_model(parsed["contradictions"], {e.marker: e for e in evidence})
            contradictions = contra.merge(det, model_c)
            s.output = {"deterministic": len(det), "model_reported": len(parsed["contradictions"]),
                        "model_accepted": len(model_c), "model_rejected_unverifiable": rejected}

        status = "partial" if any(c.kind == "fact" for c in removed) else "answered"
        return ResearchResult(
            turn_id="", status=status, style=style, question=req.question, resolved_query=u.resolved,  # type: ignore[arg-type]
            summary=summary, sections=sections, table=table, removed_claims=removed,
            answer_markdown=to_markdown(summary, sections, table, calcs), evidence=evidence,
            contradictions=contradictions, calculations=calcs, follow_ups=parsed["follow_ups"],
            missing_information=parsed["missing"], trust_score=trust, groundedness=vr.coverage, path="llm",
            model=settings.research_model,
        )

    def _index_pools(self, evidence: list[EvidenceItem]) -> dict[int, tuple[list[str], np.ndarray]]:
        """Sentence texts + vectors for each passage, read from the index (computed at ingest)."""
        pools: dict[int, tuple[list[str], np.ndarray]] = {}
        for e in evidence:
            col = self.registry.get(e.collection_id)
            if col is None:
                continue
            row = col.index.row_of(e.chunk_id)
            if row is None:
                continue
            rec = col.index.record(row)
            pairs = col.index.sentences_of(rec)
            if e.context_text:
                # Sentences of the section that fall inside the context window.
                doc = col.index.docs.get(rec.doc_id)
                if doc is not None:
                    win = e.context_text.strip("…")
                    a = doc.text.find(win[:80])
                    b = a + len(win) if a >= 0 else -1
                    allp = list(zip(doc.sentences, col.index.sent_vectors[doc.sent_offset:doc.sent_offset + doc.sent_count],
                                    strict=False))
                    if a >= 0:
                        pairs = [(s, v) for s, v in allp if s.end > a and s.start < b] or pairs
            if pairs and any(float(np.abs(v).sum()) > 0 for _, v in pairs):
                pools[e.marker] = ([s.text for s, _ in pairs], np.vstack([v for _, v in pairs]))
        return pools

    @staticmethod
    def _check_table(table: AnswerTable | None, ev_text: dict[int, str], allowed: set[str]) -> AnswerTable | None:
        if table is None:
            return None
        all_nums = set().union(*(set(numbers_in(t)) for t in ev_text.values())) if ev_text else set()
        rows = []
        for r in table.rows:
            cells = []
            for cell in r:
                body = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", cell)
                nums = set(numbers_in(body))
                bad = [n for n in nums if n not in all_nums and n not in allowed]
                cells.append(cell + (" ⚠ unverified" if bad else ""))
            rows.append(cells)
        return AnswerTable(columns=table.columns, rows=rows)

    def _pii(self, sections: list[AnswerSection], report: GuardReport) -> None:
        flagged = 0
        for s in sections:
            for c in s.claims:
                v = self.pii.check(GateContext(query="", answer=AnswerDraft(text=c.text, lang="en")))
                if v.decision is not Decision.ALLOW and self.pii.redacted_text:
                    c.text = self.pii.redacted_text
                    flagged += 1
        report.add(GuardVerdict(gate="output.pii", stage="output",
                                decision=Decision.FLAG if flagged else Decision.ALLOW, score=1.0,
                                reason=f"redacted PII in {flagged} claim(s)" if flagged else "no PII detected"))

    def _trust(self, coverage: float, weakest: float, primary: RetrievalRun | None, n_sources: int,
               report: GuardReport) -> float:
        """The v1 trust composite (guardrails/stack.py), applied to the research answer."""
        top = float(primary.result.top_score) if primary else 0.0
        margin = float(primary.result.margin) if primary else 0.0
        safety_clear = 1.0
        for v in report.verdicts:
            if v.gate in {"input.safety", "input.injection"} and v.decision is not Decision.ALLOW:
                safety_clear = 0.35 if v.decision is Decision.FLAG else 0.0
        clip = lambda x: float(np.clip(x, 0.0, 1.0))  # noqa: E731
        score = (40 * clip(coverage) + 18 * clip(top / 0.85) + 12 * clip(margin / 0.18) + 10 * clip(weakest)
                 + 10 * clip(n_sources / 2.0) + 10 * safety_clear)
        report.trust_score = round(score, 2)
        return report.trust_score

    @staticmethod
    def _fallback_followups(u: Understanding, evidence: list[EvidenceItem]) -> list[str]:
        """Template follow-ups (no LLM): other documents in the evidence, named honestly."""
        topic = " ".join(content_tokens(u.resolved)[:5])
        out: list[str] = []
        asked = normalize(u.resolved).strip(" ?.")
        for e in evidence:
            t = (e.title or "").split(" — ")[0]
            if not t or len(out) >= 2 or any(t in o for o in out):
                continue
            if e.document_id:
                out.append(f"What does '{truncate(t, 70)}' say about {topic}?")
            elif normalize(t).strip(" ?.") != asked and len(t) > 8:
                # MS MARCO passages carry the question they were collected for.
                out.append(t.strip().rstrip("?") + "?")
        if u.entities and len(out) < 3:
            out.append(f"Compare {u.entities[0]} with another source on {topic}")
        return out[:3]

    # ===================================================================== finish
    async def _finish(self, res: ResearchResult, req: ResearchRequest, session_id: str, st: Stages) -> ResearchResult:
        res.stages = st.records
        total = st.total()
        gen = sum(s.duration_ms for s in st.records if s.key == "generate")
        ver = sum(s.duration_ms for s in st.records if s.key.startswith("verify"))
        res.timings = {**res.timings, "total_ms": total, "generation_ms": round(gen, 2),
                       "verification_ms": round(ver, 2)}
        res.stages.append(StageRecord(key="final", name="Final answer", started_ms=total,
                                      output={"status": res.status, "claims": sum(len(s.claims) for s in res.sections),
                                              "removed_claims": len(res.removed_claims),
                                              "citations": sum(1 for e in res.evidence if e.cited),
                                              "trust": res.trust_score, "path": res.path}))
        if req.persist:
            await asyncio.to_thread(self._persist, res, req, session_id)
        return res

    def _persist(self, res: ResearchResult, req: ResearchRequest, session_id: str) -> None:
        if res.conversation_id is None:
            conv = self.conversations.create(session_id=session_id, title=truncate(req.question, 80),
                                             collection_ids=req.collection_ids)
            res.conversation_id = conv["id"]
        self.conversations.add_message(res.conversation_id, role="user", content=req.question, style=res.style.value,
                                       lang=res.lang, turn_id=res.turn_id)
        payload = res.model_dump(mode="json", exclude={"stages"})
        content = res.answer_markdown or res.summary or res.missing_information or res.status
        msg = self.conversations.add_message(res.conversation_id, role="assistant", content=content,
                                             resolved_query=res.resolved_query, style=res.style.value, lang=res.lang,
                                             turn_id=res.turn_id, payload=payload)
        res.message_id = msg["id"]
        self.traces.save(turn_id=res.turn_id, session_id=session_id, conversation_id=res.conversation_id,
                         kind="research", query=req.question, status=res.status, style=res.style.value,
                         pipeline_ms=res.timings.get("total_ms", 0.0), result=res.model_dump(mode="json"),
                         retain=settings.trace_db_retain)


__all__ = ["ResearchService", "RetrievalRun", "Stages"]
