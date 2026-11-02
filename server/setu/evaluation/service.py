"""Evaluation lab: datasets, reproducible runs, ablations, red team, exports.

Every number on the evaluation dashboard comes from a stored run in ``eval_runs`` /
``eval_results``. Runs record their full configuration, so re-running one reproduces it
(given the same index). Label provenance travels with every dataset and item:

* ``gold-msmarco`` - MS MARCO's own human ``is_selected`` annotations.
* ``curated`` - written by the SETU maintainers against the curated business collection;
  plausible but **not independently human-reviewed**.
* ``llm`` judgements (optional) are stored with ``judge="llm"`` and never mixed into the
  automated or human numbers.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import time
from typing import Any

import numpy as np

from ..config import settings
from ..embed import get_encoder
from ..knowledge.registry import BUILTIN_ID, CollectionRegistry
from ..research.models import ResearchRequest, ResearchStyle, RetrievalConfig
from ..storage.db import utcnow
from ..storage.repos import EvalRepo
from . import metrics as M

log = logging.getLogger("setu.eval")

# Ablation presets. Each one changes exactly one thing relative to "hybrid".
PRESETS: dict[str, dict[str, Any]] = {
    "hybrid": {"mode": "hybrid"},
    "dense-only": {"mode": "dense"},
    "sparse-only": {"mode": "sparse"},
    "hybrid-no-rerank": {"mode": "hybrid", "rerank": False},
    "hybrid-forced-rerank": {"mode": "hybrid", "rerank": True},
    "hybrid-no-expansion": {"mode": "hybrid", "expand": False},
    "hybrid+prf": {"mode": "hybrid", "query_expansion": True},
}


class EvaluationService:
    def __init__(self, *, research: Any, registry: CollectionRegistry) -> None:
        self.research = research
        self.registry = registry
        self.repo = EvalRepo()
        self._tasks: dict[str, asyncio.Task] = {}

    # ================================================================= datasets
    def seed_msmarco(self) -> dict[str, Any] | None:
        col = self.registry.get(BUILTIN_ID)
        if col is None:
            return None
        groups: dict[int, dict[str, Any]] = {}
        for did, rec in col.index.docs.items():
            qid = rec.meta.get("query_id")
            if qid is None:
                continue
            g = groups.setdefault(int(qid), {"selected": set(), "questions": {}})
            if rec.meta.get("is_selected"):
                g["selected"].add(did)
            q = str(rec.meta.get("query") or "").strip()
            if q and rec.lang not in g["questions"]:
                g["questions"][rec.lang] = q
        items = []
        for qid, g in sorted(groups.items()):
            if not g["selected"]:
                continue
            for lang, q in sorted(g["questions"].items()):
                items.append({"question": q, "lang": lang, "relevant_doc_ids": sorted(g["selected"]),
                              "reference_answer": None, "label_source": "gold-msmarco",
                              "tags": [f"qid:{qid}", f"lang:{lang}"]})
        return self.repo.upsert_dataset(
            name="msmarco-sample-gold", collection_ids=[BUILTIN_ID], label_source="gold-msmarco",
            description="Every answerable source question in the bundled MS MARCO-XI sample, in every indexed "
                        "language. Relevant passages = MS MARCO's human is_selected labels for that question, in any "
                        "language (the corpus is parallel). Document-level relevance.",
            items=items,
        )

    def upsert_curated(self, *, name: str, description: str, collection_ids: list[str],
                       items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.repo.upsert_dataset(name=name, description=description, collection_ids=collection_ids,
                                        label_source="curated", items=items)

    # ===================================================================== runs
    def start(self, *, dataset_id: str, name: str | None, presets: list[str], top_k: int,
              with_answers: bool, style: str = "quick", limit: int | None = None) -> dict[str, Any]:
        ds = self.repo.dataset(dataset_id)
        if ds is None:
            raise ValueError("dataset not found")
        bad = [p for p in presets if p not in PRESETS]
        if bad:
            raise ValueError(f"unknown preset(s): {', '.join(bad)}")
        config = {"presets": presets or ["hybrid"], "top_k": top_k, "with_answers": with_answers,
                  "style": style, "limit": limit, "collection_ids": ds["collection_ids"],
                  "embed_model": settings.embed_model, "llm_model": settings.research_model if with_answers
                  and style != "quick" else None}
        run = self.repo.create_run(dataset_id=dataset_id, name=name or f"{ds['name']} · {', '.join(config['presets'])}",
                                   config=config)
        self._tasks[run["id"]] = asyncio.create_task(self._execute(run["id"]))
        return run

    async def _execute(self, rid: str) -> None:
        run = self.repo.run(rid)
        assert run is not None
        cfg = run["config"]
        self.repo.update_run(rid, status="running", started_at=utcnow())
        try:
            items = self.repo.items(run["dataset_id"])
            if cfg.get("limit"):
                items = items[: int(cfg["limit"])]
            cols, _ = self.research.resolve_collections(cfg.get("collection_ids") or [])
            if not cols:
                raise ValueError("none of the dataset's collections are loaded")
            summary: dict[str, Any] = {"presets": {}, "items": len(items)}
            rows: list[dict[str, Any]] = []
            for preset in cfg["presets"]:
                pm, prow = await self._run_preset(preset, cfg, items, cols)
                summary["presets"][preset] = pm
                rows.extend(prow)
            if cfg.get("with_answers"):
                am, arows = await self._run_answers(cfg, items)
                summary["answers"] = am
                rows.extend(arows)
            self.repo.add_results(rid, rows)
            self.repo.update_run(rid, status="completed", metrics=summary, finished_at=utcnow())
        except Exception as exc:  # noqa: BLE001
            log.exception("evaluation run %s failed", rid)
            self.repo.update_run(rid, status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=utcnow())

    async def _run_preset(self, preset: str, cfg: dict[str, Any], items: list[dict[str, Any]],
                          cols: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rc = RetrievalConfig(top_k=max(10, int(cfg["top_k"])), **PRESETS[preset])
        per: list[dict[str, Any]] = []
        lat: list[float] = []

        def one(it: dict[str, Any]) -> dict[str, Any]:
            t = time.perf_counter()
            runs = self.research.retrieve([(it["question"], None)], cols, rc, lang=it.get("lang"),
                                          capture=False, budget_ms=10_000.0)
            ms = (time.perf_counter() - t) * 1000
            merged = self.research.merge(runs, rc.top_k, 1)
            ranked = [(h.document_id or h.doc_id) for h, _ in merged]
            rel = set(it.get("relevant_doc_ids") or [])
            m = M.retrieval_metrics(ranked, rel)
            return {"item_id": it["id"], "question": it["question"], "judge": "automated",
                    "metrics": {"preset": preset, **m, "retrieval_ms": round(ms, 3), "lang": it.get("lang"),
                                "top": M.dedup(ranked)[:5], "relevant": sorted(rel)[:10]}}

        for it in items:
            row = await asyncio.to_thread(one, it)
            per.append(row)
            lat.append(row["metrics"]["retrieval_ms"])
        keys = [k for k in per[0]["metrics"] if "@" in k] if per else []
        agg = {k: M.mean([r["metrics"][k] for r in per]) for k in keys}
        agg.update({"retrieval_ms_p50": M.percentile(lat, 50), "retrieval_ms_p95": M.percentile(lat, 95),
                    "retrieval_ms_mean": M.mean(lat), "queries": len(per)})
        by_lang: dict[str, list[dict[str, Any]]] = {}
        for r in per:
            by_lang.setdefault(r["metrics"].get("lang") or "?", []).append(r)
        agg["by_lang"] = {lang: {"recall@5": M.mean([r["metrics"]["recall@5"] for r in rs]),
                                 "mrr@10": M.mean([r["metrics"]["mrr@10"] for r in rs]), "n": len(rs)}
                          for lang, rs in sorted(by_lang.items())}
        return agg, per

    async def _run_answers(self, cfg: dict[str, Any], items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        enc = get_encoder()
        rows: list[dict[str, Any]] = []
        for it in items:
            req = ResearchRequest(question=it["question"], lang=it.get("lang"), style=ResearchStyle(cfg["style"]),
                                  collection_ids=cfg.get("collection_ids") or [], persist=False,
                                  retrieval=RetrievalConfig(top_k=int(cfg["top_k"])))
            t = time.perf_counter()
            res = await self.research.ask(req, session_id="evaluation")
            ms = (time.perf_counter() - t) * 1000
            rel = set(it.get("relevant_doc_ids") or [])
            text = " ".join([res.summary] + [c.text for s in res.sections for c in s.claims])
            cited_docs = {e.document_id or e.doc_id for e in res.evidence if e.cited}
            ctx_docs = [e.document_id or e.doc_id for e in res.evidence]
            ans_rel = 0.0
            if text.strip():
                qv = enc.encode_query(it["question"])
                av = enc.encode([text[:1500]])[0]
                ans_rel = float(np.clip(qv @ av, 0.0, 1.0))
            m = {
                "status": res.status, "path": res.path, "end_to_end_ms": round(ms, 1),
                "faithfulness": res.groundedness if text.strip() else None,
                "citation_precision": round(len(cited_docs & rel) / len(cited_docs), 4) if cited_docs and rel else None,
                "context_relevance": round(sum(1 for d in ctx_docs if d in rel) / len(ctx_docs), 4) if ctx_docs and rel else None,
                "answer_relevance": round(ans_rel, 4),
                "reference_recall": M.reference_recall(text, it["reference_answer"]) if it.get("reference_answer") else None,
                "removed_claims": len(res.removed_claims), "trust": res.trust_score,
                "answered": res.status in ("answered", "partial"),
                "abstained_with_relevant_retrieved": res.status == "abstained" and bool(set(ctx_docs) & rel),
            }
            rows.append({"item_id": it["id"], "question": it["question"], "judge": "automated",
                         "metrics": {"preset": "answers", **m}})
        ms_all = [r["metrics"]["end_to_end_ms"] for r in rows]
        agg = {
            "faithfulness": M.mean([r["metrics"]["faithfulness"] for r in rows if r["metrics"]["faithfulness"] is not None]),
            "citation_precision": M.mean([r["metrics"]["citation_precision"] for r in rows
                                          if r["metrics"]["citation_precision"] is not None]),
            "context_relevance": M.mean([r["metrics"]["context_relevance"] for r in rows
                                         if r["metrics"]["context_relevance"] is not None]),
            "answer_relevance": M.mean([r["metrics"]["answer_relevance"] for r in rows]),
            "reference_recall": M.mean([r["metrics"]["reference_recall"] for r in rows
                                        if r["metrics"]["reference_recall"] is not None]),
            "answered_rate": M.mean([1.0 if r["metrics"]["answered"] else 0.0 for r in rows]),
            "abstain_wrong": sum(1 for r in rows if r["metrics"]["abstained_with_relevant_retrieved"]),
            "end_to_end_ms_p50": M.percentile(ms_all, 50), "end_to_end_ms_p95": M.percentile(ms_all, 95),
            "queries": len(rows), "judge": "automated heuristics (see docs/EVALUATION.md)",
        }
        return agg, rows

    # =================================================================== export
    def export(self, rid: str, fmt: str) -> tuple[str, str]:
        run = self.repo.run(rid)
        if run is None:
            raise ValueError("run not found")
        results = self.repo.results(rid)
        if fmt == "json":
            return json.dumps({"run": run, "results": results}, ensure_ascii=False, indent=2, default=str), \
                "application/json"
        buf = io.StringIO()
        keys: list[str] = []
        for r in results:
            for k in r["metrics"]:
                if k not in keys and not isinstance(r["metrics"][k], (list, dict)):
                    keys.append(k)
        w = csv.writer(buf)
        w.writerow(["run_id", "item_id", "judge", "question", *keys])
        for r in results:
            w.writerow([rid, r["item_id"], r["judge"], r["question"], *[r["metrics"].get(k) for k in keys]])
        return buf.getvalue(), "text/csv"


__all__ = ["PRESETS", "EvaluationService"]
