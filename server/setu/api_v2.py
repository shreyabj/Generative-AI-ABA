"""SETU 2.0 API routes (mounted under ``/api/v1`` next to the v1 routes).

Every list route is paginated; every mutating or paid route goes through
:func:`setu.security.require_token`; conversation and trace routes are scoped to the
caller's ``X-SETU-Session``. Responses are plain JSON built from the database and the
live registry - nothing here is a placeholder value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import __version__
from .config import REPO_ROOT, settings
from .evaluation import redteam
from .evaluation.service import PRESETS
from .knowledge.service import KnowledgeError
from .research.models import ResearchRequest, ResearchResult
from .security import rate_limit, require_token, session_id
from .storage.repos import ChunkRepo, ConversationRepo, DocumentRepo, EmbeddingCacheRepo, JobRepo, TraceRepo
from .telemetry import metrics

router = APIRouter(prefix="/api/v1")
STATE: dict[str, Any] = {}


def svc(name: str) -> Any:
    s = STATE.get(name)
    if s is None:
        raise HTTPException(status_code=503, detail={"error": "SETU is still starting", "service": name})
    return s


def _kerr(exc: KnowledgeError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


# ================================================================== models
class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class CollectionPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class UrlsIn(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=20)


class ConversationIn(BaseModel):
    title: str = Field(default="New research", max_length=200)
    collection_ids: list[str] = Field(default_factory=list, max_length=20)


class ConversationPatch(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    collection_ids: list[str] | None = None


class EvalRunIn(BaseModel):
    dataset_id: str
    name: str | None = Field(default=None, max_length=200)
    presets: list[str] = Field(default_factory=lambda: ["hybrid"], max_length=10)
    top_k: int = Field(default=10, ge=1, le=30)
    with_answers: bool = False
    style: str = Field(default="quick", pattern="^(quick|detailed|comparative|critical|executive)$")
    limit: int | None = Field(default=None, ge=1, le=500)


# =============================================================== dashboard
@router.get("/dashboard/summary")
async def dashboard(session: str = Depends(session_id)) -> dict[str, Any]:
    ks = svc("knowledge")
    cols = ks.collections.list()
    reg = svc("registry")
    m = metrics.report(include_histogram=False)
    eval_runs = svc("evaluation").repo.runs(limit=5)
    latest = next((r for r in eval_runs if r["status"] == "completed"), None)
    traces = TraceRepo()
    recent_traces = traces.list(session_id=session, limit=6)
    p = STATE.get("pipeline")
    return {
        "version": __version__,
        "knowledge": ks.overview(),
        "collections": [{**c, "loaded": reg.get(c["id"]) is not None} for c in cols],
        "layers": ChunkRepo().layer_counts(),
        "builtin_layers": {lay.value: int(mask.sum()) for lay, mask in p.index.layer_masks.items()} if p else {},
        "embedding": {"model": settings.embed_model, "dim": settings.embed_dim,
                      "backend": p.encoder.backend if p else None, "cache_vectors": EmbeddingCacheRepo().count(),
                      "query_cache": p.encoder.stats().get("cache") if p else None},
        "vector_store": p.index.vector_store.stats() if p else None,
        "retrieval": {"live": {k: m.get(k) for k in ("count", "pipeline_ms", "slo_compliance", "status_counts")},
                      "latest_eval": latest},
        "research": {**traces.stats(), "conversations": ConversationRepo().count()},
        "recent_sessions": ConversationRepo().list(session, limit=6),
        "recent_traces": recent_traces,
        "recent_documents": DocumentRepo().recent(8),
        "recent_jobs": JobRepo().list(limit=8),
        "llm": {"enabled": settings.llm_enabled, "model": settings.research_model,
                "fast_model": settings.research_fast_model,
                "breaker": svc("research").llm.breaker.snapshot()},
        "speech": {"enabled": settings.stt_enabled},
        "health": {"pipeline": p is not None, "worker": bool(ks._worker and ks._worker.is_alive()),
                   "database": settings.db_url.split(":", 1)[0]},
    }


# ============================================================= collections
@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    ks = svc("knowledge")
    reg = svc("registry")
    out = []
    for c in ks.collections.list():
        loaded = reg.get(c["id"])
        out.append({**c, "loaded": loaded is not None,
                    "languages": loaded.index.languages() if loaded else [],
                    "last_rebuild": ks.last_rebuild.get(c["id"])})
    return {"collections": out, "total": len(out)}


@router.post("/collections", dependencies=[Depends(require_token)])
async def create_collection(body: CollectionIn) -> dict[str, Any]:
    try:
        return svc("knowledge").create_collection(body.name, body.description)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.get("/collections/{cid}")
async def get_collection(cid: str) -> dict[str, Any]:
    ks = svc("knowledge")
    try:
        c = ks.get_collection(cid)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc
    loaded = svc("registry").get(cid)
    return {**c, "loaded": loaded is not None, "index": loaded.index.stats() if loaded else None,
            "languages": loaded.index.languages() if loaded else [],
            "layers": ChunkRepo().layer_counts(cid) if c["kind"] != "builtin" else
            ({lay.value: int(m.sum()) for lay, m in loaded.index.layer_masks.items()} if loaded else {}),
            "jobs": ks.jobs.list(collection_id=cid, limit=20), "last_rebuild": ks.last_rebuild.get(cid)}


@router.patch("/collections/{cid}", dependencies=[Depends(require_token)])
async def patch_collection(cid: str, body: CollectionPatch) -> dict[str, Any]:
    try:
        return svc("knowledge").update_collection(cid, name=body.name, description=body.description)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.delete("/collections/{cid}", dependencies=[Depends(require_token)])
async def delete_collection(cid: str) -> dict[str, str]:
    try:
        svc("knowledge").delete_collection(cid)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc
    return {"deleted": cid}


@router.post("/collections/{cid}/rebuild", dependencies=[Depends(require_token)])
async def rebuild_collection(cid: str) -> dict[str, Any]:
    try:
        return svc("knowledge").rebuild(cid)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.get("/collections/{cid}/documents")
async def list_documents(cid: str, status: str | None = Query(default=None), q: str | None = Query(default=None),
                         limit: int = Query(default=25, ge=1, le=200), offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    ks = svc("knowledge")
    try:
        col = ks.get_collection(cid)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc
    if col["kind"] == "builtin":
        loaded = svc("registry").get(cid)
        docs = list(loaded.index.docs.values()) if loaded else []
        if q:
            docs = [d for d in docs if q.lower() in (str(d.meta.get("query") or "") + d.text).lower()]
        page = docs[offset:offset + limit]
        return {"documents": [{"id": d.doc_id, "title": d.meta.get("query"), "lang": d.lang, "status": "completed",
                               "source_type": "builtin", "char_count": len(d.text),
                               "chunk_count": len(loaded.index.rows_of_doc(d.doc_id)), "meta": d.meta}
                              for d in page], "total": len(docs), "limit": limit, "offset": offset, "read_only": True}
    rows, total = ks.documents.list(cid, status=status, q=q, limit=limit, offset=offset)
    return {"documents": rows, "total": total, "limit": limit, "offset": offset, "read_only": False}


@router.post("/collections/{cid}/documents", dependencies=[Depends(require_token), Depends(rate_limit("upload", 30))])
async def upload_documents(cid: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    ks = svc("knowledge")
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(status_code=413, detail=f"at most {settings.max_files_per_upload} files per upload")
    limit = int(settings.max_upload_mb * 1024 * 1024)
    out, errors = [], []
    for f in files:
        data = await f.read(limit + 1)
        await f.close()
        try:
            if len(data) > limit:
                raise KnowledgeError(f"'{f.filename}' is larger than the {settings.max_upload_mb:.0f} MB limit", 413)
            out.append(ks.add_upload(cid, f.filename or "upload", data, f.content_type))
        except KnowledgeError as exc:
            if exc.status == 404:
                raise _kerr(exc) from exc
            errors.append({"filename": f.filename, "error": str(exc), "status": exc.status})
    return {"accepted": out, "rejected": errors}


@router.post("/collections/{cid}/urls", dependencies=[Depends(require_token), Depends(rate_limit("url", 30))])
async def import_urls(cid: str, body: UrlsIn) -> dict[str, Any]:
    ks = svc("knowledge")
    out, errors = [], []
    for u in body.urls:
        try:
            out.append(ks.add_url(cid, u))
        except KnowledgeError as exc:
            if exc.status == 404:
                raise _kerr(exc) from exc
            errors.append({"url": u, "error": str(exc)})
    return {"accepted": out, "rejected": errors}


@router.get("/documents/{did}")
async def get_document(did: str) -> dict[str, Any]:
    ks = svc("knowledge")
    try:
        d = ks.get_document(did)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc
    jobs = [j for j in ks.jobs.list(collection_id=d["collection_id"], limit=50) if j["document_id"] == did]
    return {**d, "jobs": jobs[:10]}


@router.delete("/documents/{did}", dependencies=[Depends(require_token)])
async def delete_document(did: str) -> dict[str, Any]:
    try:
        return svc("knowledge").delete_document(did)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.post("/documents/{did}/reindex", dependencies=[Depends(require_token)])
async def reindex_document(did: str) -> dict[str, Any]:
    try:
        return svc("knowledge").reindex_document(did)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.get("/documents/{did}/chunks")
async def document_chunks(did: str, layer: str | None = Query(default=None, pattern="^(atomic|semantic|window|document)$"),
                          limit: int = Query(default=25, ge=1, le=200), offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    try:
        return svc("knowledge").document_chunks(did, layer=layer, limit=limit, offset=offset)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.get("/jobs")
async def list_jobs(collection_id: str | None = None, active: bool = False,
                    limit: int = Query(default=30, ge=1, le=200)) -> dict[str, Any]:
    states = ("uploaded", "queued", "extracting", "chunking", "embedding", "indexing") if active else None
    return {"jobs": JobRepo().list(collection_id=collection_id, states=states, limit=limit)}


@router.get("/jobs/{jid}")
async def get_job(jid: str) -> dict[str, Any]:
    j = JobRepo().get(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="job not found")
    return j


# ================================================================ research
@router.post("/research/ask", response_model=ResearchResult,
             dependencies=[Depends(require_token), Depends(rate_limit("research", 40))])
async def research_ask(req: ResearchRequest, session: str = Depends(session_id)) -> ResearchResult:
    try:
        return await svc("research").ask(req, session_id=session)
    except KnowledgeError as exc:
        raise _kerr(exc) from exc


@router.get("/conversations")
async def list_conversations(session: str = Depends(session_id), limit: int = Query(default=50, ge=1, le=200),
                             offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    return {"conversations": ConversationRepo().list(session, limit=limit, offset=offset)}


@router.post("/conversations", dependencies=[Depends(require_token)])
async def create_conversation(body: ConversationIn, session: str = Depends(session_id)) -> dict[str, Any]:
    return ConversationRepo().create(session_id=session, title=body.title, collection_ids=body.collection_ids)


def _own(cid: str, session: str) -> dict[str, Any]:
    c = ConversationRepo().get(cid)
    if c is None or c["session_id"] != session:
        raise HTTPException(status_code=404, detail="conversation not found")
    return c


@router.get("/conversations/{cid}")
async def get_conversation(cid: str, session: str = Depends(session_id)) -> dict[str, Any]:
    c = _own(cid, session)
    return {**c, "messages": ConversationRepo().messages(cid)}


@router.patch("/conversations/{cid}", dependencies=[Depends(require_token)])
async def patch_conversation(cid: str, body: ConversationPatch, session: str = Depends(session_id)) -> dict[str, Any]:
    _own(cid, session)
    vals = {k: v for k, v in body.model_dump().items() if v is not None}
    if vals:
        ConversationRepo().update(cid, **vals)
    return _own(cid, session)


@router.delete("/conversations/{cid}", dependencies=[Depends(require_token)])
async def delete_conversation(cid: str, session: str = Depends(session_id)) -> dict[str, str]:
    _own(cid, session)
    ConversationRepo().delete(cid)
    return {"deleted": cid}


@router.get("/traces")
async def list_traces(session: str = Depends(session_id), limit: int = Query(default=30, ge=1, le=200),
                      offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    return {"traces": TraceRepo().list(session_id=session, limit=limit, offset=offset)}


@router.get("/traces/{turn_id}")
async def get_research_trace(turn_id: str, session: str = Depends(session_id)) -> dict[str, Any]:
    t = TraceRepo().get(turn_id)
    if t is None or t["session_id"] != session:
        raise HTTPException(status_code=404, detail="trace not found for this session")
    return t


# ============================================================== evaluation
@router.get("/eval/presets")
async def eval_presets() -> dict[str, Any]:
    return {"presets": PRESETS}


@router.get("/eval/datasets")
async def eval_datasets() -> dict[str, Any]:
    return {"datasets": svc("evaluation").repo.datasets()}


@router.get("/eval/datasets/{did}/items")
async def eval_items(did: str, limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    items = svc("evaluation").repo.items(did)
    return {"items": items[offset:offset + limit], "total": len(items)}


@router.post("/eval/runs", dependencies=[Depends(require_token), Depends(rate_limit("eval", 10))])
async def eval_start(body: EvalRunIn) -> dict[str, Any]:
    try:
        return svc("evaluation").start(dataset_id=body.dataset_id, name=body.name, presets=body.presets,
                                       top_k=body.top_k, with_answers=body.with_answers, style=body.style,
                                       limit=body.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/eval/runs")
async def eval_runs(limit: int = Query(default=30, ge=1, le=200)) -> dict[str, Any]:
    return {"runs": svc("evaluation").repo.runs(limit)}


@router.get("/eval/runs/{rid}")
async def eval_run(rid: str) -> dict[str, Any]:
    r = svc("evaluation").repo.run(rid)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")
    return r


@router.get("/eval/runs/{rid}/results")
async def eval_results(rid: str, limit: int = Query(default=200, ge=1, le=2000), offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    rows = svc("evaluation").repo.results(rid)
    return {"results": rows[offset:offset + limit], "total": len(rows)}


@router.get("/eval/runs/{rid}/export")
async def eval_export(rid: str, format: str = Query(default="json", pattern="^(json|csv)$")) -> Response:
    try:
        body, mime = svc("evaluation").export(rid, format)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=body, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="setu-eval-{rid}.{format}"'})


@router.delete("/eval/runs/{rid}", dependencies=[Depends(require_token)])
async def eval_delete(rid: str) -> dict[str, str]:
    svc("evaluation").repo.delete_run(rid)
    return {"deleted": rid}


@router.post("/eval/redteam", dependencies=[Depends(require_token), Depends(rate_limit("eval", 10))])
async def eval_redteam() -> dict[str, Any]:
    ks = svc("knowledge")
    col = ks.collections.by_name(redteam.FIXTURE_COLLECTION)
    if col is None or svc("registry").get(col["id"]) is None:
        raise HTTPException(status_code=409, detail="red-team fixtures are not loaded; run `python -m setu.cli seed`")
    out = await redteam.run(svc("research"), col["id"])
    STATE["last_redteam"] = out
    path = settings.data_dir / "redteam_v2.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


@router.get("/eval/redteam")
async def eval_redteam_last() -> dict[str, Any]:
    if STATE.get("last_redteam"):
        return STATE["last_redteam"]
    path = settings.data_dir / "redteam_v2.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"cases": 0, "results": [], "note": "not run yet"}


class SeedIn(BaseModel):
    business: bool = True
    redteam: bool = True


@router.post("/admin/seed", dependencies=[Depends(require_token)])
async def admin_seed(body: SeedIn) -> dict[str, Any]:
    """Queue the demonstration collections (real public sources + labelled synthetic fixtures)."""
    from .knowledge import seed

    ks = svc("knowledge")
    out: dict[str, Any] = {}
    if body.business:
        out["business"] = seed.seed_business(ks)
    if body.redteam:
        out["redteam"] = seed.seed_redteam(ks)
    return out


@router.post("/eval/datasets/business-curated", dependencies=[Depends(require_token)])
async def build_curated_dataset() -> dict[str, Any]:
    """(Re)build the curated business dataset against the documents that ingested successfully."""
    from .knowledge import seed

    ks = svc("knowledge")
    import json as _json

    name = _json.loads((settings.sources_dir / "business.json").read_text(encoding="utf-8"))["collection"]
    col = ks.collections.by_name(name)
    if col is None:
        raise HTTPException(status_code=409, detail="business collection not seeded yet")
    items = seed.curated_items(ks, col["id"])
    if not items:
        raise HTTPException(status_code=409, detail="no curated item has an ingested relevant document yet")
    return svc("evaluation").upsert_curated(
        name="business-curated", collection_ids=[col["id"]], items=items,
        description="Questions written by the SETU maintainers against the India Business & Economy 2026 "
                    "collection, with the documents that answer them. Curated, not independently human-reviewed.",
    )


@router.get("/eval/reports")
async def eval_reports() -> dict[str, Any]:
    """The v1 script reports committed in docs/ (latency, quality, red team), as stored."""
    out: dict[str, Any] = {}
    for name in ("latency", "quality", "redteam", "ablation"):
        p = Path(REPO_ROOT) / "docs" / f"{name}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data.pop("turns", None)
                data.pop("rows", None)
                if name == "redteam":
                    data["results"] = data.get("results", [])[:60]
                out[name] = data
            except (OSError, ValueError):
                out[name] = {"error": "unreadable"}
    return out


# ================================================================ settings
@router.get("/settings")
async def get_settings() -> dict[str, Any]:
    p = STATE.get("pipeline")
    ks = STATE.get("knowledge")
    return {
        "embedding": {"model": settings.embed_model, "dim": settings.embed_dim, "batch": settings.embed_batch,
                      "threads": settings.embed_threads, "query_cache_size": settings.embed_cache_size},
        "chunking": {"layers": sorted(settings.layer_set), "atomic_chars": [settings.atomic_min_chars,
                     settings.atomic_max_chars], "semantic_max_chars": settings.semantic_max_chars,
                     "semantic_z": settings.semantic_z, "window_chars": settings.window_chars,
                     "window_overlap": [settings.window_overlap_min, settings.window_overlap_max],
                     "metadata_prefix": settings.metadata_prefix,
                     "section_chars": [settings.section_target_chars, settings.section_max_chars]},
        "retrieval": {"dense_k": settings.dense_k, "sparse_k": settings.sparse_k, "fuse_k": settings.fuse_k,
                      "rerank_k": settings.rerank_k, "final_k": settings.final_k, "rrf_k": settings.rrf_k,
                      "dense_weight": settings.dense_weight, "sparse_weight": settings.sparse_weight,
                      "presets": PRESETS},
        "guardrails": {"topicality_min_score": settings.topicality_min_score,
                       "topicality_min_margin": settings.topicality_min_margin,
                       "groundedness_min_coverage": settings.groundedness_min_coverage,
                       "trust_abstain_below": settings.trust_abstain_below,
                       "safety_similarity_threshold": settings.safety_similarity_threshold},
        "generation": {"llm_enabled": settings.llm_enabled, "base_url": settings.llm_base_url,
                       "research_model": settings.research_model, "fast_model": settings.research_fast_model,
                       "deliberate_model": settings.llm_model, "reasoning_effort": settings.research_reasoning_effort,
                       "timeout_s": settings.research_timeout_s},
        "speech": {"enabled": settings.stt_enabled, "stt_model": settings.elevenlabs_stt_model,
                   "tts_model": settings.elevenlabs_tts_model, "fallback": "browser Web Speech API"},
        "vector_store": {"backend_setting": settings.vector_backend, "builtin": p.index.vector_store.stats() if p else None,
                         "builtin_error": p.index.backend_error if p else None},
        "storage": {"database": settings.db_url.split("://", 1)[0], "collections_dir": str(settings.collections_dir)},
        "limits": {"max_upload_mb": settings.max_upload_mb, "max_files_per_upload": settings.max_files_per_upload,
                   "max_url_mb": settings.max_url_mb, "rate_limit_per_min": settings.rate_limit_per_min},
        "security": {"api_token_required": bool(settings.api_token), "cors_origins": settings.cors_list},
        "slo_ms": settings.budget.total_ms,
        "diagnostics": {
            "pipeline_ready": p is not None,
            "ingest_worker_alive": bool(ks and ks._worker and ks._worker.is_alive()),
            "embedding_cache_vectors": EmbeddingCacheRepo().count(),
            "boot": STATE.get("boot"),
            "circuit_breakers": (p.registry.breakers() if p else []) + (
                [STATE["research"].llm.breaker.snapshot()] if STATE.get("research") else []),
        },
    }


__all__ = ["STATE", "router"]
