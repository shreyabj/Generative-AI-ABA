"""FastAPI application.

The API is deliberately transparent: every response carries the span waterfall, the
router's decision and its reason, every guardrail verdict with its score and threshold,
and the budget report including which optional stages were skipped. The frontend is a
renderer for that payload, not a place where explanations are invented - if the UI shows
a number, the server measured it.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api_v2 import STATE as V2
from .api_v2 import router as v2_router
from .config import settings
from .harness.contracts import AnswerMode, AskRequest, TurnResult
from .ingest.build import load_or_build
from .pipeline import SetuPipeline
from .security import rate_limit, require_token
from .speech.elevenlabs import VOICE_BY_LANG, SpeechError
from .telemetry import metrics
from .text import LANG_NAMES, detect_lang

STATE: dict[str, Any] = {"pipeline": None, "ready": False, "error": None, "warm": {}}


def _boot_v2(pipeline: SetuPipeline) -> dict[str, Any]:
    """SETU 2.0 services: metadata DB, collections, research, evaluation."""
    from .evaluation.service import EvaluationService
    from .knowledge.registry import CollectionRegistry
    from .knowledge.service import KnowledgeService
    from .research.llm import LLMClient
    from .research.service import ResearchService
    from .storage.db import get_engine

    get_engine()
    registry = CollectionRegistry()
    knowledge = KnowledgeService(registry)
    knowledge.register_builtin(pipeline.index)
    boot = knowledge.boot()
    research = ResearchService(pipeline=pipeline, registry=registry, llm=LLMClient())
    evaluation = EvaluationService(research=research, registry=registry)
    evaluation.seed_msmarco()
    V2.update({"pipeline": pipeline, "registry": registry, "knowledge": knowledge, "research": research,
               "evaluation": evaluation, "boot": boot})
    return boot


def get_pipeline() -> SetuPipeline:
    p = STATE.get("pipeline")
    if p is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "index not loaded",
                "reason": STATE.get("error"),
                "hint": "run: python -m setu.cli ingest --langs hi,ta --rows 300",
            },
        )
    return p


async def _boot() -> None:
    """Load the index and warm every model off the request path."""
    loop = asyncio.get_running_loop()
    try:
        index = await loop.run_in_executor(None, load_or_build)
        pipeline = SetuPipeline(index)
        STATE["warm"] = await loop.run_in_executor(None, pipeline.warm)
        pipeline.index.attach_vector_backend()
        STATE["pipeline"] = pipeline
        STATE["v2_boot"] = await loop.run_in_executor(None, _boot_v2, pipeline)
        STATE["ready"] = True
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        STATE["error"] = f"{type(exc).__name__}: {exc}"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_boot())
    yield
    task.cancel()
    p = STATE.get("pipeline")
    if p is not None:
        with contextlib.suppress(Exception):
            await p.speech.aclose()
            await p.deliberate.aclose()
    with contextlib.suppress(Exception):
        V2["knowledge"].stop()
        await V2["research"].llm.aclose()


app = FastAPI(
    title="SETU",
    description="Speech to Evidence to Truth to Utterance - a sub-200ms multilingual voice RAG engine.",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    # Credentials with a wildcard origin is both invalid per the CORS spec and unsafe;
    # the API authenticates with headers, not cookies, so credentials are never needed.
    allow_credentials="*" not in settings.cors_list,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-SETU-Token", "X-SETU-Session"],
)


@app.middleware("http")
async def security_headers(request, call_next):  # noqa: ANN001
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    return resp


app.include_router(v2_router)

API = "/api/v1"


# ==================================================================== meta routes
@app.get(f"{API}/health")
async def health() -> dict[str, Any]:
    p = STATE.get("pipeline")
    return {
        "status": "ready" if STATE["ready"] else ("error" if STATE["error"] else "warming"),
        "version": __version__,
        "error": STATE["error"],
        "warmup": STATE["warm"],
        "detail": p.health() if p else None,
    }


@app.get(f"{API}/config")
async def config() -> dict[str, Any]:
    p = STATE.get("pipeline")
    return {
        "app": settings.app_name,
        "tagline": settings.app_tagline,
        "version": __version__,
        "slo_ms": settings.budget.total_ms,
        "stt": {
            "provider": settings.stt_provider,
            "enabled": settings.stt_enabled,
            "model": settings.elevenlabs_stt_model,
        },
        "tts": {"enabled": settings.stt_enabled, "model": settings.elevenlabs_tts_model},
        "deliberate": {"enabled": settings.llm_enabled, "model": settings.llm_model},
        # Two different lists, because they answer two different questions.
        #
        # `languages`           which languages have passages in the index right now
        # `supported_languages` every language SETU can be *asked* in
        #
        # They differ, and the UI needs both. The encoder shares one vector space across
        # all 15, so a Kannada question retrieves English passages perfectly well even
        # when no Kannada passage is indexed - it just answers in the language it found.
        # Hiding the other 11 from the picker would hide a working feature; showing them
        # without marking coverage would imply passages that are not there.
        "languages": (
            [{"code": c, "name": LANG_NAMES.get(c, c)} for c in p.index.languages()] if p else []
        ),
        "supported_languages": [
            {"code": c, "name": n, "in_corpus": bool(p and c in set(p.index.languages()))}
            for c, n in LANG_NAMES.items()
        ],
        "modes": [m.value for m in AnswerMode],
        "budget": settings.budget.model_dump(),
        "thresholds": {
            "topicality_min_score": settings.topicality_min_score,
            "topicality_min_margin": settings.topicality_min_margin,
            "groundedness_min_coverage": settings.groundedness_min_coverage,
            "trust_abstain_below": settings.trust_abstain_below,
            "safety_similarity_threshold": settings.safety_similarity_threshold,
        },
    }


@app.get(f"{API}/corpus")
async def corpus() -> dict[str, Any]:
    p = get_pipeline()
    idx = p.index
    return {
        **idx.stats(),
        "languages": [{"code": c, "name": LANG_NAMES.get(c, c)} for c in idx.languages()],
        "layers": [
            {
                "layer": layer.value,
                "chunks": int(mask.sum()),
                "avg_chars": round(
                    float(sum(len(idx.chunks[i].text) for i in mask.nonzero()[0][:400]) / max(min(int(mask.sum()), 400), 1)),
                    1,
                ),
            }
            for layer, mask in idx.layer_masks.items()
        ],
    }


@app.get(f"{API}/tools")
async def tools() -> dict[str, Any]:
    p = get_pipeline()
    return {
        "count": len(p.registry.names()),
        "manifest": p.registry.manifest(),
        "circuit_breakers": p.registry.breakers(),
    }


@app.get(f"{API}/metrics")
async def metrics_route(histogram: bool = Query(default=True)) -> dict[str, Any]:
    return metrics.report(include_histogram=histogram)


@app.delete(f"{API}/metrics", dependencies=[Depends(require_token)])
async def metrics_reset() -> dict[str, str]:
    metrics.clear()
    return {"status": "cleared"}


@app.get(f"{API}/trace/{{turn_id}}")
async def trace(turn_id: str) -> TurnResult:
    p = get_pipeline()
    t = p.trace_of(turn_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"no retained trace for turn '{turn_id}'")
    return t


# ================================================================ question bank
#
# Every question the corpus was actually built from, in every language it was indexed in,
# with an honest flag for whether this index can answer it.
#
# The flag matters more than it sounds. This slice holds 24 source questions, and MS MARCO
# marks only some of them as having an answer present in their passages - the rest are
# genuinely unanswerable here, and offering one as a suggested example teaches a visitor
# that the system is broken when it correctly says "I don't know". So the examples the UI
# puts in front of people are drawn only from the verified subset, while the browser shows
# everything and labels which is which.
#
# Two sources of truth, strongest first:
#   gold      - data/corpus/queries.jsonl has a gold answer string and gold passage ids
#   selected  - a passage in this group carries MS MARCO's is_selected label


def _load_gold() -> dict[tuple[int, str], str]:
    """``(query_id, lang) -> gold answer``, for questions with evidence in the corpus."""
    path = settings.data_dir / "corpus" / "queries.jsonl"
    out: dict[tuple[int, str], str] = {}
    if not path.exists():
        return out
    with contextlib.suppress(OSError, ValueError):
        import json

        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if not row.get("gold_doc_ids"):
                    continue
                answer = str(row.get("gold_answer") or "").strip()
                english = str(row.get("gold_answer_en") or "")
                if not answer or "No Answer Present" in english:
                    continue
                qid, lang = row.get("query_id"), row.get("lang")
                if qid is not None and lang:
                    out[(int(qid), str(lang))] = answer
                # The gold set is written per Indic language; the English question rides
                # along on every row, so record it once too.
                eng = str(row.get("eng_query") or "").strip()
                if eng and row.get("gold_answer_en"):
                    out.setdefault((int(qid), "en"), str(row["gold_answer_en"]))
    return out


def _question_bank() -> list[dict[str, Any]]:
    """One entry per (source question, language), cached for the process lifetime."""
    cached = STATE.get("questions")
    if cached is not None:
        return cached  # type: ignore[return-value]

    p = get_pipeline()
    gold = _load_gold()
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    selected: set[tuple[int, str]] = set()

    for doc in p.index.docs.values():
        qid = doc.meta.get("query_id")
        text = str(doc.meta.get("query") or "").strip()
        if qid is None or len(text) < 8:
            continue
        key = (int(qid), doc.lang)
        if doc.meta.get("is_selected"):
            selected.add(key)
        rows.setdefault(
            key,
            {
                "query": text,
                "lang": doc.lang,
                "lang_name": LANG_NAMES.get(doc.lang, doc.lang),
                "query_id": int(qid),
                "query_type": doc.meta.get("query_type"),
                "passages": 0,
            },
        )
        rows[key]["passages"] += 1

    out: list[dict[str, Any]] = []
    for key, row in rows.items():
        has_gold = key in gold
        row["has_gold"] = has_gold
        row["answerable"] = has_gold or key in selected
        out.append(row)

    # Verified first, then the questions with the most evidence behind them.
    out.sort(key=lambda r: (not r["has_gold"], not r["answerable"], -r["passages"], r["query_id"]))
    STATE["questions"] = out
    return out


@app.get(f"{API}/samples")
async def samples(limit: int = Query(default=12, ge=1, le=60)) -> dict[str, Any]:
    """One verified example question per indexed language, for the UI's suggestion chips.

    Distinct source questions across languages where the corpus allows it, so the row reads
    as ten languages of coverage rather than one sentence translated ten times.
    """
    bank = _question_bank()
    langs = get_pipeline().index.languages()

    picked: dict[str, dict[str, Any]] = {}
    used_ids: set[int] = set()
    for want_fresh in (True, False):
        for row in bank:
            if not row["answerable"] or row["lang"] in picked:
                continue
            if want_fresh and row["query_id"] in used_ids:
                continue
            picked[row["lang"]] = row
            used_ids.add(row["query_id"])
        if len(picked) >= len(langs):
            break

    order = {code: i for i, code in enumerate(langs)}
    out = sorted(picked.values(), key=lambda r: order.get(r["lang"], 99))
    return {"samples": out[:limit], "languages": langs}


@app.get(f"{API}/questions")
async def questions(
    lang: str | None = Query(default=None, description="ISO-639-1 filter, e.g. kn"),
    q: str | None = Query(default=None, description="substring match on the question"),
    answerable_only: bool = Query(default=False),
    limit: int = Query(default=60, ge=1, le=400),
) -> dict[str, Any]:
    """Browse every question this corpus was built from.

    Backs the "explore more questions" panel. ``answerable`` is reported per row rather than
    filtered out by default, because a visitor is better served by seeing that some questions
    have no answer in this slice than by wondering why the list is short.
    """
    bank = _question_bank()
    code = detect_lang("", lang) if lang else None
    needle = (q or "").strip().casefold()

    out = []
    for row in bank:
        if code and row["lang"] != code:
            continue
        if answerable_only and not row["answerable"]:
            continue
        if needle and needle not in row["query"].casefold():
            continue
        out.append(row)

    counts: dict[str, int] = {}
    for row in bank:
        if row["answerable"]:
            counts[row["lang"]] = counts.get(row["lang"], 0) + 1

    return {
        "questions": out[:limit],
        "total": len(out),
        "returned": min(len(out), limit),
        "answerable_by_lang": counts,
        "languages": [
            {"code": c, "name": LANG_NAMES.get(c, c), "answerable": counts.get(c, 0)}
            for c in get_pipeline().index.languages()
        ],
    }


# ==================================================================== turn routes
@app.post(f"{API}/ask", response_model=TurnResult)
async def ask(
    req: AskRequest,
    authorization: str | None = Header(default=None),
    x_setu_token: str | None = Header(default=None),
) -> TurnResult:
    # The reflex path is local and free; the deliberate path spends LLM quota, so it
    # needs the API token when one is configured.
    if req.mode is not AnswerMode.REFLEX:
        require_token(authorization, x_setu_token)
    return await get_pipeline().ask(req)


@app.post(f"{API}/prefetch", dependencies=[Depends(rate_limit("prefetch", 240))])
async def prefetch(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Speculative retrieval from a partial transcript, while the user is still talking."""
    text = str(payload.get("text") or "")[:2000]
    lang = payload.get("lang")
    return await get_pipeline().prefetch(text, lang_hint=str(lang)[:8] if lang else None)


async def _read_capped(audio: UploadFile) -> bytes:
    limit = int(settings.max_audio_mb * 1024 * 1024)
    blob = await audio.read(limit + 1)
    if len(blob) > limit:
        raise HTTPException(status_code=413, detail=f"audio larger than {settings.max_audio_mb:.0f} MB")
    return blob


@app.post(f"{API}/voice", dependencies=[Depends(require_token), Depends(rate_limit("speech", 30))])
async def voice(
    audio: UploadFile = File(...),
    lang: str | None = Form(default=None),
    mode: str = Form(default="reflex"),
    speak: bool = Form(default=False),
    top_k: int | None = Form(default=None),
) -> JSONResponse:
    """Full voice turn. Returns the turn payload with base64 audio when ``speak`` is set."""
    p = get_pipeline()
    if not p.speech.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "speech-to-text is not configured",
                "hint": "set ELEVENLABS_API_KEY, or use the browser transcription fallback in the UI",
            },
        )
    try:
        answer_mode = AnswerMode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"mode must be one of {[m.value for m in AnswerMode]}") from exc
    blob = await _read_capped(audio)
    if not blob:
        raise HTTPException(status_code=400, detail="empty audio upload")
    try:
        result, tts = await p.voice(
            blob,
            filename=audio.filename or "turn.webm",
            content_type=audio.content_type or "audio/webm",
            lang=lang,
            mode=answer_mode,
            speak=bool(speak),
            top_k=top_k,
        )
    except SpeechError as exc:
        raise HTTPException(status_code=502, detail=f"speech-to-text failed: {exc}") from exc

    body: dict[str, Any] = {"turn": result.model_dump(mode="json")}
    if tts:
        import base64

        body["audio_base64"] = base64.b64encode(tts).decode("ascii")
        body["audio_mime"] = "audio/mpeg"
    return JSONResponse(body)


@app.post(f"{API}/stt", dependencies=[Depends(require_token), Depends(rate_limit("speech", 30))])
async def stt(
    audio: UploadFile = File(...), lang: str | None = Form(default=None)
) -> dict[str, Any]:
    """Transcribe only - used by the UI to show the transcript before answering."""
    p = get_pipeline()
    if not p.speech.enabled:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured")
    blob = await _read_capped(audio)
    try:
        tr = await p.speech.transcribe(
            blob, filename=audio.filename or "turn.webm",
            content_type=audio.content_type or "audio/webm", lang_hint=lang,
        )
    except SpeechError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return tr.model_dump(mode="json")


@app.post(f"{API}/stt/realtime-token", dependencies=[Depends(require_token), Depends(rate_limit("speech", 30))])
async def realtime_token() -> dict[str, Any]:
    """Mint a single-use Scribe realtime token. The API key never reaches the browser."""
    p = get_pipeline()
    if not p.speech.enabled:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured")
    try:
        return await p.speech.realtime_token()
    except SpeechError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(f"{API}/tts", dependencies=[Depends(require_token), Depends(rate_limit("speech", 40))])
async def tts(payload: dict[str, Any] = Body(...)) -> Response:
    p = get_pipeline()
    if not p.speech.enabled:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured")
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="no text supplied")
    if len(text) > settings.max_tts_chars:
        # A paid per-character API behind an open endpoint: cap what one call can spend.
        raise HTTPException(status_code=413, detail=f"text longer than {settings.max_tts_chars} characters")
    # Language decides voice, model and pronunciation pinning. Detect it from the script
    # when the caller does not say, so a Kannada string is never read by an English voice.
    lang = str(payload.get("lang") or "")[:8] or detect_lang(text)
    voice_id = payload.get("voice_id")
    allowed_voices = set(VOICE_BY_LANG.values()) | {settings.elevenlabs_voice_id}
    if voice_id is not None and voice_id not in allowed_voices:
        raise HTTPException(status_code=400, detail="voice_id is not one of the configured voices")
    try:
        audio = await p.speech.synthesize(text, lang=lang, voice_id=voice_id)
    except SpeechError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/mpeg")


# ======================================================================= websocket
@app.websocket(f"{API}/stream")
async def stream(ws: WebSocket) -> None:
    """Streaming turn channel.

    The client sends ``{"type": "partial", "text": ...}`` as speech recognition produces
    interim results; the server speculatively retrieves against each one. On
    ``{"type": "final", ...}`` it runs the full turn, which frequently finds the retrieval
    already done and reports ``speculative_hit``.
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "ping":
                await ws.send_json({"type": "pong"})
                continue
            if kind == "partial":
                try:
                    info = await get_pipeline().prefetch(
                        str(msg.get("text") or ""), lang_hint=msg.get("lang")
                    )
                    await ws.send_json({"type": "prefetch", **info})
                except HTTPException as exc:
                    await ws.send_json({"type": "error", "error": str(exc.detail)})
                continue
            if kind == "final":
                try:
                    req = AskRequest(
                        text=str(msg.get("text") or ""),
                        lang=msg.get("lang"),
                        mode=AnswerMode(msg.get("mode") or "reflex"),
                        top_k=msg.get("top_k"),
                    )
                    result = await get_pipeline().ask(req)
                    await ws.send_json({"type": "turn", "turn": result.model_dump(mode="json")})
                except HTTPException as exc:
                    await ws.send_json({"type": "error", "error": str(exc.detail)})
                except ValueError:
                    await ws.send_json({"type": "error", "error": "invalid request"})
                except Exception:  # noqa: BLE001 - never echo internals to the client
                    await ws.send_json({"type": "error", "error": "internal error"})
                continue
            await ws.send_json({"type": "error", "error": f"unknown message type '{kind}'"})
    except Exception:  # noqa: BLE001 - normal client disconnect
        return


# ==================================================================== static site
def mount_frontend() -> None:
    dist = settings.web_dist
    if not dist.exists():
        return
    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> Response:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="unknown API route")
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


mount_frontend()


if not settings.web_dist.exists():

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "app": "SETU",
            "tagline": settings.app_tagline,
            "version": __version__,
            "note": "frontend not built - run `npm install && npm run build` in web/",
            "api_docs": "/api/docs",
            "health": f"{API}/health",
        }


__all__ = ["app", "get_pipeline"]
