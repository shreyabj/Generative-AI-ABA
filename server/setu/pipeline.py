"""The SETU harness - orchestration around the model, not a prompt around it.

Every turn is a walk through an explicit state machine:

    NORMALISE -> GUARD_ADMISSION -> ROUTE -> EMBED -> GUARD_SAFETY -> RETRIEVE
              -> GUARD_TOPICALITY -> ANSWER -> GUARD_OUTPUT -> TRUST -> FINALISE
                       |                            |               |
                       +--> REFUSE                  +--> ABSTAIN <--+

Guard placement is not arbitrary. The vector-free gates (sanity, injection) run before
anything expensive, so a junk or injected turn costs ~0.2 ms. The semantic safety gate
runs immediately after ``embed_query``, because its multilingual centroid classifier
needs the same query vector retrieval is about to use - and still lands before the
index is touched.

Four properties make this a harness rather than a function call:

**Tool calls.** Every capability is registered in :class:`~setu.harness.registry.ToolRegistry`
with a declared Pydantic input and output model. Calling one validates both directions,
opens a span, and applies its resilience policy. ``GET /api/v1/tools`` returns the JSON-schema
manifest, and the deliberate path uses the same registry for real function calling.

**Structured I/O.** Nothing crosses a stage boundary as a loose dict. A stage that returns
the wrong shape is a validation failure the harness can retry, not a ``KeyError`` three
stages later. Vectors and audio travel in an in-process context object rather than through
the schema, so the control plane is fully typed without serialising a 384-float array at
every hop.

**Error recovery.** Tools carry timeouts, jittered retries and circuit breakers, and declare
fallbacks. The deliberate path falls back to the reflex path; ElevenLabs STT falls back to
browser transcription; Qdrant falls back to the in-process exact store. A turn degrades; it
does not crash. Whatever degraded is named in ``warnings``.

**Budget governance.** The turn holds a wall-clock deadline. Optional stages ask whether they
can still be afforded and are skipped and recorded if not. This is what makes the SLO a
property of the system rather than a hope about the average case.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from .answer.compose import NO_EVIDENCE, localized
from .answer.extractive import ExtractiveAnswerer
from .answer.llm import DeliberateAnswerer
from .config import settings
from .embed import Encoder, get_encoder
from .guardrails.base import GateContext
from .guardrails.stack import GuardrailStack
from .harness.contracts import (
    AnswerDraft,
    AnswerMode,
    AskRequest,
    RetrievalResult,
    RoutePlan,
    TranscriptResult,
    TurnResult,
    TurnStatus,
)
from .harness.policy import CircuitBreaker, RetryPolicy
from .harness.registry import ToolRegistry
from .harness.tracing import LatencyBudget, Trace, now_ms
from .index.corpus import CorpusIndex
from .retrieval.engine import RetrievalEngine
from .retrieval.router import QueryRouter
from .speech.elevenlabs import ElevenLabsSpeech, SpeechError
from .telemetry import TurnRecord, metrics
from .text import detect_lang, normalize


# ============================================================== tool I/O models
class NormalizeIn(BaseModel):
    text: str
    lang_hint: str | None = None


class NormalizeOut(BaseModel):
    text: str
    lang: str
    changed: bool = False


class RouteIn(BaseModel):
    text: str
    lang_hint: str | None = None


class EmbedIn(BaseModel):
    text: str


class EmbedOut(BaseModel):
    dim: int
    cache_hit: bool = False


class RetrieveIn(BaseModel):
    text: str
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerIn(BaseModel):
    text: str
    max_sentences: int | None = None


class GuardIn(BaseModel):
    stage: Literal["admission", "safety", "retrieval", "output"]


class GuardOut(BaseModel):
    blocked: bool = False
    blocked_by: str | None = None
    message: str | None = None
    verdicts: int = 0


class TranscribeIn(BaseModel):
    filename: str = "turn.webm"
    content_type: str = "audio/webm"
    size_bytes: int = 0
    lang_hint: str | None = None


class SpeakIn(BaseModel):
    text: str
    lang: str | None = None
    voice_id: str | None = None


class SpeakOut(BaseModel):
    bytes: int
    content_type: str = "audio/mpeg"


# =================================================================== turn state
@dataclass
class TurnContext:
    """In-process data plane for one turn (vectors and audio never hit a schema)."""

    query: str
    lang_hint: str | None = None
    lang: str = "en"
    qvec: np.ndarray | None = None
    plan: RoutePlan | None = None
    retrieval: RetrievalResult | None = None
    draft: AnswerDraft | None = None
    audio: bytes | None = None
    warnings: list[str] = field(default_factory=list)
    speculative_hit: bool = False


@dataclass
class Prefetch:
    """A speculative retrieval computed from a partial transcript."""

    key: str
    qvec: np.ndarray
    plan: RoutePlan
    retrieval: RetrievalResult
    at: float


class SetuPipeline:
    """Owns the corpus, the tools and the state machine."""

    def __init__(self, index: CorpusIndex, encoder: Encoder | None = None) -> None:
        self.index = index
        self.encoder = encoder or get_encoder()
        self.router = QueryRouter()
        self.engine = RetrievalEngine(index)
        self.reflex = ExtractiveAnswerer(index)
        self.deliberate = DeliberateAnswerer()
        self.guards = GuardrailStack(index)
        self.speech = ElevenLabsSpeech()
        self.registry = ToolRegistry()
        self.traces: dict[str, TurnResult] = {}
        self._trace_order: list[str] = []
        self._prefetch: dict[str, Prefetch] = {}
        self._register_tools()

    # ==================================================================== warmup
    def warm(self) -> dict[str, Any]:
        t0 = now_ms()
        embed_ms = self.encoder.warm()
        self.guards.warm()
        # One throwaway turn so every numpy/scipy code path is JIT-warm and every
        # allocation pool is sized before the first real user query.
        probe = "what is the capital of india"
        qvec = self.encoder.encode_query(probe)
        plan = self.router.route(probe)
        budget = LatencyBudget(10_000)
        trace = Trace("warmup")
        res = self.engine.search(probe, qvec, plan, budget=budget, trace=trace, top_k=3)
        self.reflex.answer(probe, qvec, res.hits, plan)
        return {
            "warm_ms": round(now_ms() - t0, 2),
            "embed_warm_ms": round(embed_ms, 2),
            "probe_hits": len(res.hits),
            "safety_centroids": self.guards.safety.ready,
        }

    # ============================================================ tool definitions
    def _register_tools(self) -> None:
        r = self.registry

        @r.register(
            "normalize_query",
            description="NFKC-normalise the query, collapse whitespace, detect language.",
            input_model=NormalizeIn, output_model=NormalizeOut, est_ms=0.3, tags=("text",),
        )
        def _normalize(p: NormalizeIn, context: dict[str, Any]) -> NormalizeOut:
            ctx: TurnContext = context["turn"]
            cleaned = normalize(p.text, casefold=False)
            ctx.query = cleaned
            ctx.lang = detect_lang(cleaned, p.lang_hint)
            return NormalizeOut(text=cleaned, lang=ctx.lang, changed=cleaned != p.text)

        @r.register(
            "route_query",
            description="Classify query shape and choose which chunk-lattice layers to search.",
            input_model=RouteIn, output_model=RoutePlan, est_ms=0.4, tags=("retrieval",),
        )
        def _route(p: RouteIn, context: dict[str, Any]) -> RoutePlan:
            ctx: TurnContext = context["turn"]
            plan = self.router.route(
                p.text, lang_hint=p.lang_hint or ctx.lang,
                available_layers=set(self.index.layer_masks),
            )
            ctx.plan = plan
            return plan

        @r.register(
            "embed_query",
            description="Encode the query into the shared multilingual vector space (LRU cached).",
            input_model=EmbedIn, output_model=EmbedOut, est_ms=18.0, tags=("model",),
        )
        def _embed(p: EmbedIn, context: dict[str, Any]) -> EmbedOut:
            ctx: TurnContext = context["turn"]
            before = self.encoder.stats()["cache"]["hits"]  # type: ignore[index]
            ctx.qvec = self.encoder.encode_query(p.text)
            after = self.encoder.stats()["cache"]["hits"]  # type: ignore[index]
            return EmbedOut(dim=int(ctx.qvec.shape[0]), cache_hit=after > before)

        @r.register(
            "retrieve",
            description="Anytime hybrid retrieval: per-layer dense + BM25, RRF fusion, parent expansion, late-interaction rerank.",
            input_model=RetrieveIn, output_model=RetrievalResult, est_ms=45.0, tags=("retrieval",),
        )
        def _retrieve(p: RetrieveIn, context: dict[str, Any]) -> RetrievalResult:
            ctx: TurnContext = context["turn"]
            assert ctx.qvec is not None and ctx.plan is not None
            res = self.engine.search(
                p.text, ctx.qvec, ctx.plan,
                budget=context["budget"], trace=context["trace"], top_k=p.top_k,
            )
            ctx.retrieval = res
            return res

        @r.register(
            "answer_reflex",
            description="Deterministic extractive answering: span selection over retrieved evidence. Cannot hallucinate tokens.",
            input_model=AnswerIn, output_model=AnswerDraft, est_ms=8.0, tags=("answer",),
        )
        def _reflex(p: AnswerIn, context: dict[str, Any]) -> AnswerDraft:
            ctx: TurnContext = context["turn"]
            assert ctx.qvec is not None and ctx.plan is not None
            draft = self.reflex.answer(
                p.text, ctx.qvec, ctx.retrieval.hits if ctx.retrieval else [], ctx.plan,
                max_sentences=p.max_sentences,
            )
            ctx.draft = draft
            return draft

        @r.register(
            "answer_deliberate",
            description="Grounded generation under a strict JSON contract with mandatory citations. Outside the 200ms SLO.",
            input_model=AnswerIn, output_model=AnswerDraft,
            timeout_s=settings.llm_timeout_s,
            retry=RetryPolicy(attempts=2, base_delay_ms=90.0),
            breaker=CircuitBreaker(name="llm", fail_threshold=3, cooldown_s=30.0),
            optional=True, est_ms=900.0, tags=("answer", "network"),
        )
        async def _deliberate(p: AnswerIn, context: dict[str, Any]) -> AnswerDraft:
            ctx: TurnContext = context["turn"]
            assert ctx.plan is not None
            draft = await self.deliberate.answer(
                p.text, ctx.retrieval.hits if ctx.retrieval else [], ctx.plan
            )
            ctx.draft = draft
            return draft

        @r.register(
            "guard",
            description="Run one guardrail stage (input | retrieval | output) and return its decision.",
            input_model=GuardIn, output_model=GuardOut, est_ms=3.0, tags=("safety",),
        )
        def _guard(p: GuardIn, context: dict[str, Any]) -> GuardOut:
            ctx: TurnContext = context["turn"]
            report = context["report"]
            gctx = GateContext(
                query=ctx.query, lang=ctx.lang, qvec=ctx.qvec, plan=ctx.plan,
                retrieval=ctx.retrieval, answer=ctx.draft,
            )
            context["gate_ctx"] = gctx
            runner = {
                "admission": self.guards.run_admission,
                "safety": self.guards.run_safety,
                "retrieval": self.guards.run_retrieval,
                "output": self.guards.run_output,
            }[p.stage]
            blocking = runner(gctx, report)
            return GuardOut(
                blocked=blocking is not None,
                blocked_by=blocking.gate if blocking else None,
                message=report.safe_message,
                verdicts=len(report.verdicts),
            )

        @r.register(
            "transcribe",
            description="ElevenLabs Scribe speech-to-text. Returns transcript plus detected language.",
            input_model=TranscribeIn, output_model=TranscriptResult,
            timeout_s=settings.stt_timeout_s,
            retry=RetryPolicy(attempts=2, base_delay_ms=140.0),
            breaker=CircuitBreaker(name="elevenlabs-stt", fail_threshold=3, cooldown_s=25.0),
            est_ms=700.0, tags=("speech", "network"),
        )
        async def _transcribe(p: TranscribeIn, context: dict[str, Any]) -> TranscriptResult:
            audio: bytes = context["audio"]
            return await self.speech.transcribe(
                audio, filename=p.filename, content_type=p.content_type, lang_hint=p.lang_hint
            )

        @r.register(
            "speak",
            description="ElevenLabs Flash text-to-speech for the verified answer.",
            input_model=SpeakIn, output_model=SpeakOut,
            timeout_s=settings.stt_timeout_s,
            retry=RetryPolicy(attempts=2, base_delay_ms=120.0),
            breaker=CircuitBreaker(name="elevenlabs-tts", fail_threshold=3, cooldown_s=25.0),
            est_ms=600.0, tags=("speech", "network"),
        )
        async def _speak(p: SpeakIn, context: dict[str, Any]) -> SpeakOut:
            audio = await self.speech.synthesize(p.text, lang=p.lang, voice_id=p.voice_id)
            context["tts"] = audio
            return SpeakOut(bytes=len(audio))

    # ============================================================ speculative path
    async def prefetch(self, partial: str, *, lang_hint: str | None = None) -> dict[str, Any]:
        """Retrieve against a partial transcript while the user is still speaking.

        Voice adds hundreds of milliseconds of dead time that the retrieval side spends
        idle. Streaming partial transcripts lets us spend it: embed and retrieve the
        partial, and if the final transcript matches, the pipeline is already done.
        A miss costs the user nothing - it happened during speech - and the encoder LRU
        keeps the vector regardless.

        Reuse requires an exact normalised match. Approximate reuse would be faster and
        occasionally wrong, which is not a trade worth making.
        """
        cleaned = normalize(partial, casefold=False)
        if len(cleaned) < 8:
            return {"prefetched": False, "reason": "partial too short"}
        key = cleaned.casefold()
        if key in self._prefetch:
            return {"prefetched": True, "cached": True, "key_len": len(key)}

        t0 = now_ms()
        qvec = self.encoder.encode_query(cleaned)
        plan = self.router.route(cleaned, lang_hint=lang_hint, available_layers=set(self.index.layer_masks))
        budget = LatencyBudget(settings.budget.total_ms)
        trace = Trace(f"prefetch-{int(t0)}")
        res = self.engine.search(cleaned, qvec, plan, budget=budget, trace=trace, top_k=settings.final_k)

        self._prefetch[key] = Prefetch(key=key, qvec=qvec, plan=plan, retrieval=res, at=time.monotonic())
        # Bounded: a voice turn produces a handful of partials, not thousands.
        if len(self._prefetch) > 24:
            oldest = min(self._prefetch.values(), key=lambda p: p.at)
            self._prefetch.pop(oldest.key, None)
        return {
            "prefetched": True, "cached": False, "hits": len(res.hits),
            "ms": round(now_ms() - t0, 2), "top_score": res.top_score,
        }

    def _take_prefetch(self, query: str, *, ttl_s: float = 60.0) -> Prefetch | None:
        hit = self._prefetch.pop(normalize(query, casefold=False).casefold(), None)
        # A prefetch is only valid for the utterance in progress. Without a TTL, any
        # client's partial could satisfy another client's question minutes later.
        if hit is not None and time.monotonic() - hit.at > ttl_s:
            return None
        return hit

    # ==================================================================== the turn
    async def ask(
        self,
        req: AskRequest,
        *,
        transcript: TranscriptResult | None = None,
        trace: Trace | None = None,
    ) -> TurnResult:
        """Run one turn. Answer text only - synthesis is :meth:`speak_answer`'s job.

        Speech output deliberately lives outside this method. It used to be folded in via a
        ``speak`` flag, which quietly did nothing: the ``speak`` tool wrote the audio into
        ``ask``'s own private context, so ``voice()`` - the only caller that can actually
        return audio to a client - never saw it and every voice turn came back silent.
        Keeping synthesis at the boundary that owns the response makes that impossible.
        """
        from .harness.contracts import GuardReport

        trace = trace or Trace()
        wall0 = now_ms()
        budget = LatencyBudget(req.budget_ms or settings.budget.total_ms)
        report = GuardReport()
        ctx = TurnContext(query=req.text, lang_hint=req.lang or (transcript.lang if transcript else None))
        top_k = req.top_k or settings.final_k

        context: dict[str, Any] = {
            "turn": ctx, "budget": budget, "trace": trace, "report": report, "audio": None,
        }

        def finalize(status: TurnStatus, answer: str = "") -> TurnResult:
            pipeline_ms = trace.elapsed_ms
            stt_ms = transcript.duration_ms if transcript else 0.0
            gctx = context.get("gate_ctx") or GateContext(
                query=ctx.query, lang=ctx.lang, qvec=ctx.qvec, plan=ctx.plan,
                retrieval=ctx.retrieval, answer=ctx.draft,
            )
            self.guards.score(gctx, report)
            answered = status in (TurnStatus.ANSWERED, TurnStatus.DEGRADED)
            result = TurnResult(
                turn_id=trace.turn_id,
                status=status,
                query=req.text,
                normalized_query=ctx.query,
                answer=answer,
                answer_lang=(ctx.draft.lang if ctx.draft and answered else ctx.lang),
                path=(ctx.draft.path if ctx.draft else "none"),
                model=(ctx.draft.model if ctx.draft else ""),
                citations=(ctx.draft.citations if ctx.draft and answered else []),
                claims=(ctx.draft.claims if ctx.draft else []),
                evidence=(ctx.retrieval.hits if ctx.retrieval else []),
                route=ctx.plan,
                retrieval=ctx.retrieval,
                guards=report,
                trust_score=(report.trust_score if answered else 0.0),
                transcript=transcript,
                spans=trace.models(),
                budget=budget.report(),
                pipeline_ms=round(pipeline_ms, 3),
                stt_ms=round(stt_ms, 3),
                e2e_ms=round(stt_ms + (now_ms() - wall0), 3),
                slo_met=pipeline_ms <= budget.total_ms,
                warnings=ctx.warnings,
                speculative_hit=ctx.speculative_hit,
            )
            metrics.record(
                TurnRecord(
                    turn_id=result.turn_id, pipeline_ms=result.pipeline_ms, stt_ms=result.stt_ms,
                    e2e_ms=result.e2e_ms, status=status.value, path=result.path, lang=result.answer_lang,
                    shape=(ctx.plan.shape.value if ctx.plan else "unknown"), trust=result.trust_score,
                    slo_met=result.slo_met, degraded=budget.degraded, stages=trace.by_name(),
                )
            )
            self._remember(result)
            return result

        # ------------------------------------------------------------ NORMALISE
        out = await self.registry.call(
            "normalize_query", NormalizeIn(text=req.text, lang_hint=req.lang),
            trace=trace, context=context,
        )
        if not out.ok:
            ctx.warnings.append(f"normalise failed: {out.error}")

        # ------------------------------------------------------- GUARD_ADMISSION
        g = await self.registry.call("guard", GuardIn(stage="admission"), trace=trace, context=context)
        if g.ok and g.value and g.value.blocked:
            return finalize(TurnStatus.REFUSED, report.safe_message or "")

        # ---------------------------------------------------------------- ROUTE
        out = await self.registry.call(
            "route_query", RouteIn(text=ctx.query, lang_hint=ctx.lang_hint), trace=trace, context=context
        )
        if not out.ok or ctx.plan is None:
            ctx.plan = self.router.route(ctx.query, lang_hint=ctx.lang_hint)
            ctx.warnings.append("router fell back to defaults")

        # -------------------------------------------------------- EMBED (or reuse)
        spec = self._take_prefetch(req.text)
        if spec is not None:
            ctx.qvec, ctx.plan, ctx.retrieval = spec.qvec, spec.plan, spec.retrieval
            ctx.speculative_hit = True
            trace.mark("embed_query", status="speculative-hit")
        else:
            out = await self.registry.call("embed_query", EmbedIn(text=ctx.query), trace=trace, context=context)
            if not out.ok or ctx.qvec is None:
                return finalize(TurnStatus.ERROR, f"embedding unavailable: {out.error}")

        # ---------------------------------------------------------- GUARD_SAFETY
        # Placed here because the semantic classifier needs the query vector, and still
        # ahead of retrieval so an unsafe request never reaches the index.
        g = await self.registry.call("guard", GuardIn(stage="safety"), trace=trace, context=context)
        if g.ok and g.value and g.value.blocked:
            return finalize(TurnStatus.REFUSED, report.safe_message or "")

        # -------------------------------------------------------------- RETRIEVE
        if ctx.retrieval is not None and spec is not None:
            trace.mark("retrieve", status="speculative-hit", hits=len(ctx.retrieval.hits))
        else:
            out = await self.registry.call(
                "retrieve", RetrieveIn(text=ctx.query, top_k=top_k), trace=trace, context=context
            )
            if not out.ok or ctx.retrieval is None:
                return finalize(TurnStatus.ERROR, f"retrieval failed: {out.error}")

        if not ctx.retrieval.hits:
            report.safe_message = localized(NO_EVIDENCE, ctx.lang)
            return finalize(TurnStatus.ABSTAINED, report.safe_message)

        # ------------------------------------------------------ GUARD_TOPICALITY
        g = await self.registry.call("guard", GuardIn(stage="retrieval"), trace=trace, context=context)
        if g.ok and g.value and g.value.blocked:
            return finalize(TurnStatus.ABSTAINED, report.safe_message or "")

        # --------------------------------------------------------------- ANSWER
        await self._answer(req, ctx, context, trace, budget)
        if ctx.draft is None or not ctx.draft.text.strip():
            report.safe_message = localized(NO_EVIDENCE, ctx.lang)
            return finalize(TurnStatus.ABSTAINED, report.safe_message)

        # ---------------------------------------------------------- GUARD_OUTPUT
        g = await self.registry.call("guard", GuardIn(stage="output"), trace=trace, context=context)
        if g.ok and g.value and g.value.blocked:
            return finalize(TurnStatus.ABSTAINED, report.safe_message or "")
        if not g.ok:
            ctx.warnings.append(f"output guard error: {g.error}")

        # ---------------------------------------------------------- SCORE / TRUST
        gctx = context.get("gate_ctx")
        if gctx is not None:
            self.guards.score(gctx, report)
            if self.guards.abstain_below_trust(report, ctx.lang):
                return finalize(TurnStatus.ABSTAINED, report.safe_message or "")

        answer_text = ctx.draft.text
        # Finalise with the real status so telemetry records "degraded", not "answered".
        status = TurnStatus.DEGRADED if budget.degraded else TurnStatus.ANSWERED
        return finalize(status, answer_text)

    async def _answer(
        self, req: AskRequest, ctx: TurnContext, context: dict[str, Any], trace: Trace, budget: LatencyBudget
    ) -> None:
        """Pick an answer path, with the reflex path as a declared fallback."""
        want_deliberate = req.mode is AnswerMode.DELIBERATE or (
            req.mode is AnswerMode.AUTO and settings.llm_enabled and _needs_synthesis(ctx)
        )
        payload = AnswerIn(text=ctx.query)

        if want_deliberate and settings.llm_enabled:
            async def _fallback() -> AnswerDraft:
                ctx.warnings.append("deliberate path unavailable - served by reflex path")
                draft = self.reflex.answer(
                    ctx.query, ctx.qvec, ctx.retrieval.hits if ctx.retrieval else [], ctx.plan  # type: ignore[arg-type]
                )
                ctx.draft = draft
                return draft

            out = await self.registry.call(
                "answer_deliberate", payload, trace=trace, context=context, fallbacks=(_fallback,)
            )
            if out.ok and ctx.draft is not None and ctx.draft.text.strip():
                return
            if out.ok and ctx.draft is not None and ctx.draft.path == "deliberate":
                # The model abstained. That is a decision, not a failure: falling through
                # to the extractive path here used to answer anyway, silently overriding
                # an explicit "the evidence does not contain this".
                ctx.warnings.append("deliberate path abstained - evidence judged insufficient")
                return
            if not out.ok:
                ctx.warnings.append(f"deliberate path failed: {out.error}")
        elif want_deliberate:
            ctx.warnings.append("deliberate mode requested but no LLM key configured")

        await self.registry.call("answer_reflex", payload, trace=trace, context=context)

    # ================================================================ voice turn
    async def voice(
        self,
        audio: bytes,
        *,
        filename: str = "turn.webm",
        content_type: str = "audio/webm",
        lang: str | None = None,
        mode: AnswerMode = AnswerMode.REFLEX,
        speak: bool = False,
        top_k: int | None = None,
    ) -> tuple[TurnResult, bytes | None]:
        """Full voice turn: Scribe -> pipeline -> (optional) spoken answer."""
        trace = Trace()
        context: dict[str, Any] = {"audio": audio}
        with trace.span("stt.elevenlabs", bytes=len(audio), content_type=content_type):
            out = await self.registry.call(
                "transcribe",
                TranscribeIn(
                    filename=filename, content_type=content_type,
                    size_bytes=len(audio), lang_hint=lang,
                ),
                trace=trace, context=context,
            )
        if not out.ok or out.value is None:
            raise SpeechError(out.error or "transcription failed")

        transcript: TranscriptResult = out.value

        # An explicit language choice beats speech-to-text's guess. The user picked Kannada
        # in the UI; if Scribe returns a transliterated English-looking string it will be
        # detected as English, and we would then retrieve and answer in English - which is
        # the opposite of what was asked for. Auto-detection is only used when the caller
        # said "any language".
        answer_lang = lang or transcript.lang

        # STT time is measured but must not consume the pipeline's own budget.
        inner = Trace(trace.turn_id)
        result = await self.ask(
            AskRequest(text=transcript.text, lang=answer_lang, mode=mode, top_k=top_k),
            transcript=transcript, trace=inner,
        )
        result.spans = trace.models() + result.spans

        audio_out: bytes | None = None
        if speak and result.answer and result.status is not TurnStatus.REFUSED:
            audio_out, err = await self.speak_answer(result, trace=trace)
            if audio_out is None:
                # Carry the actual reason. "could not synthesise" with no cause attached is
                # not a diagnosable warning, and this failure is invisible to the user
                # otherwise - they just get silence.
                result.warnings.append(f"spoken reply unavailable: {err or 'unknown error'}")
        return result, audio_out

    async def speak_answer(
        self, result: TurnResult, *, trace: Trace | None = None
    ) -> tuple[bytes | None, str | None]:
        """Synthesise a finished answer. Returns ``(audio, error)`` - one of them is None."""
        if not self.speech.enabled:
            return None, "ELEVENLABS_API_KEY not configured"
        spoken = strip_citations(result.answer)
        if not spoken:
            return None, "nothing to speak"
        sink: dict[str, Any] = {}
        out = await self.registry.call(
            "speak", SpeakIn(text=spoken, lang=result.answer_lang), trace=trace, context=sink
        )
        if not out.ok:
            return None, out.error
        audio = sink.get("tts")
        return (audio, None) if audio else (None, "synthesiser returned no audio")

    # ==================================================================== traces
    def _remember(self, result: TurnResult) -> None:
        self.traces[result.turn_id] = result
        self._trace_order.append(result.turn_id)
        while len(self._trace_order) > settings.trace_retain:
            self.traces.pop(self._trace_order.pop(0), None)

    def trace_of(self, turn_id: str) -> TurnResult | None:
        return self.traces.get(turn_id)

    # ===================================================================== health
    def health(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.stats(),
            "index": self.index.stats(),
            "tools": self.registry.names(),
            "circuit_breakers": self.registry.breakers(),
            "guardrails": self.guards.gate_names,
            "speech": {
                "provider": self.speech.provider,
                "enabled": self.speech.enabled,
                "stt_model": settings.elevenlabs_stt_model,
                "tts_model": settings.elevenlabs_tts_model,
            },
            "deliberate": {"enabled": settings.llm_enabled, "model": settings.llm_model},
            "prefetch_entries": len(self._prefetch),
            "slo_ms": settings.budget.total_ms,
        }


_CITE_RE = re.compile(r"\s*\[\d+\]")


def strip_citations(text: str) -> str:
    """Remove ``[1]`` markers before speaking.

    Citation markers are essential on screen and actively harmful in audio - a synthesiser
    reads them out as "one", "two", turning a clean answer into "...publicly held
    corporation one. Corporation two." They belong to the visual layer only.
    """
    return _CITE_RE.sub("", text).strip()


def _needs_synthesis(ctx: TurnContext) -> bool:
    """AUTO mode heuristic: escalate only when extraction is likely to read badly.

    Descriptive and enumerative questions whose evidence is spread across several
    documents are the cases where stitched spans read like a list of quotes rather than
    an answer. Everything else is served better - and 400x faster - by extraction.
    """
    if ctx.plan is None or ctx.retrieval is None:
        return False
    from .harness.contracts import QueryShape

    broad = ctx.plan.shape in {QueryShape.DESCRIPTION, QueryShape.LIST, QueryShape.DEFINITION}
    spread = len({h.doc_id for h in ctx.retrieval.hits[:4]}) >= 3
    return broad and spread


__all__ = ["Prefetch", "SetuPipeline", "TurnContext"]
