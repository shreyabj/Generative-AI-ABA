"""The Deliberate path - grounded generation with a structured contract.

This path exists for questions the extractive answerer genuinely cannot serve well:
ones whose answer must fuse two passages, or be rephrased to match how the question
was asked. It is **explicitly outside the 200 ms SLO** and is reported separately in
the latency table, because time-to-first-token on a hosted model is not something a
local pipeline can put a bound on. Pretending otherwise would make the headline
number meaningless.

What makes this a harness call rather than a prompt
--------------------------------------------------
* The model must return JSON matching :class:`LLMAnswer`. A malformed body is a
  validation failure the harness retries, not a string we regex and hope about.
* Every sentence must carry citation markers pointing at supplied evidence ids.
  Sentences whose markers do not resolve are dropped before the answer is returned.
* ``abstain`` is a first-class field. The model is told, explicitly, that refusing is
  a correct and expected outcome - so "I don't know" is a structured decision the
  guardrail layer can act on rather than prose we have to detect.
* Temperature 0, capped tokens, hard timeout, retry with jitter, and a circuit
  breaker registered on the tool. If the endpoint degrades, the turn falls back to
  the reflex path and the response says ``fell_back: true``.

Any OpenAI-compatible endpoint works. Groq is the default because its TTFT is the
lowest of the hosted options, which keeps this path usable rather than theoretical.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..config import settings
from ..harness.contracts import AnswerDraft, ChunkHit, RoutePlan
from ..text import LANG_NAMES, truncate
from .compose import build_citations

SYSTEM_PROMPT = """You are SETU, a retrieval-grounded answering engine.

Absolute rules:
1. Answer ONLY from the numbered EVIDENCE passages. You have no other knowledge.
2. Every sentence you write must end with citation markers, e.g. "... 89 calories [2]."
3. If the evidence does not contain the answer, set "abstain": true and leave "answer" empty.
   Abstaining is a correct outcome, not a failure. Never fill a gap with plausible text.
4. Never state a number, date, name or quantity that does not appear verbatim in the evidence.
5. Reply in the language named in TARGET_LANGUAGE, regardless of the evidence language.
6. Be direct. Two or three sentences. No preamble, no "based on the evidence".

Return ONLY a JSON object, no markdown fence:
{"answer": str, "citations": [int], "abstain": bool, "reason": str}"""


class LLMAnswer(BaseModel):
    """The structured contract the deliberate path must satisfy."""

    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    abstain: bool = False
    reason: str = ""


class DeliberateError(RuntimeError):
    pass


def reasoning_params(model: str, effort: str | None = None) -> dict[str, Any]:
    """Extra request fields for reasoning models.

    gpt-oss and qwen3 on Groq emit a reasoning preamble; without ``reasoning_effort`` the
    preamble can collide with ``response_format=json_object``. Sending it keeps the
    reasoning in a separate channel and the message content a clean JSON object.
    """
    m = model.lower()
    if m.startswith("openai/gpt-oss"):
        return {"reasoning_effort": effort or settings.research_reasoning_effort}
    if m.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


def build_evidence_block(hits: list[ChunkHit], *, max_chars: int = 700) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(
            f"[{i}] (layer={h.layer.value}, lang={h.lang}, score={h.score:.3f})\n"
            f"{truncate(h.text, max_chars)}"
        )
    return "\n\n".join(lines)


class DeliberateAnswerer:
    """Calls an OpenAI-compatible chat endpoint under a strict JSON contract."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.llm_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def answer(
        self,
        query: str,
        hits: list[ChunkHit],
        plan: RoutePlan,
        *,
        timeout_s: float | None = None,
    ) -> AnswerDraft:
        if not settings.llm_enabled:
            raise DeliberateError("no LLM API key configured (set GROQ_API_KEY)")
        if not hits:
            raise DeliberateError("deliberate path requires evidence")

        target = LANG_NAMES.get(plan.lang, "English")
        user = (
            f"TARGET_LANGUAGE: {target}\n\n"
            f"EVIDENCE:\n{build_evidence_block(hits)}\n\n"
            f"QUESTION: {query}"
        )

        client = await self._get_client()
        resp = await client.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": settings.llm_max_tokens,
                "response_format": {"type": "json_object"},
                **reasoning_params(settings.llm_model),
            },
            timeout=timeout_s or settings.llm_timeout_s,
        )
        if resp.status_code >= 400:
            raise DeliberateError(f"LLM HTTP {resp.status_code}: {truncate(resp.text, 200)}")

        body: dict[str, Any] = resp.json()
        try:
            raw = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeliberateError(f"unexpected LLM envelope: {truncate(str(body), 200)}") from exc

        parsed = self._parse(raw)

        if parsed.abstain or not parsed.answer.strip():
            return AnswerDraft(
                text="", lang=plan.lang, path="deliberate", model=settings.llm_model,
                confidence=0.0,
            )

        text, markers = self._validate_citations(parsed, len(hits))
        if not markers:
            raise DeliberateError("model returned no resolvable citations")

        cite_rows = [
            (m, hits[m - 1].chunk_id, hits[m - 1].doc_id, hits[m - 1].layer,
             hits[m - 1].text, hits[m - 1].score)
            for m in markers
        ]
        return AnswerDraft(
            text=text,
            lang=plan.lang,
            citations=build_citations(cite_rows),  # type: ignore[arg-type]
            path="deliberate",
            model=settings.llm_model,
            confidence=0.72,  # provisional; the groundedness gate sets the real trust score
        )

    # ================================================================= internals
    @staticmethod
    def _parse(raw: str) -> LLMAnswer:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        try:
            return LLMAnswer.model_validate_json(raw)
        except ValidationError:
            pass
        # Tolerate a JSON object embedded in prose - once. Anything worse is a retry.
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return LLMAnswer.model_validate(json.loads(raw[start : end + 1]))
            except Exception:  # noqa: BLE001
                pass
        raise DeliberateError(f"model did not return valid JSON: {truncate(raw, 160)}")

    @staticmethod
    def _validate_citations(parsed: LLMAnswer, n_hits: int) -> tuple[str, list[int]]:
        """Keep only markers that resolve to real evidence; drop uncited sentences."""
        import re

        valid = {m for m in parsed.citations if 1 <= m <= n_hits}
        kept_sentences: list[str] = []
        used: list[int] = []
        for sent in re.split(r"(?<=[.!?\u0964\u06d4])\s+", parsed.answer.strip()):
            if not sent.strip():
                continue
            found = [int(m) for m in re.findall(r"\[(\d+)\]", sent)]
            resolvable = [m for m in found if 1 <= m <= n_hits]
            if not resolvable:
                continue  # an uncited sentence is an ungrounded sentence
            kept_sentences.append(sent.strip())
            used.extend(resolvable)
        if not kept_sentences and valid:
            # Model cited in the field but not inline - attach them explicitly.
            markers = sorted(valid)
            tag = "".join(f"[{m}]" for m in markers)
            return f"{parsed.answer.strip()} {tag}".strip(), markers
        seen: list[int] = []
        for m in used:
            if m not in seen:
                seen.append(m)
        return " ".join(kept_sentences), sorted(seen)


__all__ = ["DeliberateAnswerer", "DeliberateError", "LLMAnswer", "build_evidence_block"]
