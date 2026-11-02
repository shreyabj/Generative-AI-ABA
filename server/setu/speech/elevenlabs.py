"""ElevenLabs speech services - Scribe for transcription, Flash for synthesis.

The task allows either Sarvam or ElevenLabs. We use **ElevenLabs Scribe**: it covers all
14 MS MARCO-XI languages, returns a language code with a confidence score (which we feed
straight into the router as a language hint instead of guessing from the script), and word
timestamps that the UI uses to align the waveform with the transcript.

Latency accounting
------------------
STT is a network call to a third party. It is measured and reported as ``stt_ms``, kept
*separate* from ``pipeline_ms``, and excluded from the 200 ms SLO - because a bound on
somebody else's inference endpoint is not something this pipeline can honestly promise.
``e2e_ms`` reports the total so nothing is hidden. See docs/LATENCY.md for the full
argument and the measured numbers for each.

Both clients go through the harness resilience policy: bounded timeout, two attempts with
jittered backoff, and a circuit breaker so a degraded endpoint fails fast into the
browser-side fallback instead of stalling every turn.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import settings
from ..harness.contracts import TranscriptResult
from ..text import ISO1_TO_ISO3, resolve_lang

API_ROOT = "https://api.elevenlabs.io/v1"

# ---------------------------------------------------------------- voice routing
#
# Reading a Kannada answer with an English voice and English prosody is the single most
# jarring thing a multilingual voice product can do, so synthesis is routed per language.
#
# Measured against this account (see the probe in git history): `eleven_flash_v2_5` and
# `eleven_multilingual_v2` both *return audio* for all ten corpus languages, but the
# `language_code` parameter - the one that actually pins pronunciation - is only accepted
# for a subset. Passing it for the others returns HTTP 400 `invalid_parameters`.
#
# So: pin the language where the API lets us, and use the higher-fidelity multilingual
# model everywhere else rather than the speed-optimised one, because prosody matters more
# than 100 ms when the alternative is unintelligible.
FLASH_LANGUAGE_CODES: frozenset[str] = frozenset({"en", "hi", "ta"})

# One voice per language instead of one voice for everything. These are all built-in
# voices (available on free accounts); the goal is that switching language audibly
# switches speaker, rather than one narrator reading ten scripts.
VOICE_BY_LANG: dict[str, str] = {
    "en": "EXAVITQu4vr4xnSDxMaL",  # Sarah  - mature, reassuring
    "hi": "XrExE9yKIg1WjnnlVkGX",  # Matilda - knowledgeable, professional
    "bn": "Xb7hH8MSUJpSbSDYk0k2",  # Alice  - clear educator
    "ta": "cgSgspJ2msm6clMCkdW9",  # Jessica - bright, warm
    "te": "SAz9YHcvj6GT2YYXdXww",  # River  - relaxed, informative
    "kn": "JBFqnCBsd6RMkjVDRZzb",  # George - warm storyteller
    "ml": "cjVigY5qzO86Huf0OWal",  # Eric   - smooth, trustworthy
    "mr": "FGY2WhTYpPnrIDTdsKH5",  # Laura
    "gu": "bIHbv24MWmeRgasZH58o",  # Will   - relaxed optimist
    "pa": "IKne3meq5aSn9XLyUdCD",  # Charlie - confident
    "ur": "CwhRBWXzGAHq8TQ4Fs17",  # Roger
    "or": "TX3LPaxmHKxFdv7VOQHJ",  # Liam
    "as": "N2lVS1w4EtoT3dr4eOWO",  # Callum
    "ne": "SOYHLrjzK2X1ezoPC6cr",  # Harry
    "sa": "XrExE9yKIg1WjnnlVkGX",  # Matilda
}


class SpeechError(RuntimeError):
    pass


class ElevenLabsSpeech:
    """Thin, well-behaved client for the two endpoints we need."""

    provider = "elevenlabs"

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self._client = client
        self._owns = client is None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.stt_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns:
            await self._client.aclose()
            self._client = None

    def _headers(self, **extra: str) -> dict[str, str]:
        if not self.api_key:
            raise SpeechError("ELEVENLABS_API_KEY is not set")
        return {"xi-api-key": self.api_key, **extra}

    # ==================================================================== scribe
    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "turn.webm",
        content_type: str = "audio/webm",
        lang_hint: str | None = None,
        keyterms: list[str] | None = None,
    ) -> TranscriptResult:
        if not audio:
            raise SpeechError("empty audio payload")

        data: dict[str, Any] = {"model_id": settings.elevenlabs_stt_model}
        if lang_hint:
            iso3 = ISO1_TO_ISO3.get(lang_hint.lower()[:2])
            if iso3:
                data["language_code"] = iso3
        if keyterms:
            data["keyterms"] = str(keyterms[:16])

        client = await self._http()
        t0 = time.perf_counter()
        resp = await client.post(
            f"{API_ROOT}/speech-to-text",
            headers=self._headers(),
            files={"file": (filename, audio, content_type)},
            data=data,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            raise SpeechError(f"Scribe HTTP {resp.status_code}: {resp.text[:220]}")

        body = resp.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise SpeechError("Scribe returned an empty transcript")

        # Scribe's own language code is a claim, and it loses to the script it returned.
        # "language_code: eng" alongside Kannada characters is a misdetection, and honouring
        # it would answer a Kannada question in English. Claims inside the same script are
        # kept, because they distinguish what the script cannot (Hindi vs Marathi).
        raw_lang = body.get("language_code") or ""
        lang = resolve_lang(text, raw_lang or None)
        return TranscriptResult(
            text=text,
            lang=lang,
            lang_confidence=body.get("language_probability"),
            provider=self.provider,
            model=settings.elevenlabs_stt_model,
            duration_ms=round(elapsed, 3),
            words=(body.get("words") or [])[:400],
        )

    async def realtime_token(self) -> dict[str, Any]:
        """Mint a single-use realtime token for browser-side Scribe.

        The API key must never reach the browser, so the client asks this endpoint
        for a short-lived token immediately before opening its websocket.
        """
        client = await self._http()
        resp = await client.post(
            f"{API_ROOT}/tokens/single-use",
            headers=self._headers(**{"Content-Type": "application/json"}),
            json={"purpose": "realtime_scribe"},
        )
        if resp.status_code >= 400:
            raise SpeechError(f"token HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # ======================================================================= tts
    def plan_voice(self, lang: str | None) -> tuple[str, str, str | None]:
        """Return ``(voice_id, model_id, language_code)`` for a language."""
        code = (lang or "en").lower()[:2]
        voice = VOICE_BY_LANG.get(code, settings.elevenlabs_voice_id)
        if code in FLASH_LANGUAGE_CODES:
            # Pronunciation can be pinned, so use the low-latency model.
            return voice, settings.elevenlabs_tts_model, code
        # No language pin available: prefer fidelity over speed.
        return voice, settings.elevenlabs_tts_fallback_model, None

    async def synthesize(
        self,
        text: str,
        *,
        lang: str | None = None,
        voice_id: str | None = None,
        output_format: str = "mp3_44100_128",
    ) -> bytes:
        """Speak an answer in the language it was written in.

        Voice, model and (where supported) an explicit ``language_code`` are all chosen from
        ``lang`` - see :data:`VOICE_BY_LANG` for why. A slightly slower model with correct
        prosody beats a fast one that mispronounces the whole answer.
        """
        if not text.strip():
            raise SpeechError("nothing to synthesise")

        voice, model, lang_code = self.plan_voice(lang)
        if voice_id:
            voice = voice_id

        payload: dict[str, Any] = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "style": 0.1,
                "speed": 1.0,
                "use_speaker_boost": True,
            },
        }
        if lang_code:
            payload["language_code"] = lang_code

        client = await self._http()
        resp = await client.post(
            f"{API_ROOT}/text-to-speech/{voice}",
            headers=self._headers(**{"Content-Type": "application/json", "Accept": "audio/mpeg"}),
            params={"output_format": output_format},
            json=payload,
        )
        if resp.status_code >= 400:
            raise SpeechError(f"TTS HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.content


__all__ = ["API_ROOT", "ElevenLabsSpeech", "SpeechError"]
