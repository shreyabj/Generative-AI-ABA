"""JSON-mode chat client for the research pipeline.

OpenAI-compatible (Groq by default). Every call returns parsed JSON or raises
:class:`LLMError` - callers never regex prose. Retries cover rate limits and transient
5xx; a circuit breaker stops hammering a failing endpoint; every call reports its model,
latency, attempts and token usage so the pipeline explorer can show what the model cost.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx

from ..answer.llm import reasoning_params
from ..config import settings
from ..harness.policy import CircuitBreaker
from ..text import truncate


class LLMError(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(raw: str) -> dict[str, Any]:
    s = _FENCE.sub("", raw.strip())
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise LLMError(f"model returned non-JSON: {truncate(raw, 160)}") from None
        try:
            out = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned malformed JSON: {exc}") from exc
    if not isinstance(out, dict):
        raise LLMError("model returned JSON that is not an object")
    return out


class LLMClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self.breaker = CircuitBreaker(name="research-llm", fail_threshold=4, cooldown_s=30.0)

    @property
    def enabled(self) -> bool:
        return settings.llm_enabled

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.research_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 3000,
        timeout_s: float | None = None,
        attempts: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            raise LLMError("no LLM API key configured (set GROQ_API_KEY)")
        if self.breaker.state == "open":
            raise LLMError("LLM circuit breaker is open after repeated failures - retry shortly")
        model = model or settings.research_model
        client = await self._http()
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            **reasoning_params(model),
        }
        last: str = ""
        t0 = time.perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.post(
                    f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json=body, timeout=timeout_s or settings.research_timeout_s,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}: {truncate(resp.text, 160)}"
                    retry_after = float(resp.headers.get("retry-after") or 0) or 1.5 * attempt
                    await asyncio.sleep(min(retry_after, 8.0))
                    continue
                if resp.status_code >= 400:
                    # 400s are usually JSON-mode validation failures; one retry is worth it.
                    last = f"HTTP {resp.status_code}: {truncate(resp.text, 240)}"
                    if attempt < attempts and "json" in resp.text.lower():
                        continue
                    break
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                parsed = parse_json(content)
                self.breaker.record_success()
                return parsed, {
                    "model": model, "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "attempts": attempt, "usage": data.get("usage", {}),
                }
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = f"{type(exc).__name__}"
                await asyncio.sleep(0.8 * attempt)
            except (LLMError, KeyError, IndexError, ValueError) as exc:
                last = str(exc)
        self.breaker.record_failure()
        raise LLMError(f"LLM call failed after {attempts} attempt(s): {last}")


__all__ = ["LLMClient", "LLMError", "parse_json"]
