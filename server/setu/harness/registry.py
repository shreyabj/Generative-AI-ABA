"""Tool registry - the callable surface of the harness.

Every capability in SETU (embed, dense search, sparse search, fuse, rerank,
compose, verify, transcribe, speak) is registered here as a *tool* with a
declared input and output schema. Calling a tool goes through validation,
tracing and the resilience policy. The registry can also emit a JSON-schema
manifest, which is what the deliberate (LLM) path uses for real tool calling.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .policy import CircuitBreaker, Outcome, RetryPolicy, run_resilient
from .tracing import Trace

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


class ToolError(RuntimeError):
    pass


@dataclass
class Tool(Generic[TIn, TOut]):
    name: str
    description: str
    input_model: type[TIn]
    output_model: type[TOut]
    handler: Callable[..., Any]
    timeout_s: float | None = None
    retry: RetryPolicy | None = None
    breaker: CircuitBreaker | None = None
    optional: bool = False
    est_ms: float = 5.0
    tags: tuple[str, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}

    # ------------------------------------------------------------- registration
    def register(
        self,
        name: str,
        *,
        description: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        timeout_s: float | None = None,
        retry: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        optional: bool = False,
        est_ms: float = 5.0,
        tags: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = Tool(
                name=name,
                description=description,
                input_model=input_model,
                output_model=output_model,
                handler=fn,
                timeout_s=timeout_s,
                retry=retry,
                breaker=breaker,
                optional=optional,
                est_ms=est_ms,
                tags=tags,
            )
            return fn

        return deco

    def add(self, tool: Tool[Any, Any]) -> None:
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool '{name}'") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def manifest(self) -> list[dict[str, Any]]:
        return [t.json_schema() for t in self._tools.values()]

    def breakers(self) -> list[dict[str, Any]]:
        return [t.breaker.snapshot() for t in self._tools.values() if t.breaker]

    # ----------------------------------------------------------------- calling
    async def call(
        self,
        name: str,
        payload: BaseModel | dict[str, Any],
        *,
        trace: Trace | None = None,
        context: dict[str, Any] | None = None,
        fallbacks: tuple[Callable[[], Awaitable[Any]], ...] = (),
    ) -> Outcome[Any]:
        """Validate -> run under policy -> validate -> record span."""
        tool = self.get(name)

        try:
            data = payload if isinstance(payload, tool.input_model) else tool.input_model.model_validate(
                payload.model_dump() if isinstance(payload, BaseModel) else payload
            )
        except ValidationError as exc:
            if trace:
                trace.mark(f"tool.{name}", status="invalid-input", error=exc.errors()[:2])
            return Outcome(ok=False, error=f"input validation failed for '{name}': {exc.error_count()} issue(s)")

        async def invoke() -> Any:
            kwargs: dict[str, Any] = {}
            sig = inspect.signature(tool.handler)
            if "context" in sig.parameters:
                # `context or {}` would be a bug: an *empty* dict is falsy, so a caller
                # passing a fresh dict to collect out-of-band artifacts (audio bytes, say)
                # would silently receive a throwaway one instead and see nothing written
                # back. Identity, not truthiness.
                kwargs["context"] = {} if context is None else context
            if "trace" in sig.parameters:
                kwargs["trace"] = trace
            result = tool.handler(data, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, tool.output_model):
                result = tool.output_model.model_validate(
                    result.model_dump() if isinstance(result, BaseModel) else result
                )
            return result

        span_cm = trace.span(f"tool.{name}") if trace else None
        if span_cm is not None:
            with span_cm as sp:
                outcome = await run_resilient(
                    invoke, stage=name, timeout_s=tool.timeout_s,
                    retry=tool.retry, breaker=tool.breaker, fallbacks=fallbacks,
                )
                sp.attrs["attempts"] = outcome.attempts
                sp.attrs["ok"] = outcome.ok
                if outcome.fell_back:
                    sp.attrs["fell_back"] = True
                if not outcome.ok:
                    sp.status = "error"
                    sp.attrs["error"] = outcome.error
                return outcome

        return await run_resilient(
            invoke, stage=name, timeout_s=tool.timeout_s,
            retry=tool.retry, breaker=tool.breaker, fallbacks=fallbacks,
        )


registry = ToolRegistry()

__all__ = ["Tool", "ToolError", "ToolRegistry", "registry"]
