# The harness: orchestration around the model, not a prompt around it

> The brief: *"Your model/pipeline should be run inside a proper harness — structured orchestration
> around the model (tool calls, retries, structured input/output handling, error recovery) rather
> than a single raw prompt-in, text-out call."*

Four requirements, four mechanisms. `server/setu/harness/` + `server/setu/pipeline.py`.

---

## The state machine

Every turn is an explicit walk, not a nested chain of function calls:

```
 NORMALISE ─▶ GUARD_ADMISSION ─▶ ROUTE ─▶ EMBED ─▶ GUARD_SAFETY ─▶ RETRIEVE
           ─▶ GUARD_TOPICALITY ─▶ ANSWER ─▶ GUARD_OUTPUT ─▶ TRUST ─▶ FINALISE
                    │                          │                │
                    └──▶ REFUSE                └──▶ ABSTAIN ◀────┘
```

Terminal states are first-class: `answered`, `degraded`, `abstained`, `refused`, `error`. Every one
of them produces a complete, well-formed `TurnResult` with spans, guard verdicts and a budget
report — including the error state. There is no path that returns a bare string or throws into the
request handler.

---

## 1. Tool calls

Every capability is registered in a `ToolRegistry` with a **declared Pydantic input and output
model**. Nine tools:

| tool | timeout | retry | breaker | est |
|---|---|---|---|---|
| `normalize_query` | – | – | – | 0.3 ms |
| `route_query` | – | – | – | 0.4 ms |
| `embed_query` | – | – | – | 18 ms |
| `retrieve` | – | – | – | 45 ms |
| `answer_reflex` | – | – | – | 8 ms |
| `answer_deliberate` | 8 s | 2 × jittered | `llm` | 900 ms |
| `guard` | – | – | – | 3 ms |
| `transcribe` | 20 s | 2 × jittered | `elevenlabs-stt` | 700 ms |
| `speak` | 20 s | 2 × jittered | `elevenlabs-tts` | 600 ms |

Calling a tool goes through one path — `ToolRegistry.call` — which validates the input, opens a
trace span, applies the resilience policy, validates the output, and returns a structured `Outcome`
that **never raises at the call site**:

```python
outcome = await registry.call("retrieve", RetrieveIn(text=q, top_k=5),
                              trace=trace, context=context)
if not outcome.ok:
    return finalize(TurnStatus.ERROR, outcome.error)
```

`GET /api/v1/tools` returns the JSON-schema manifest. The registry is the same surface the
Deliberate path uses for real function calling, so there is one definition of every capability
rather than a Python function and a separate JSON schema that drift apart.

### Control plane vs data plane

Vectors and audio travel in an in-process `TurnContext`, not through the tool schema. Serialising a
384-float array at every hop would cost more than the retrieval it describes. The **control plane is
fully typed and validated**; the data plane is a typed dataclass passed by reference. This is a
deliberate boundary, not an omission — and it is documented here so nobody mistakes it for one.

---

## 2. Structured input/output

Nothing crosses a stage boundary as a loose dict. `server/setu/harness/contracts.py` defines every
shape: `AskRequest`, `RoutePlan`, `RetrievalResult`, `ChunkHit`, `AnswerDraft`, `Claim`,
`GuardVerdict`, `GuardReport`, `SpanModel`, `BudgetReport`, `TurnResult`.

The payoff is that a stage returning the wrong shape is a **validation failure the harness can
retry**, not a `KeyError` three stages later with no context.

This matters most on the Deliberate path, where the "stage" is a language model:

```python
class LLMAnswer(BaseModel):
    answer: str = ""
    citations: list[int] = []
    abstain: bool = False
    reason: str = ""
```

The model must return JSON matching that contract. A malformed body raises `DeliberateError`, which
the retry policy catches and retries with a fresh attempt. Beyond parsing, the output is
**validated for substance**:

- Citation markers that do not resolve to supplied evidence are stripped.
- Sentences carrying **no** resolvable marker are dropped entirely — an uncited sentence is an
  ungrounded sentence.
- If nothing survives, the tool fails and falls back to the reflex path rather than returning
  unattributable prose.
- `abstain` is a first-class field, so "I don't know" is a structured decision the guardrail layer
  can act on, not prose we have to detect with a regex.

---

## 3. Error recovery

### Retries with jitter
`RetryPolicy(attempts, base_delay_ms, multiplier, jitter, retry_on)` — exponential backoff with
±35 % jitter, so concurrent failures do not synchronise into a thundering herd.

### Circuit breakers
Three named breakers (`llm`, `elevenlabs-stt`, `elevenlabs-tts`). After 4 consecutive failures the
circuit opens and calls fail *immediately* for the cooldown, then half-open to probe. A degraded
third-party endpoint stops costing every turn its full timeout. Live state is exposed at
`GET /api/v1/tools` and `/api/v1/health`.

### Declared fallback chains

| primary | fallback | surfaced as |
|---|---|---|
| Deliberate (LLM) answering | Reflex extractive answering | `warnings: ["deliberate path unavailable - served by reflex path"]` |
| ElevenLabs Scribe | browser speech recognition | UI logs `browser transcript → …` |
| Qdrant vector store | in-process exact store | `vector_store.backend` in `/api/v1/corpus` |
| Router | balanced default plan | `warnings: ["router fell back to defaults"]` |
| fastembed (ONNX) | sentence-transformers | `encoder.backend` in `/api/v1/health` |

**A turn degrades; it does not crash.** And whatever degraded is *named* in the response — silent
fallback is worse than failure, because you cannot debug what you cannot see.

### Per-layer isolation
If one chunker throws on a pathological document, that layer yields `[]` and the document is still
indexed by the other three. A single bad passage cannot fail an ingest run.

---

## 4. Budget governance

The fourth property, and the one most harnesses lack: the harness owns a **deadline**, and optional
stages must ask permission.

```python
if budget.can_afford(cost_ms, reserve_ms=still_to_come):
    ...run...
else:
    budget.skip(name, cost_ms)          # recorded, returned to the client
```

This is what makes the SLO a property of the system rather than a hope about the average case. See
[LATENCY.md §3](LATENCY.md).

---

## Observability

Tracing is not a debug flag; it is part of the response contract.

```json
{
  "turn_id": "a3f9c1d20b48",
  "status": "answered",
  "pipeline_ms": 7.12,
  "slo_met": true,
  "spans": [
    {"name": "tool.embed_query", "start_ms": 0.31, "duration_ms": 4.17, "status": "ok"},
    {"name": "retrieve.dense",   "start_ms": 4.62, "duration_ms": 0.38,
     "attrs": {"layers": ["document","semantic","atomic","window"], "candidates": 40}},
    {"name": "rerank.late-interaction", "start_ms": 5.4, "duration_ms": 0.36,
     "attrs": {"reranked": 12}}
  ],
  "budget": {"total_ms": 200, "used_ms": 7.12, "slo_met": true, "skipped": [], "degraded": false},
  "guards": {"verdicts": [ /* score + threshold + reason per gate */ ]},
  "route": {"shape": "definition", "reason": "definition cue", "layers": [...]}
}
```

Spans nest, carry attributes, and include zero-duration marks for skipped and cache-hit stages. The
last 256 turns are retained and addressable at `GET /api/v1/trace/{turn_id}`.

**The frontend is a renderer for this payload, not a place where explanations are invented.** If the
UI shows a number, the server measured it — which is why the waterfall, the guard panel and the
route card required no server changes to build.

---

## What this buys, concretely

| failure | without a harness | with this harness |
|---|---|---|
| LLM returns prose instead of JSON | crash or garbage answer | validation failure → retry → fall back to reflex |
| LLM cites evidence that does not exist | confident wrong answer | markers stripped, uncited sentences dropped |
| ElevenLabs is down | every voice turn hangs for 20 s | breaker opens, fails fast, browser STT takes over |
| Index is larger than expected | SLO silently breached | rerank skipped, `degraded: true`, deadline held |
| A chunker hits a pathological passage | ingest run dies | that layer yields `[]`, document still indexed |
| Answer looks fine but is not grounded | shipped | groundedness gate abstains, reason returned |
| "Why was this slow?" | guesswork | span waterfall in the response |
