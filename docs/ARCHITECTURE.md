# Architecture

```
                             ┌─────────────────────────────────────────┐
   🎙  browser               │  React console (web/)                   │
   ───────────────────────▶  │  · push-to-talk orb, real RMS waveform  │
   MediaRecorder             │  · renders spans / verdicts / route      │
        │                    └─────────────────────────────────────────┘
        │  interim transcripts (Web Speech API)          ▲
        │        │                                       │ TurnResult (typed)
        │        ▼  ws /api/v1/stream                    │
        │   ┌──────────────────┐                         │
        │   │ speculative      │  embeds + retrieves the partial
        │   │ prefetch         │  while the user is still speaking
        │   └──────────────────┘
        │        ╲                                       │
        ▼         ╲                                      │
  POST /api/v1/voice                                     │
        │                                                │
        ▼                                                │
  ┌───────────────────────────────────────────────────────────────────────┐
  │  ElevenLabs Scribe  ──▶  transcript + language + confidence           │  stt_ms
  │  (retry · timeout · circuit breaker · browser fallback)               │  ┄┄┄┄┄┄
  └───────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┏━━━━━━━━━━━━━━━━━━━━━━━━ HARNESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  LatencyBudget(200 ms) · Trace(spans) · ToolRegistry(9 typed tools)   ┃
  ┃                                                                       ┃
  ┃  NORMALISE ─▶ ADMISSION ─▶ ROUTE ─▶ EMBED ─▶ SAFETY ─▶ RETRIEVE      ┃
  ┃            ─▶ TOPICALITY ─▶ ANSWER ─▶ VERIFY ─▶ TRUST ─▶ FINALISE     ┃
  ┃                   │                     │           │                 ┃  pipeline_ms
  ┃                   └─▶ REFUSE            └─▶ ABSTAIN ◀┘                 ┃  ≤ 200 ms
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        │                    │                      │
        ▼                    ▼                      ▼
  ┌───────────┐   ┌──────────────────────┐   ┌────────────────────┐
  │ Router    │   │ Hybrid retrieval     │   │ Guardrails ×7      │
  │ shape ─▶  │   │ dense per layer      │   │ sanity, injection, │
  │ layers,k  │   │  + BM25 (CSC)        │   │ safety, topicality,│
  │ ~0.3 ms   │   │  + RRF fusion        │   │ groundedness, PII, │
  └───────────┘   │  + parent expansion  │   │ trust floor        │
                  │  + late interaction  │   └────────────────────┘
                  └──────────────────────┘
                             │
        ┌────────────────────┴─────────────────────┐
        ▼                                          ▼
  ┌──────────────────────────┐        ┌──────────────────────────────┐
  │ Vector store             │        │ Polymorphic Chunk Lattice    │
  │ FlatStore (exact matmul) │        │ L0 atomic  L1 semantic       │
  │ or Qdrant (HNSW)         │        │ L2 window  L3 document       │
  └──────────────────────────┘        └──────────────────────────────┘
                             │
                             ▼
                  ai4bharat/MSMARCO-XI (15 languages)
```

---

## Module map

| path | responsibility |
|---|---|
| `server/setu/config.py` | every tunable in one place, env-overridable (`SETU_*`) |
| `server/setu/text.py` | script detection, Unicode tokenisation, sentence splitting across 5 writing systems |
| `server/setu/embed.py` | ONNX encoder, warm-up, LRU query cache |
| `server/setu/harness/` | contracts, span tracing, latency budget, retry/breaker policy, tool registry |
| `server/setu/chunking/` | the four chunkers + lattice builder with parent links and dedup |
| `server/setu/index/` | BM25 inverted index, vector stores, rank fusion, persistable corpus bundle |
| `server/setu/retrieval/` | query router, anytime hybrid engine, late-interaction reranker |
| `server/setu/answer/` | reflex (extractive) path, deliberate (LLM) path, localised composition |
| `server/setu/guardrails/` | 7 gates across 4 stages + composite trust score |
| `server/setu/speech/` | ElevenLabs Scribe (STT) and Flash (TTS) |
| `server/setu/telemetry.py` | always-on percentile ring buffer |
| `server/setu/pipeline.py` | the state machine that ties it together |
| `server/setu/app.py` | FastAPI routes + static SPA hosting |
| `web/src/` | React console — a renderer for the server's telemetry |

---

## Data flow, in detail

### Ingest (offline, once)

```
MS MARCO-XI parquet shard
   ↓  row → 2 documents per passage (English + translated, sharing a query_id)
Document(text, lang, meta{query_type, is_selected, passage_rank, …})
   ↓  sentence split → sentence embeddings (computed ONCE, shared by everything)
ChunkLattice
   ↓  4 chunkers in parallel → dedup coarse→fine → parent links
Chunk[]  (embed_text may carry a metadata header; text stays clean for citation)
   ↓  one embedding pass over chunks
CorpusIndex → data/index/{vectors,sent_vectors,cohesion}.npy
              data/index/{chunks,docs}.jsonl
              data/index/bm25_*.npz
```

The row-aligned invariant matters: **chunk row `i` is the same `i`** in the vector matrix, the
BM25 matrix, the layer masks and the chunk table. There is no id translation anywhere on the hot
path.

### Serve (per turn)

Measured P50 over 300 distinct queries:

```
query ─▶ NFKC normalise, detect script/language                      0.14 ms
      ─▶ sanity + injection gates            (no vectors)            0.2  ms
      ─▶ router: shape → {layer: k, weight}, dense/sparse split      0.25 ms
      ─▶ encode query (LRU cached, 2 intra-op threads)              55.0  ms
      ─▶ semantic safety gate (centroid + seed-max dot products)     0.2  ms
      ─▶ dense search per selected layer (boolean mask views)        3.9  ms
      ─▶ BM25 over query-term posting lists only                     0.5  ms
      ─▶ RRF + normalised blend + layer & cohesion priors            1.2  ms
      ─▶ parent expansion (fine hits → coherence blocks)             0.05 ms
      ─▶ span diversification (collapse overlapping evidence)        0.05 ms
      ─▶ late-interaction rerank over cached sentence vectors        2.6  ms
      ─▶ extractive span selection + stitching with citations        1.7  ms
      ─▶ topicality, groundedness, numeric, entity, PII, trust       0.7  ms
                                                              ─────────────
                                                                   70.5  ms   (SLO 200)
```

The encoder is ~78 % of the budget; everything after it is numpy over resident data. See
[LATENCY.md §2](LATENCY.md) for the thread-pool measurement that sets its cost.

---

## The five design decisions that define this system

### 1. Four chunk views instead of one good one
Any single chunk size is a bet on the shape of future questions. The lattice refuses the bet and
lets a 0.3 ms router choose per query. → [CHUNKING.md](CHUNKING.md)

### 2. Extraction as the primary answerer, generation as an escalation
MS MARCO passages were collected *because they contain the answer*. A span selector is the
right tool, it is ~400× faster than a hosted LLM, and it **cannot hallucinate a token**.
Generation is available, measured separately, and outside the SLO. → [LATENCY.md §5](LATENCY.md)

### 3. The deadline is a resource, not a hope
Optional stages ask the budget for permission and are skipped-and-reported if the answer is no.
The pipeline degrades quality, never the deadline. → [LATENCY.md §3](LATENCY.md)

### 4. Guardrails that also know when *to* answer
Semantic harm centroids give cross-lingual and paraphrase-robust coverage; a **benign centroid
set** stops the gate from refusing the medical, historical and security questions the corpus
actually contains. Both error directions are tested. → [GUARDRAILS.md](GUARDRAILS.md)

### 5. Observability as part of the response contract
Spans, guard verdicts with thresholds, router reasons and the budget report are all returned to
the client. The UI invents nothing. → [HARNESS.md](HARNESS.md)

---

## API surface

| method | route | purpose |
|---|---|---|
| `POST` | `/api/v1/ask` | text turn → `TurnResult` |
| `POST` | `/api/v1/voice` | audio turn → `TurnResult` (+ base64 TTS) |
| `POST` | `/api/v1/stt` | transcribe only |
| `POST` | `/api/v1/tts` | synthesise an answer |
| `POST` | `/api/v1/stt/realtime-token` | mint a single-use Scribe token (key never reaches the browser) |
| `POST` | `/api/v1/prefetch` | speculative retrieval on a partial transcript |
| `WS` | `/api/v1/stream` | partial → prefetch, final → turn |
| `GET` | `/api/v1/metrics` | live P50/P70/P90/P95/P99/P100 + per-stage + histogram |
| `GET` | `/api/v1/corpus` | lattice, layer counts, languages, store backends |
| `GET` | `/api/v1/tools` | JSON-schema tool manifest + circuit-breaker state |
| `GET` | `/api/v1/trace/{id}` | full retained trace for a turn |
| `GET` | `/api/v1/samples` | real corpus queries for the UI chips |
| `GET` | `/api/v1/health` | readiness, warm-up timings, backends |
| `GET` | `/api/docs` | OpenAPI |

---

## Deliberate extension points

- **Vector backend** — `VectorStore` protocol; `FlatStore` (exact) and `QdrantStore` (HNSW,
  server-side payload filters) both implement it. One env var switches.
- **Encoder** — `fastembed` ONNX with a `sentence-transformers` fallback; any multilingual model
  of the same shape drops in via `SETU_EMBED_MODEL`.
- **STT provider** — the brief allows Sarvam or ElevenLabs; the `transcribe` tool is a registry
  entry with a typed contract, so a Sarvam client is a new class and one registration.
- **Chunkers** — implement `Chunker` (`layer`, `name`, `split`) and add it to `ChunkLattice`.
  `SETU_ENABLED_LAYERS` toggles layers for ablation.
- **Guardrail gates** — implement `BaseGate.check`, add to `GuardrailStack`. Verdicts flow to the
  API and UI automatically.
- **LLM** — any OpenAI-compatible endpoint via `SETU_LLM_BASE_URL`.
