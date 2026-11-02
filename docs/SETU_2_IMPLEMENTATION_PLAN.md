# SETU 2.0 — Audit and Implementation Blueprint

Status: **audit complete, no code changed** (except `server/setu/__init__.py`, recreated earlier
so the project could start). This document is the input for the remaining SETU 2.0 steps
(frontend, ingestion, reasoning, evaluation lab, production readiness).

All paths are relative to the repo root (`setu-voice-rag/`). Line numbers refer to the code as of
this audit.

---

## Contents

1. [Audit: what exists today](#1-audit-what-exists-today)
2. [Audit: limitations and defects](#2-audit-limitations-and-defects)
3. [Target architecture](#3-target-architecture)
4. [Feature plans A–J](#4-feature-plans-aj)
5. [API plan](#5-api-plan)
6. [Database schema](#6-database-schema)
7. [Phasing, risks and open decisions](#7-phasing-risks-and-open-decisions)

---

## 1. Audit: what exists today

### 1.1 Backend (`server/setu/`)

| module | what it does | state |
|---|---|---|
| `pipeline.py` | Turn state machine: normalise → admission guard → route → embed (or speculative-prefetch reuse) → safety guard → retrieve → topicality guard → answer → output guard → trust floor → finalise. Stores the last 256 `TurnResult`s in memory. | Working |
| `harness/contracts.py` | Pydantic models: `AskRequest`, `TurnResult`, `ChunkHit`, `RetrievalResult`, `Citation`, `Claim`, `AnswerDraft`, `GuardVerdict`, `GuardReport`, `SpanModel`, `BudgetReport`. | Working. No session, conversation or collection fields. |
| `harness/registry.py`, `policy.py`, `tracing.py` | Typed tool registry (9 tools), retry with jitter, circuit breaker, timeouts, fallbacks, span tracing, `LatencyBudget` governor. | Working. Spans record counts only, not inputs/outputs. |
| `retrieval/router.py` | Rule-based query shape classifier (8 shapes) → per-layer weights, dense/sparse balance, rerank/expand flags. | Working |
| `retrieval/engine.py` | Per-layer dense search → BM25 → RRF + score blend → parent expansion → diversification (≤2 per doc) → late-interaction rerank (top 12) → emit top-k. Budget-aware: sparse, expand and rerank are skipped when the budget can't afford them. | Working. No metadata filtering. |
| `retrieval/rerank.py` | `0.46·maxsim + 0.16·meansim + 0.22·lexical + 0.06·numeric + 0.10·brevity` over cached sentence vectors. | Working |
| `index/corpus.py` | `CorpusIndex`: row-aligned chunk table, vectors, sentence vectors, layer masks, BM25, manifest. Saves to `data/index/` as `.npy` / `.jsonl` / `.npz` files. | Working. **Full rebuild only**, no add/delete. |
| `index/store.py` | `FlatStore`: exact matmul search with boolean masks. Has `upsert`, no delete. | Working |
| `index/qdrant_store.py` | Optional Qdrant backend (`SETU_VECTOR_BACKEND`). Payload `{layer, lang, doc_id}`. `search_filtered` exists but is never called. | Partial (see §2.3) |
| `index/bm25.py` | BM25 as a precomputed CSC sparse matrix. Static `fit()`; `candidate_mask` parameter unused. | Working, static |
| `index/fusion.py` | `RankFuser`: `0.72·norm(rrf) + 0.28·norm(blend) + 0.05·cohesion`. | Working |
| `chunking/` | Polymorphic Chunk Lattice: L0 atomic, L1 semantic (embedding-drift boundaries), L2 adaptive window, L3 document (metadata-prefixed). Coarse→fine dedup and `parent_id` links. Sentence embeddings computed once and shared. | Working. Designed for short passages (see §2.2). |
| `ingest/msmarco_xi.py`, `ingest/build.py` | Downloads MS MARCO-XI shards from Hugging Face → `corpus.jsonl` (`{doc_id, text, lang, meta}`) → full index build. | Working. JSONL input only. |
| `answer/extractive.py` | Reflex path: scores sentences (semantic, lexical, retrieval prior, shape, language, position), greedy selection with redundancy veto, inline `[n]` markers. ~0.4 ms. | Working |
| `answer/llm.py` | Deliberate path: OpenAI-compatible `/chat/completions` (default Groq, `groq/compound-mini`), temperature 0, 320 max tokens, JSON contract `{answer, citations, abstain, reason}`, citation-index validation. Single-turn, no tool calling. | Working when a key is set. |
| `answer/compose.py` | Localised refusal/abstention messages in 15 languages; citation stitching. | Working. Messages name "MS MARCO-XI". |
| `guardrails/` | 7 gates: sanity, injection (regex), semantic safety (harm vs benign centroids), topicality (score + margin + temporal deixis + lexical anchor), groundedness (per-claim containment / semantic support, numeric hallucination, entity check), PII redaction (Verhoeff, Luhn), trust floor 45. | Working |
| `embed.py` | `paraphrase-multilingual-MiniLM-L12-v2` (384-d) via fastembed ONNX, 2 threads, length-bucketed batching, 4096-entry query LRU. | Working |
| `speech/elevenlabs.py` | Scribe v2 batch STT, single-use realtime token, TTS with a voice per language (`eleven_flash_v2_5` for en/hi/ta, `eleven_multilingual_v2` otherwise). Key stays server-side. | Working when a key is set. |
| `telemetry.py` | In-memory ring buffer (4096 turns) → P50…P100 per stage, histogram. | Working |
| `app.py` | FastAPI routes under `/api/v1`, WebSocket `/stream`, SPA hosting from `web/dist`. | Working |
| `cli.py` | `ingest`, `build`, `serve`, `ask`, `stats`, `sample`. | Working |

**How the LLM path is activated.** The answer path is chosen **per request** by
`AskRequest.mode` (`reflex` default, `deliberate`, `auto`). An LLM key (`GROQ_API_KEY`,
`OPENAI_API_KEY` or `SETU_LLM_API_KEY`) only *enables* it (`settings.llm_enabled`). `auto` uses the
LLM when the query shape is description/list/definition and the top 4 hits come from ≥3 documents
(`pipeline._needs_synthesis`). `settings.answer_mode` exists but is never read. The frontend
hardcodes `mode: "reflex"`, so the LLM path is unreachable from the UI today.

**Persistence.** There is **no database**. Everything except `data/index/*` lives in process memory
and is lost on restart: traces, metrics, the prefetch cache and the question-bank cache.

### 1.2 Frontend (`web/`)

- React 18 + TypeScript 5.6 + Vite 5. Only runtime dependencies are `react` and `react-dom`. No router,
  no state library, no data-fetching library, no chart library, no tests.
- Single page (`App.tsx`) holds all state and passes props down. Overlays (`Welcome`,
  `QuestionBrowser`) are toggled by booleans.
- Components: `AskPanel` (mic orb, input, keyboard shortcuts), `Answer` (citation markers, copy,
  plain-language refusal reasons), `Sources`, `Examples`, `QuestionBrowser`, `LanguagePicker`
  (portal listbox, 15 languages), `Header`, `Hero`, `Notice`, `Welcome`, `ui` (icons).
- Hooks: `useMic` (MediaRecorder + analyser level), `useLiveSpeech` (browser SpeechRecognition for
  interim text and no-key fallback), `useVoiceOut` (ElevenLabs or `speechSynthesis`, native voice
  preference), `useTheme`.
- Styling: hand-written CSS with design tokens in `styles/tokens.css`; light (default) and dark
  themes. `app.css` is ~1,700 lines.
- `types.ts` mirrors `contracts.py` by hand.
- **API surface the UI never uses:** `/corpus`, `/metrics`, `/tools`, `/trace/{id}`,
  `/stt/realtime-token`, the `deliberate`/`auto` modes, and most of `TurnResult` (`spans`, `route`,
  `retrieval`, `guards.verdicts`, `budget`, `claims`). This is useful: the pipeline visualizer can
  start from data the backend already returns.

### 1.3 Evaluation, tests and deployment

- `scripts/bench.py` (latency, gates CI), `evaluate.py` (recall@k, MRR@10, answer token recall,
  `abstain_wrong`), `redteam.py` (40 cases, gates CI), `ablate.py` (chunking/retrieval ablation;
  **`docs/ablation.json` is missing** even though three docs cite it).
- `tests/`: 133 test functions across chunking, embedding, guardrails, harness, retrieval,
  telemetry and text. **Nothing tests `app.py`, the WebSocket, speech or the frontend.**
- Docker (two-stage, bakes the encoder and sample index), Compose with an optional Qdrant profile,
  `fly.toml`, `render.yaml`, GitHub Actions (ruff, pytest, bench gate, redteam gate, web build,
  docker build).

### 1.4 Current sample corpus

| fact | value | consequence |
|---|---|---|
| Size | 700 passages, 2,945 lattice chunks | Small enough that any question outside it is declined |
| Languages | 10 (en, bn, hi, kn, ml, pa, gu, ta, te, mr), ~70 passages each | |
| Distinct questions | 7–8 per language, **the same questions translated** | The "question bank" is really ~8 topics |
| Passage length | avg 295 chars (L3 document layer) | The lattice was tuned for short web passages |
| Gold labels | `data/corpus/queries.jsonl` is absent; `is_selected` lives in doc meta | `evaluate.py` falls back to doc meta |
| Domain | MS MARCO web snippets (definitions, trivia) | No business content at all |

The README's quality and latency figures (2,361 passages / 10,841 chunks) come from a larger ingest
than the index currently on disk (700 / 2,945). Both are valid, but the README should say which
index produced which number.

---

## 2. Audit: limitations and defects

### 2.1 Defects found (to fix during SETU 2.0, not now)

| # | where | defect |
|---|---|---|
| D1 | `pipeline.py:573-582` | **LLM abstention is overridden.** When the deliberate path abstains it returns an empty draft with `ok=True`, so the code falls through to `answer_reflex`, which answers anyway. A deliberate "I can't answer" is silently replaced. |
| D2 | `pipeline.py:548-552` | `DEGRADED` status is set *after* `finalize()` records telemetry, so metrics count degraded turns as `answered`. |
| D3 | `engine.py` `_to_hit` | `ChunkHit.score` is overwritten by the rerank score, so the fused score is lost; `top_score`/`margin` mix rerank and fusion scales when only some hits were reranked. |
| D4 | `engine.py` | `rank_dense` is the rank within one layer's list, not a global rank. Rerank sub-signals (maxsim, lexical…) are computed and discarded. |
| D5 | `engine.py` sparse stage | BM25 ignores the routed layers; dense is layer-routed, sparse is not. |
| D6 | `pipeline.py` prefetch | Prefetch cache is global across all clients, has no TTL, overrides the current route, ignores `req.top_k`, and its spans are missing from the turn trace. |
| D7 | `pipeline.voice` | Spans from two `Trace` objects with different origins are concatenated, so the voice waterfall is misaligned. |
| D8 | `qdrant_store.py` | Upsert is skipped when the point count matches, so a rebuilt index with the same size keeps stale vectors. `SETU_VECTOR_BACKEND=qdrant` without a URL silently uses an in-memory Qdrant. Errors fall back to flat silently. |
| D9 | `corpus.py load()` | Doesn't check that the index's `embed_model`/`dim` match the running encoder. |
| D10 | `policy.py` | The circuit breaker counts every failed retry attempt, so one call with 2 attempts uses 2 of the 3 allowed failures. |
| D11 | `app.py /voice`, WS | An invalid `mode` raises `ValueError` → HTTP 500. `/prefetch` and `/tts` take unvalidated dicts. |
| D12 | app-wide | CPU-bound work (encoding, numpy search, guards) runs on the asyncio event loop. Concurrent requests serialise and timeouts can't interrupt it. |
| D13 | docs | Inconsistent latency figures across README, LATENCY.md and SUBMISSION.md; several docs describe UI panels (waterfall, guard panel, header metrics badge) that no longer exist; `docs/ablation.json` missing. |
| D14 | `config.py` | `answer_mode`, `AskRequest.explain`, `LatencyBudget.deadline_s`, `Trace.event`, `GuardrailStack.components`, `FlatStore.save/load`, `StageFailed` are dead code. |

### 2.2 Structural limits that block SETU 2.0

1. **The index can't change incrementally.** Rows are positional; masks, BM25 statistics and the
   sentence table assume one contiguous build. There is no add, no delete, no notion of a
   collection.
2. **MS MARCO assumptions are baked in.** Doc ids are parsed as `qid:rank:lang` in
   `engine._passage_group`, `corpus.parallel_doc_id`/`group_doc_ids`, `extractive._sources` and
   `app._question_bank`. The metadata header (`chunking/metadata.py`) expects `passage_rank`,
   `query_type`, `is_selected`. Arbitrary documents will make the margin check and the
   same-language preference behave wrongly.
3. **The encoder truncates at 128 tokens.** `paraphrase-multilingual-MiniLM-L12-v2` has
   `max_seq_length = 128`. For a 295-char MS MARCO passage the L3 document node is meaningful; for a
   60-page annual report, a "whole document" vector is the embedding of its first paragraph.
   Long documents need a **segmentation step** before the lattice (see §3.4).
4. **Single-turn everything.** No conversation id, no history, no query rewriting; the LLM prompt is
   one evidence block plus one question.
5. **Answers are capped for snippets.** Reflex answers are capped at 3 sentences / 620 chars; the
   LLM at 320 tokens. Detailed research, comparisons and executive summaries need different limits
   and a different (non-200 ms) budget.
6. **Groundedness blocks the whole answer.** Coverage < 0.42 or any hallucinated number rejects the
   entire answer. That is right for a 2-sentence answer and wrong for a 15-claim report, where one
   unsupported claim should be removed or flagged, not the whole report withheld.
7. **Traces are shallow and ephemeral.** Spans hold counts, not candidate lists; 256 in memory.
   A pipeline explorer needs persisted, per-stage inputs and outputs.

### 2.3 Security

| risk | today | plan |
|---|---|---|
| No authentication | every route is open | Optional `SETU_API_TOKEN` for mutating and paid routes; anonymous session id for conversation ownership (decision O2) |
| Paid-API proxy | `/tts`, `/voice`, `/stt`, `/stt/realtime-token`, deliberate `/ask` spend ElevenLabs/LLM quota for anyone; `/tts` takes unbounded text and caller-chosen `voice_id` | Token gate, per-IP rate limit, text length cap, `voice_id` allowlist |
| Unbounded uploads | `await audio.read()` reads any size into memory | Streaming size cap on every upload route |
| CORS | `*` with `allow_credentials=True`; no WebSocket Origin check | Explicit origin list in production; WS Origin check |
| Trace exposure | `/trace/{id}` returns anyone's query by id; `DELETE /metrics` is open | Scope traces to session; token-gate admin routes |
| Error leakage | WS echoes exception text | Generic client error + server log |
| SSRF (future URL ingest) | n/a | §4.C |
| Prompt injection via documents (future) | n/a | §4.C / §4.E |

### 2.4 Performance and scalability

- One uvicorn worker; all state in process memory; can't scale horizontally as-is.
- Event-loop blocking (D12) means one slow request stalls the WebSocket and every other request.
- Exact search is a full matmul; fine to roughly 10⁵ vectors, then Qdrant.
- Render starter is 512 MB; the index is fully resident.
- `/voice` returns base64 audio in JSON (+33% payload).
- Fly/Render volumes mounted at `/app/data` start empty and hide the image's baked index.
  First boot may fail unless `load_or_build` can rebuild from a path outside the volume (unverified).

---

## 3. Target architecture

### 3.1 Principles

1. **Extend, don't replace.** The lattice, hybrid retrieval, RRF, late-interaction rerank, harness
   and guardrails stay. New capabilities are new stages, new tools and new modules around them.
2. **Quick answers keep the 200 ms SLO.** `bench.py` must still pass for the Quick Answer mode on the
   sample collection. Longer modes get their own measured budgets and are reported separately, as
   the deliberate path already is.
3. **Everything degrades and says so.** No LLM key → extractive fallbacks that are clearly labelled,
   never a fake "analysis". No ElevenLabs key → browser speech (already works).
4. **No number without a source.** Dashboard and evaluation numbers come from the database or the
   live index. Nothing is hardcoded.

### 3.2 Component diagram

```
                         ┌──────────────────────── web (React SPA) ────────────────────────┐
                         │ Dashboard · Research · Knowledge · Explorer · Eval Lab · Settings │
                         └───────────────┬─────────────────────────────────────┬────────────┘
                                         │ REST /api/v1 (typed, OpenAPI-gen)   │ WS /api/v1/stream
┌────────────────────────────────────────▼─────────────────────────────────────▼──────────────┐
│ FastAPI app                                                                                  │
│  routers: health · collections · documents · jobs · ask · conversations · analysis ·         │
│           traces · eval · speech · dashboard · settings                                      │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Services                                                                                     │
│  IngestionService ──► JobRunner (thread pool)       ConversationService                      │
│  CollectionRegistry (CorpusIndex per collection,    QueryUnderstanding (resolve, classify,   │
│     atomic swap on rebuild)                            split, expand)                         │
│  SetuPipeline (existing state machine + new stages) Generation (5 styles) · Analysis tools   │
│  EvaluationService                                  CitationVerifier · ContradictionDetector │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Storage                                                                                      │
│  SQLite (SQLAlchemy 2 + Alembic) ─ collections, documents, chunks(meta), jobs, conversations, │
│                                    messages, traces, eval runs, embedding cache               │
│  data/collections/<cid>/index/   ─ existing CorpusIndex files (vectors, BM25, chunks.jsonl)   │
│  data/collections/<cid>/raw/     ─ original uploads (never executed)                          │
│  Qdrant (optional)               ─ one Qdrant collection per SETU collection                  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Frontend architecture

| concern | choice | why |
|---|---|---|
| Routing | `react-router-dom` | Seven pages need URLs (`/research/:conversationId`, `/explorer/:turnId`) so they can be shared and refreshed |
| Server state | `@tanstack/react-query` | Caching, polling for ingestion jobs, pagination, loading/error states without hand-rolled `useEffect`s |
| Charts | `recharts` | Mature, SVG, works with CSS variables; enough for latency histograms, recall curves, ablation bars |
| API types | `openapi-typescript` generated from `/api/openapi.json` | Removes the hand-synced `types.ts` drift. Requires typed response models on the backend (§5) |
| Styling | Keep the custom token system; add a navy/charcoal **dark-default** theme with teal/cyan/violet accents | Reuses `tokens.css` naming; no UI-kit lock-in. Light theme stays available |
| Reused as-is | `useMic`, `useLiveSpeech`, `useVoiceOut`, `LanguagePicker`, `languages.ts`, citation rendering from `Answer`, `StreamChannel` | These work and are the hardest parts to rebuild |

Page map:

| route | page | data source |
|---|---|---|
| `/` | Overview dashboard | `GET /dashboard/summary`, `/metrics`, `/health` |
| `/research`, `/research/:id` | Research workspace | `/conversations`, `/ask`, `/stream` |
| `/knowledge`, `/knowledge/:cid`, `/knowledge/docs/:did` | Knowledge base | `/collections`, `/documents`, `/jobs` |
| `/explorer`, `/explorer/:turnId` | RAG explorer | `/traces`, `/trace/{id}` |
| `/evaluation` | Evaluation lab | `/eval/*` |
| `/voice` | Voice interface | `/voice`, `/stt`, `/tts`, `/stream` |
| `/settings` | Settings & diagnostics | `/settings`, `/health`, `/tools` |

### 3.4 Data ingestion architecture

```
upload / URL ──► validate ──► store raw ──► extract ──► clean ──► metadata ──► segment
                                                                               │
   ready ◄── persist meta ◄── index (vector + BM25, atomic swap) ◄── embed ◄── lattice chunk
```

- **Validation:** extension *and* magic-byte allowlist (PDF, DOCX, TXT, MD, CSV, HTML via URL),
  per-file and per-request size caps, filename sanitised; originals stored under a generated id,
  never under the user's filename, and never executed.
- **Extraction:**

  | format | library | keeps |
  |---|---|---|
  | PDF | `pypdf` (BSD) | page numbers. Scanned PDFs have no text layer; OCR is out of scope and the job is marked `failed: no extractable text` |
  | DOCX | `python-docx` (MIT) | heading styles → sections |
  | Markdown | stdlib parsing of `#` headings | sections |
  | TXT | stdlib, charset detection | — |
  | CSV | stdlib `csv` | header row; each row rendered as `col: value; …` text, row number as locator |
  | URL (HTML) | `trafilatura` (Apache-2.0) | title, publication date, main text without boilerplate |

  `pymupdf` would extract better but is AGPL; not recommended for an MIT project.
- **Segmentation (new, fixes §2.2.3):** documents longer than ~2,500 chars are split into
  **sections** on headings, then pages, then paragraph boundaries. Each section becomes a lattice
  `Document` whose `doc_id` is `<document_id>#s<n>` and whose `meta` carries `document_id`,
  `section`, `page_start`, `page_end`. The existing four layers then run per section, so L3 means
  "section" for long documents and "whole document" for short ones. Page and heading for any chunk
  are recovered from its `char_start`/`char_end` via the section's offset table, which the lattice
  already preserves.
- **Embedding cache:** a SQLite table keyed by `(model, sha256(text))` stores float32 vectors for
  sentences and chunk `embed_text`. Rebuilding a collection then re-chunks and refits BM25 but only
  embeds *new* text.
- **Background processing:** an in-process `JobRunner` (`ThreadPoolExecutor`, 1 worker by default)
  with job state in SQLite. Celery/Redis is not justified for a single-process deployment; the
  `JobRunner` interface keeps that option open.

### 3.5 Metadata database

SQLite file at `data/setu.db`, accessed through **SQLAlchemy 2.0 Core + Alembic migrations** behind
repository classes (`CollectionRepo`, `DocumentRepo`, …). Moving to PostgreSQL is a connection-string
change plus running the same migrations. WAL mode, foreign keys on. Schema in §6.

### 3.6 Vector database and indexing

**Decision: one `CorpusIndex` per collection, rebuilt and atomically swapped on change.**

| option | pros | cons |
|---|---|---|
| **A. Index per collection, rebuild on change** (chosen) | Reuses `CorpusIndex`, lattice, BM25, masks unchanged. Collection isolation is physical. With the embedding cache, a rebuild of a 5k-chunk collection is chunking + BM25 fit (seconds) | Rebuild cost grows with collection size; past ~50k chunks per collection a segmented index is needed |
| B. Make `CorpusIndex` incremental | No rebuilds | Rewrites the most delicate code (row-positional masks, sentence offsets, global BM25 statistics); high regression risk to the latency SLO |
| C. Qdrant as primary, BM25 elsewhere | Native delete/filter | Makes Qdrant mandatory; BM25 still needs rebuilding; loses the 0.1 ms in-process path |

- `CollectionRegistry` holds `{collection_id: CorpusIndex}`. A rebuild builds a new index in the
  background and swaps the reference; in-flight queries keep the old one.
- The existing MS MARCO sample becomes a built-in read-only collection `msmarco-sample` loaded from
  `data/index/` (backward compatible; `bench.py`, `evaluate.py`, `redteam.py` keep working).
- **Multi-collection search:** run `RetrievalEngine.search` per selected collection, then merge hit
  lists with RRF (rank-based, so BM25 scale differences between collections don't matter), then
  rerank the merged top-k once. Rerank scores are comparable across collections because they come
  from the same encoder.
- **Qdrant:** one Qdrant collection per SETU collection (`setu_<cid>`), re-upserted on every
  rebuild using a content version hash instead of the point count (fixes D8). Flat stays the
  default.
- **Generalise MS MARCO assumptions:** `_passage_group` and the parallel-language lookup only apply
  when doc meta says `source == "msmarco-xi"`; for other documents the passage group is
  `document_id`.

### 3.7 Embedding generation

Keep `paraphrase-multilingual-MiniLM-L12-v2` (384-d, multilingual, fast enough for the SLO). The
128-token limit is handled by segmentation and the existing small-to-big lattice rather than by a
bigger model. The model name is recorded per chunk and per collection; `load()` refuses an index
built with a different model (fixes D9). A stronger model (for example multilingual-e5-base or
bge-m3) is a later, measured change: it means re-embedding everything and re-running the latency
benchmark.

### 3.8 Hybrid retrieval and reranking

Unchanged algorithm. Changes:

- **Filters:** `RetrievalEngine.search(..., filters)` accepts `collection_ids` (via registry),
  `document_ids`, `langs`, `date_range`. They become a boolean row mask ANDed with the layer mask
  for dense search and passed as `candidate_mask` to BM25 (both parameters already exist).
- **Configurable knobs per request:** `top_k`, `dense_k`, `sparse_k`, `rerank` on/off,
  `retrieval_mode` (`hybrid` | `dense` | `sparse`), validated with safe bounds. Used by the
  workspace and the evaluation lab.
- **Score fidelity (fixes D3, D4):** `ChunkHit` gains `fused_score`, `rank_fused`,
  `rank_dense_global`, and `rerank_signals {maxsim, meansim, lexical, numeric, brevity}`.
  `score` keeps its current meaning for backward compatibility.
- **Query expansion:** pseudo-relevance feedback (top terms from the top 3 BM25 hits added to the
  sparse query) behind a flag. LLM-based expansion only when a key is present, and only for the
  non-Quick modes.

### 3.9 Context assembly (new stage)

Between retrieval and generation. For non-Quick modes:

1. Group hits by document; expand to parent sections (existing `parent_id`).
2. Deduplicate overlapping spans (existing diversification logic).
3. Fit a token budget per mode (e.g. Quick 0, Detailed ~3k, Comparative ~4k tokens) with a
   per-document quota so one long report can't crowd out the others in a comparison.
4. Wrap every passage in delimiters with its citation id, document title, page, date and URL, and an
   explicit instruction that passage text is **data, not instructions** (§4.E injection defence).
5. Record the assembled context in the trace for the explorer.

### 3.10 LLM generation

- Keep the OpenAI-compatible client; any compatible provider works via `SETU_LLM_BASE_URL`.
- New request field **`style`**: `quick | detailed | comparative | critical | executive`, separate
  from the existing `mode` (`reflex | deliberate | auto`). `mode` chooses the engine, `style` the
  output shape, so existing clients are unaffected.
- One JSON contract for all styles:

  ```json
  {
    "abstain": false,
    "missing": "what evidence would be needed, if abstaining or partial",
    "sections": [{"heading": "...", "claims": [
      {"text": "...", "kind": "fact|calculation|interpretation|assumption", "citations": [3, 7]}
    ]}],
    "table": {"columns": ["..."], "rows": [["...", "[3]"]]},
    "follow_ups": ["..."]
  }
  ```

- **Calculations are done in Python, not by the model.** The model may propose
  `{"op": "pct_change", "inputs": [claim_ref_a, claim_ref_b]}`; the backend extracts the cited numbers,
  computes the value and labels it `calculation`. This keeps "calculated values" verifiable.
- `max_tokens`, timeout and budget come from the style, not the global 320-token cap.
- Without a key: Quick → existing reflex path. Other styles → an **extractive structured fallback**
  (evidence grouped by document/section, with the reflex answer per group), labelled
  "Extractive summary — no language model configured".

### 3.11 Citation verification

Reuse `GroundednessGate` claim-level support (lexical containment + semantic max-sim over sentence
vectors, numeric consistency, entity check), applied **per claim to the passages that claim cites**,
not to the whole retrieved set.

- `fact` claims below the support threshold are **removed** and listed under "unsupported claims
  removed" in the response, instead of withholding the whole answer (fixes §2.2.6). The current
  whole-answer block stays for Quick mode.
- A citation that doesn't support its claim is dropped; if a claim loses all citations it is removed.
- `interpretation` and `assumption` claims are shown with their label and are not required to be
  entailed, but any number inside them must appear in the evidence or in a `calculation`.
- Citations carry document title, page, section, URL and the supporting sentence (the
  `best_sentence` the reranker already computes).

### 3.12 Guardrails

The seven gates stay. Additions:

- **Document-side injection scan at ingest:** run `InjectionGate` patterns over every chunk; flagged
  chunks are kept but marked `meta.injection_flag`, shown in the UI, and quoted with an extra warning
  in assembled context.
- **Topicality per collection:** the score/margin thresholds were tuned on MS MARCO; they become
  per-collection settings with the current values as defaults, recalibrated by the evaluation lab.
- **Localised messages:** remove the hardcoded "MS MARCO-XI" from refusal texts.

### 3.13 Conversation storage

`conversations` and `messages` tables (§6). Each assistant message stores the `turn_id`, the
resolved query, the cited chunk ids and the style. Only the **relevant** context is used:
the last 3 turns' resolved queries and entities for query resolution, and the previously cited
documents as a soft retrieval prior. The full history is never pasted into the prompt.

### 3.14 Evaluation

A new `server/setu/evaluation/` package; the existing scripts become thin CLIs over it, so the lab
UI and CI run the same code. Detail in §4.I.

---

## 4. Feature plans A–J

Legend: **R** = reuse existing code, **E** = extend existing code, **N** = new implementation.

### A. Business intelligence dashboard

| item | source | R/E/N |
|---|---|---|
| Documents, chunks, collections, per-language/per-layer counts | `CollectionRegistry` stats + `documents` table; `CorpusIndex.stats()` | E |
| Embedding model, dim, backend, vector store | `/health.detail.encoder`, `index.manifest` | R |
| Retrieval performance (live P50/P95 per stage) | `/metrics` (`telemetry.py`) | R |
| Latest stored evaluation (recall@k, MRR) | `eval_runs` table | N |
| Recent research sessions, uploads, activity | `conversations`, `documents`, `jobs` tables | N |
| System health, circuit breakers | `/health`, `/tools` | R |

New endpoint `GET /dashboard/summary` aggregates these so the page makes one request.

### B. Advanced research workspace

| item | R/E/N | notes |
|---|---|---|
| Conversation sidebar, history, new/delete | N | `ConversationService` + `/conversations` |
| Text + voice input, language selector | R | `AskPanel`, `useMic`, `useLiveSpeech`, `LanguagePicker` |
| Five answer styles | N (backend §3.10), E (`AskRequest.style`) | Quick works without a key |
| Collection selector | N | `collection_ids` on `/ask` |
| Citations + evidence panel with page/URL/score | E | `Answer`/`Sources` components; scores now shown in an expandable "details" view |
| Follow-up suggestions | N | From the LLM contract `follow_ups`; without a key, from query-bank neighbours in the same collection |
| Copy / export (Markdown, JSON) | E | Copy exists; export is client-side from the stored message |

### C. Document and URL ingestion

| step | R/E/N | notes |
|---|---|---|
| Upload API (multi-file), validation, raw storage | N | `python-multipart` already installed |
| Extraction (PDF/DOCX/TXT/MD/CSV/HTML) | N | §3.4 libraries |
| Cleaning (whitespace, hyphenation, headers/footers repeated across pages) | N | |
| Metadata (title, author, dates, page count, language via `text.detect_lang`) | N + R | |
| Segmentation | N | §3.4 |
| Chunking | R | `ChunkLattice` unchanged |
| Embedding | R + N | `Encoder.encode` + embedding cache |
| Vector + BM25 indexing | R | `CorpusIndex.build` per collection |
| Status states `uploaded → extracting → chunking → embedding → indexing → completed / failed` | N | `jobs` table, polled by the UI |

**URL ingestion security (SSRF):**

1. Only `http`/`https`, only ports 80/443, no credentials in the URL.
2. Resolve DNS once; reject if *any* resolved address is loopback, private, link-local, CGNAT
   (100.64/10), multicast, reserved, unspecified, or IPv4-mapped/6to4 forms of those; connect to the
   **resolved IP** (prevents DNS rebinding) with the original Host/SNI.
3. `follow_redirects=False`; follow at most 3 redirects manually, re-validating each hop.
4. `robots.txt` checked with `urllib.robotparser` for the SETU user agent; disallowed → refused.
5. Streamed download with a byte cap and timeout; `Content-Type` allowlist (`text/html`,
   `application/pdf`, `text/plain`); no other file types downloaded.
6. Identify with a descriptive `User-Agent`; per-domain rate limit.

**Curated business collection.** Sources such as RBI, SEBI, MCA, World Bank and company
investor-relations pages differ in licence: World Bank material is largely CC BY 4.0; regulator
publications generally allow reproduction with attribution; company annual reports are copyrighted.
Plan: commit a **source manifest** (`data/sources/business.yaml`: URL, publisher, title, date,
licence note) and a fetch command that downloads at install time through the same safe URL
pipeline, rather than committing third-party PDFs. Every URL is verified (reachable, robots-allowed)
before it's added. If a source can't be fetched, synthetic documents are generated into a
**separate** collection named `synthetic-demo`, with `meta.synthetic = true`, a visible
"Synthetic demonstration data" badge, and a disclaimer on every citation from them.

### D. Persistent knowledge collections

| item | R/E/N |
|---|---|
| CRUD collections, add/remove documents | N (`CollectionRepo`, routes) |
| Per-collection index, atomic rebuild, isolation | E (`CorpusIndex` per collection, `CollectionRegistry`) |
| Inspect chunks per document (paginated, with layer, page, section, hash) | N (route) + R (`chunks.jsonl` data) |
| Search one / several collections | E (§3.6 multi-collection merge) |
| Built-in `msmarco-sample` collection | R |

### E. Multi-document reasoning

| capability | approach | R/E/N |
|---|---|---|
| Multi-part questions | Split on question marks and coordinators ("and", "vs", "compare … with …") into sub-queries; retrieve each; union with per-sub-query quotas | N |
| Comparative questions | Entity extraction (capitalised spans, document titles, collection document names) → one retrieval per entity, filtered to that entity's documents when known | N |
| Evidence balance | Context assembly per-document quota (§3.9) | N |
| Contradiction detection | Deterministic first: same metric + same period + different numeric value across documents → conflict. LLM second (when a key exists): pairwise check of claims about the same entity, output labelled "model-detected, unverified". Shows both passages and their publication dates; never picks a winner | N |
| Injection in retrieved text | Delimited evidence, ingest-time flagging, post-generation check that the answer doesn't follow passage instructions (e.g. no URLs/instructions absent from the question) | N + R (`InjectionGate` patterns) |

### F. Conversation context management

| item | approach | R/E/N |
|---|---|---|
| Follow-up resolution ("How does it compare with last year?") | Rule-based first: detect anaphora (it, they, this, that company, उसका, ಅದು…) and relative time ("last year") → substitute entities/periods from the last turn's resolved query. LLM rewrite when a key exists and the rule-based pass is unsure. The **resolved query is shown to the user** and stored | N |
| Context-aware retrieval | Previously cited documents get a small fusion prior | E (`RankFuser`) |
| Persistence | `conversations`, `messages` | N |
| Prefetch per session | Key the prefetch cache by session id + text, add a TTL (fixes D6) | E |

### G. RAG pipeline visualizer

The backend already returns spans, route, retrieval, verdicts, budget and claims. What's missing:

| gap | change | R/E/N |
|---|---|---|
| Stage inputs/outputs | `trace_level` on `/ask` (`basic` default, `full` for the explorer). At `full`, each stage records a compact input/output snapshot: normalised query, detected language + method, route plan, per-layer dense candidates (id, doc, score, rank), BM25 candidates (id, score), fused list (rrf, blend, final), rerank list with sub-signals, parent expansions, assembled context, raw LLM output (truncated), citation verification per claim, guard verdicts | E (`Trace`, `RetrievalEngine`) |
| Persistence | `traces` table (JSON), retention by count/age, scoped to the session | N |
| Skipped stages | Already recorded by `LatencyBudget.skip`; render as skipped with the reason | R |
| Voice span alignment | One trace origin for voice turns (fixes D7) | E |

`basic` stays the default so Quick-mode latency is unaffected; `bench.py` measures `basic`.

### H. Voice interaction improvements

| item | R/E/N | notes |
|---|---|---|
| Waveform | E | `useMic` already exposes an analyser; switch from RMS bars to time-domain waveform on `<canvas>` |
| Recording indicator, timer | R | |
| Live transcription | E | Browser interim text today. With a key: ElevenLabs Scribe realtime via the existing `/stt/realtime-token` (server-minted, key never reaches the browser) |
| Stop/resume | N | `MediaRecorder.pause()/resume()` where supported; hidden where not |
| Spoken playback, per-language voices | R | `useVoiceOut`, `VOICE_BY_LANG` |
| Transcript history | N | Voice turns are stored as conversation messages with `transcript` |
| Scribe failure → browser fallback | E | Today a Scribe error surfaces as an error; fall back to the browser transcript when one exists |
| Long answers | E | Speak the executive summary / first section only, with "read full answer" |

### I. Evaluation and experimentation dashboard

| item | approach | R/E/N |
|---|---|---|
| Datasets | `eval_datasets` + `eval_items` (question, lang, collection, relevant document ids, optional relevant chunk ids, reference answer, **label source: `gold-msmarco` / `human` / `llm-proposed`**). Seed with the existing MS MARCO gold and a hand-written business set | N + R |
| Retrieval metrics | Recall@k, Precision@k, MRR@k, nDCG@k at document level (chunk level when chunk labels exist) | E (`evaluate.py` logic moved into `setu/evaluation/`) |
| Answer metrics | Token recall vs reference (existing); **faithfulness** = groundedness coverage from `CitationVerifier` (automated heuristic, labelled as such); citation correctness = fraction of citations whose passage supports the claim; context relevance = fraction of assembled passages from labelled-relevant documents | E + N |
| LLM-as-judge (optional) | Answer relevance judged by the LLM when a key exists, stored as `judge: llm` and shown separately from human-reviewed scores. Never presented as ground truth | N |
| Latency | Retrieval and end-to-end per run, from the run's own traces | R (`bench.py` logic) |
| Ablations | Retrieval ablations without rebuilding (`retrieval_mode` dense/sparse/hybrid, rerank on/off, top-k sweep) via request knobs; chunking ablations rebuild a temporary index (existing `ablate.py`) and run as a background job | E |
| Red team | Existing 40 cases + new families: injection inside documents, contradictory documents, misleading passages, unsupported numbers, missing evidence, irrelevant questions | E (`redteam.py`) |
| Storage + export | `eval_runs`, `eval_results`; `GET /eval/runs/{id}/export?format=json|csv` | N |

### J. Business analysis capabilities

| tool | output | evidence rule |
|---|---|---|
| SWOT | 4 lists of claims | Each claim cites passages; `interpretation` allowed, labelled |
| PESTLE | 6 lists | same |
| Porter's Five Forces | 5 forces × (rating, rationale claims) | rating is `interpretation`; rationale claims cite evidence |
| Financial performance | Table of metrics × periods from cited passages; derived ratios as `calculation` | Every figure cites a passage; calculations done in Python |
| Risk analysis | Risks × (likelihood, impact, evidence) | likelihood/impact are `interpretation` |
| Competitor comparison | Table entity × dimension | Missing cells show "not in sources", never filled by the model |
| Strategic recommendations | Recommendations linked to the findings they rely on | `interpretation`, linked to cited findings |
| Scenario analysis ("opex +15%") | Base figures from sources (cited) → deterministic calculation → result, with any missing input requested from the user or marked `assumption` | Calculator in Python; the LLM only maps the question to an operation |

All tools are registered as harness tools (`ToolRegistry`) with Pydantic input/output models, so
they get retries, tracing and fallbacks for free. They require an LLM key; without one the endpoint
returns the evidence grouped by framework heading via keyword/semantic matching, clearly labelled
as unstructured evidence, not as an analysis.

---

## 5. API plan

Existing routes stay with the same request/response shapes (new fields are optional). All new
routes return Pydantic response models so the OpenAPI schema can generate frontend types.

| method | path | purpose |
|---|---|---|
| GET/POST | `/collections` | list (paginated) / create |
| GET/PATCH/DELETE | `/collections/{cid}` | detail / rename / delete (index files + rows + Qdrant collection) |
| POST | `/collections/{cid}/rebuild` | rebuild index (job) |
| GET | `/collections/{cid}/documents` | paginated, filter by status |
| POST | `/collections/{cid}/documents` | multipart upload, multiple files → jobs |
| POST | `/collections/{cid}/urls` | import URLs → jobs |
| GET/DELETE | `/documents/{did}` | metadata / delete |
| GET | `/documents/{did}/chunks` | paginated chunks with layer, page, section, hash |
| POST | `/documents/{did}/reindex` | re-extract + rebuild (job) |
| GET | `/jobs`, `/jobs/{jid}` | ingestion status |
| POST | `/ask` | **extended**: `conversation_id`, `collection_ids`, `style`, `retrieval {top_k, mode, rerank, filters}`, `trace_level` |
| GET/POST | `/conversations` | list / create |
| GET/PATCH/DELETE | `/conversations/{id}` | detail with messages / rename / delete |
| POST | `/analysis` | `{framework, question, collection_ids, entities}` → structured analysis |
| POST | `/scenario` | `{question, collection_ids, assumptions}` → calculation with provenance |
| GET | `/traces` | recent traces for the session (paginated) |
| GET | `/trace/{turn_id}` | **extended**: served from the DB, `full` detail when recorded |
| GET/POST | `/eval/datasets`, `/eval/datasets/{id}/items` | datasets |
| POST/GET | `/eval/runs`, `/eval/runs/{id}` | start a run (job) / results |
| GET | `/eval/runs/{id}/export?format=json\|csv` | export |
| GET | `/dashboard/summary` | aggregate counts for the overview |
| GET | `/settings` | effective, non-secret configuration (read-only) |

`/metrics`, `/tools`, `/health`, `/config`, `/samples`, `/questions`, `/voice`, `/stt`, `/tts`,
`/stt/realtime-token`, `/prefetch` and `WS /stream` keep their contracts (with the validation
fixes from §2.3).

---

## 6. Database schema

SQLite now, PostgreSQL-compatible types. All ids are UUIDv4 strings except the legacy sample.

```
collections      id PK, name UNIQUE, description, kind ('builtin'|'user'|'synthetic'),
                 embed_model, index_version, status ('ready'|'rebuilding'|'error'),
                 doc_count, chunk_count, settings_json, created_at, updated_at

documents        id PK, collection_id FK→collections ON DELETE CASCADE,
                 source_type ('upload'|'url'|'msmarco'), filename, source_url, domain,
                 title, mime_type, byte_size, content_hash (sha256 of raw bytes), lang,
                 page_count, published_at, retrieved_at, status, error, synthetic BOOL,
                 meta_json, created_at, updated_at
                 UNIQUE(collection_id, content_hash)            -- dedupe re-uploads

chunks           id PK (= lattice chunk_id), document_id FK ON DELETE CASCADE, collection_id,
                 section_id, layer, chunk_index, page_start, page_end, section_heading,
                 char_start, char_end, parent_id, lang, text, content_hash, embed_model,
                 injection_flag BOOL, created_at
                 INDEX(collection_id, document_id), INDEX(content_hash)

jobs             id PK, kind ('ingest'|'reindex'|'rebuild'|'eval'), collection_id, document_id,
                 state ('uploaded'|'extracting'|'chunking'|'embedding'|'indexing'|'completed'|'failed'),
                 progress REAL, message, error, created_at, started_at, finished_at

embedding_cache  model, text_hash, dim, vector BLOB, PRIMARY KEY(model, text_hash)

conversations    id PK, session_id, title, collection_ids_json, created_at, updated_at
messages         id PK, conversation_id FK ON DELETE CASCADE, role ('user'|'assistant'),
                 content, resolved_query, style, lang, turn_id, citations_json,
                 transcript_json, created_at

traces           turn_id PK, session_id, conversation_id, status, style, pipeline_ms,
                 level ('basic'|'full'), result_json, created_at      -- retention job prunes

eval_datasets    id PK, name, description, collection_ids_json, created_at
eval_items       id PK, dataset_id FK, question, lang, relevant_doc_ids_json,
                 relevant_chunk_ids_json, reference_answer, label_source, reviewed_by, created_at
eval_runs        id PK, dataset_id FK, config_json, status, metrics_json, started_at, finished_at
eval_results     id PK, run_id FK ON DELETE CASCADE, item_id FK, turn_id, metrics_json,
                 judge ('automated'|'llm'|'human')
```

Vectors live in the per-collection index files (or Qdrant), not in SQLite; `chunks` is the metadata
of record and `chunks.jsonl` stays the index's own row map.

---

## 7. Phasing, risks and open decisions

### 7.1 Recommended build order

The steps were written frontend-first, but the dashboard, knowledge base and explorer need backend
data that doesn't exist yet. Building the data layer first avoids writing pages against mock data
(which the steps forbid anyway).

| phase | contents | exit check |
|---|---|---|
| 1. Foundations | DB + migrations, repositories, `CollectionRegistry` with `msmarco-sample`, embedding cache, defect fixes D1–D12, run CPU work in a threadpool | All 133 existing tests pass; `bench.py` and `redteam.py` gates pass unchanged |
| 2. Ingestion (step 3) | Extraction, segmentation, jobs, URL safety, collection APIs, business source manifest | Ingestion, deletion, persistence and isolation tests |
| 3. Reasoning (step 4) | Query understanding, conversations, styles, context assembly, citation verifier, contradictions, analysis tools, scenario calculator | Multi-doc, follow-up, citation and hallucination tests |
| 4. Frontend (step 2) | Router, query layer, generated types, all seven pages | `tsc`, build, manual walkthrough of every page against a live backend |
| 5. Observability + eval lab (step 5) | `trace_level=full`, trace persistence, evaluation service, datasets, ablations, exports | Stored runs reproduce from the CLI |
| 6. Production readiness (step 6) | Auth/rate limits, upload limits, Docker volumes, docs, API integration tests | Full test + build + bench + redteam run, reported as passed/failed |

### 7.2 Risks

| risk | mitigation |
|---|---|
| Quick-mode latency regresses (filters, threadpool hop, extra fields) | `bench.py` gate after every phase; `trace_level=basic` default; filters compile to masks once per request |
| Collection rebuilds get slow as collections grow | Embedding cache; measure rebuild time per 1k chunks; document the ~50k-chunk ceiling and the segmented-index follow-up |
| Without an LLM key most of steps 4 and J are fallbacks | Labelled extractive fallbacks; decision O1 |
| Topicality thresholds tuned on MS MARCO misfire on business PDFs | Per-collection thresholds, calibrated with the evaluation lab before demoing |
| PDF extraction quality varies (tables, columns, scans) | Report extraction warnings per document; OCR explicitly out of scope |
| Scope: this is several weeks of work for one engineer | Phase gates; each phase is usable on its own |

### 7.3 Open decisions (need your input)

| # | decision | options | recommendation |
|---|---|---|---|
| O1 | **LLM provider and key.** Detailed/comparative/critical/executive styles, analysis tools, contradiction explanations and follow-up rewriting all need one | Groq (current default, free tier, `groq/compound-mini`), OpenAI, another OpenAI-compatible endpoint, or no key (extractive fallbacks only) | Groq with a stronger model (e.g. a Llama 3.x 70B variant) for structured output; I'll verify the current model list when implementing |
| O2 | **Access control** | (a) none, demo only; (b) one `SETU_API_TOKEN` for mutating/paid routes + anonymous browser session for conversation ownership; (c) real user accounts | (b). (c) is a separate project |
| O3 | **Default theme** | The README calls light-by-default a deliberate choice; step 2 asks for dark | Dark default as step 2 asks, light kept as an option |
| O4 | **Third-party documents** | Commit PDFs to the repo vs. commit a URL manifest and fetch at setup | Manifest + fetch (licensing) |
| O5 | **Build order** | Your order (frontend first) vs. §7.1 (data layer first) | §7.1 |
| O6 | **Runtime settings page** | Read-only effective config + per-request knobs vs. editable server-wide config | Read-only + per-request knobs; editing server-wide config at runtime is a security and consistency risk |
