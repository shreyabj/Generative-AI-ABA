# SETU 2.0: Research, Knowledge and Evaluation

SETU 2.0 extends the v1 voice RAG engine. It does not replace it. `POST /api/v1/ask`, the
chunk lattice, hybrid retrieval, RRF, late-interaction rerank, the seven guardrails and the
200 ms Quick path are all unchanged in behaviour, and the v1 latency benchmark and red team
still run against them. The audit and design rationale are in [SETU_2_IMPLEMENTATION_PLAN.md](SETU_2_IMPLEMENTATION_PLAN.md).

## Run it

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[ingest,dev]"
cd server; ..\.venv\Scripts\python -m setu.cli build --sample; cd ..   # sample index (~2 min)
cd web; npm install; npm run build; cd ..                              # UI
copy .env.example .env   # set GROQ_API_KEY (needed for the 4 non-Quick styles)
cd server; ..\.venv\Scripts\python -m setu.cli serve                   # http://localhost:8000
# second terminal, while the server runs:
cd server; ..\.venv\Scripts\python -m setu.cli seed                    # demo collections + eval dataset
```

## What is new

| area | where | notes |
|---|---|---|
| Metadata DB | `server/setu/storage/` | SQLite through SQLAlchemy Core, behind repositories. Set `SETU_DATABASE_URL` for PostgreSQL. Tables: collections, documents, chunks, jobs, embedding_cache, conversations, messages, traces, eval_datasets, eval_items, eval_runs, eval_results. |
| Ingestion | `server/setu/knowledge/` | PDF (pypdf), DOCX, TXT, MD, CSV, URLs (trafilatura). Flow: validate by magic bytes → store raw file → extract → clean → segment into sections → lattice → embed (cached by content hash) → vector + BM25 → persist. Runs on one background worker. Job states: uploaded, extracting, chunking, embedding, indexing, completed, failed. |
| Collections | `knowledge/service.py`, `registry.py` | One `CorpusIndex` per collection, rebuilt and swapped atomically when documents change. Collections are physically isolated. The v1 sample is the read-only built-in collection `col_msmarco_sample`. |
| URL safety | `knowledge/fetch.py` | http/https only, ports 80/443 only. Private, loopback, link-local and metadata IPs are blocked, both on resolution and on the connected peer. Each redirect hop is re-validated. robots.txt is honoured. Content types are allow-listed and downloads are size-capped. |
| Research pipeline | `server/setu/research/` | Query understanding: intent, follow-up resolution (rules first, LLM rewrite when a key is set), multi-part and comparison decomposition. Then multi-collection retrieval with per-sub-query quotas, context assembly, five styles plus eight business frameworks, claim-level citation verification, contradiction detection and a deterministic calculator. |
| Evaluation lab | `server/setu/evaluation/` | Recall/Precision/hit/nDCG@k and MRR@10 at document level, plus automated answer heuristics. Ablation presets: dense, sparse, hybrid, rerank on/off, parent expansion, pseudo-relevance feedback. A red-team suite runs against synthetic fixtures. Results export as JSON and CSV. |
| Security | `server/setu/security.py` | Optional `SETU_API_TOKEN` on mutating and paid routes. `X-SETU-Session` scopes conversations and traces. Per-IP rate limits, upload, audio and TTS size caps, a `voice_id` allow-list, CORS without credentials for `*`, and nosniff/frame headers. |
| UI | `web/src/v2/` | Dark, hash-routed shell with Overview, Research, Knowledge Base, RAG Explorer, Evaluation Lab, Voice (the v1 voice UI, kept intact) and Settings. It adds no new npm dependencies; charts are plain SVG/CSS. |

## How each answer mode behaves (`POST /api/v1/research/ask`)

| style | engine | output |
|---|---|---|
| quick | Extractive (no LLM) | Verbatim spans, checked by the v1 groundedness gate and the trust floor |
| detailed | LLM, `openai/gpt-oss-120b` | 3–5 sections with multiple sources |
| comparative | LLM | Required table, "not in sources" for unsupported cells, and per-entity sub-queries filtered to each entity's documents |
| critical | LLM | Claim, support, counter-evidence, limitations, assessment |
| executive | LLM | Summary, findings, implications and linked recommendations |

`framework` adds swot, pestle, porter, financial, risk, competitor, recommendations or scenario.

Every claim carries a `kind` (fact, calculation, interpretation or assumption). The verifier
scores each claim only against the passages it cites, using max(lexical containment,
semantic max-sim). Citations that don't support their claim are dropped. A fact claim left
without support, or carrying a number found in no passage, is removed and listed under
`removed_claims`. Numbers in `calculations` are computed by Python from cited inputs; the
model only names the operation.

Without an LLM key, the non-Quick styles fall back to an extractive answer per sub-query, and
the response says so.

## Verified results (2026-11-01, this machine)

- Tests: `pytest` reports **196 passed** (179 v1 + 17 new for ingestion, isolation,
  persistence, SSRF, segmentation, the verifier, follow-ups, the calculator, contradictions
  and metrics). All are offline, with no LLM calls.
- v1 latency: `scripts/bench.py --n 100` gives P50 4.5 ms and P100 15.6 ms; the 200 ms SLO holds.
- v1 red team: `scripts/redteam.py` passes **40/40**. One keyboard-mash case needed a
  sanity-gate fix on the 700-passage sample index.
- v2 red team (research pipeline, synthetic fixtures): **8/8**. The cases cover question
  injection, a harmful request, an irrelevant question, missing evidence, injection inside a
  document, contradictory sources, a misleading passage and an unsupported number.
- Stored evaluation runs (document-level recall@5, hybrid / dense / sparse / no-rerank):
  - **business-curated**, 8 items: 1.0 for all four presets. The set is too small and too
    easy to separate the configurations. Treat it as a smoke test.
  - **msmarco-sample-gold**: 0.075, 0.048, 0.076 and 0.067. Hybrid MRR@10 is 0.44. Recall
    is low because each question has several relevant passages in 10 languages, so recall@5
    can't reach 1.
- Business collection: 9 of 10 sources ingested, 1,242 chunks. The HCLTech page timed out.
  Infosys, TCS and PIB are excluded because their robots.txt disallows fetching, and IMF
  because it returns HTTP 403. The manifest lists all of these with reasons.

## Known limitations

- Collection rebuild time grows with collection size, because each change re-chunks and
  refits BM25; embeddings are cached. Past about 50k chunks, a segmented index is needed.
- Scanned PDFs are not supported (no OCR). Table extraction from PDFs is plain text.
- The answer metrics (faithfulness, citation precision, answer relevance) are automated
  heuristics. No human-reviewed labels exist beyond MS MARCO's own `is_selected` labels.
  The curated business set was written by the maintainers.
- Rate limiting and the prefetch cache are per process. Use Redis if you run several workers.
- Voice output uses ElevenLabs only when `ELEVENLABS_API_KEY` is set. Without it, the browser
  speech APIs are used (Chrome or Edge recommended).
- The session id is an ownership boundary for a demo, not authentication. Set
  `SETU_API_TOKEN` before exposing the server.
