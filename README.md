<div align="center">

# SETU

### **S**peech → **E**vidence → **T**ruth → **U**tterance

**A voice-enabled RAG engine for 15 Indian languages. Ask out loud, get a verified, cited answer in 15 ms.**

*सेतु — "bridge". A bridge from voice to grounded knowledge.*

[![SLO](https://img.shields.io/badge/pipeline_P50-15_ms-C6F24E?style=flat-square&labelColor=06070A)](docs/LATENCY.md)
[![SLO](https://img.shields.io/badge/SLO-200_ms-8B7BFF?style=flat-square&labelColor=06070A)](docs/LATENCY.md)
[![Corpus](https://img.shields.io/badge/corpus-MS_MARCO--XI-4CE0D2?style=flat-square&labelColor=06070A)](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
[![Languages](https://img.shields.io/badge/languages-15-FFB347?style=flat-square&labelColor=06070A)](#languages)
[![Guardrails](https://img.shields.io/badge/guardrails-7_gates-FF6B6B?style=flat-square&labelColor=06070A)](docs/GUARDRAILS.md)
[![License](https://img.shields.io/badge/license-MIT-9BA3B4?style=flat-square&labelColor=06070A)](LICENSE)



</div>

---

> **SETU 2.0** adds knowledge collections (PDF/DOCX/TXT/MD/CSV/URL ingestion), a research
> workspace with five verified answer styles and business frameworks, a RAG pipeline explorer
> and an evaluation lab. See **[docs/SETU_2.md](docs/SETU_2.md)** for setup and the measured results.

## What it is

Speak a question in Hindi, Tamil, Bengali, English or eleven other Indian languages. SETU
transcribes it, retrieves evidence from the MS MARCO-XI corpus, composes an answer with inline
citations, **verifies that answer against its own evidence**, and speaks it back.

Measured over **300 distinct queries**: **P50 15 ms · P70 16 ms · P100 24 ms**, with
**285/285 turns inside the 200 ms budget**. Every number in this README comes from a script in
`scripts/` that you can run yourself, and the raw reports are committed in `docs/*.json`.

> **The one-line summary of the design:** SETU doesn't pick a chunk size, doesn't trust a
> generative model to stay grounded, and doesn't treat its latency budget as a hope.

---

## Try it in 60 seconds

No API keys, no dataset download. The repo ships a sample corpus.

```bash
git clone <your-repo-url> && cd setu-voice-rag

python -m venv .venv && .venv\Scripts\activate     # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS / Linux

pip install -e ".[ingest]"
python -m setu.cli build --sample                   # index the sample corpus (~20 s)

cd web && npm install && npm run build && cd ..     # build the console
cd server && python -m setu.cli serve               # → http://localhost:8000
```

Type a question, or hold **Space** to talk. Everything works without keys except voice input and
spoken output — for those, copy `.env.example` to `.env` and add `ELEVENLABS_API_KEY`.

<details>
<summary><b>Run on the real corpus (recommended)</b></summary>

```bash
cd server
python -m setu.cli ingest --langs hi,ta,bn --rows 220
```

Downloads one ~460 MB validation shard per language (cached by `huggingface_hub`, resumable),
then builds the lattice and the index. Produces ~8.7k documents and ~43k lattice nodes, plus a
gold-labelled evaluation set at `data/corpus/queries.jsonl`.

`--langs` accepts any of `as bn gu hi kn ml mr ne or pa sa ta te ur`.

</details>

<details>
<summary><b>Docker</b></summary>

```bash
docker compose up --build              # → http://localhost:8000
docker compose --profile qdrant up     # with Qdrant as the vector database
```

The image bakes in the encoder weights and builds the sample index at build time, so a fresh
container is warm on its first request.

</details>

---

## The pipeline

```
🎙 voice ─▶ ElevenLabs Scribe ─▶ ┏━━━━━━━━━━━━ HARNESS (200 ms budget) ━━━━━━━━━━━━┓ ─▶ 🔊 answer
                                ┃                                                 ┃
                                ┃  normalise ▸ admission ▸ route ▸ embed          ┃
                                ┃    ▸ safety ▸ retrieve ▸ topicality             ┃
                                ┃    ▸ answer ▸ verify ▸ trust                    ┃
                                ┃                                                 ┃
                                ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                        │                   │
                            Polymorphic Chunk Lattice   7 guardrail gates
                            (4 views of every doc)      (knows when NOT to answer)
```

---

## What makes it different

### 1 · The Polymorphic Chunk Lattice — four views, chosen per query

Every chunk size is a bet about the shape of future questions. Small chunks win precision and lose
context; large chunks do the reverse. You can't resolve that at ingest time, because the question
hasn't arrived yet.

So SETU doesn't choose. It builds **four simultaneous views** of every document and lets a
**0.3 ms router** pick between them per query:

| layer | granularity | built by | best for |
|---|---|---|---|
| **L0 atomic** | one proposition | forward accretion + clause fission | factoid, numeric — *the answer span itself* |
| **L1 semantic** | coherence block | **embedding-drift boundaries**, per-document threshold | descriptive, causal |
| **L2 window** | fixed budget | **overlap scaled to local cohesion** (14–50 %) | coverage guarantee, robust to bad punctuation |
| **L3 document** | whole passage | **metadata-aware embedding** (`[hi \| NUMERIC \| rank 2/10]`) | definitions, broad recall |

Plus:

- **Parent links** — retrieval runs on the fine layers where signal is concentrated, then expands
  through `parent_id` to the containing coherence block. Small-to-search, big-to-read.
- **Per-document drift thresholds** — `μ + 0.45σ` of *that document's own* drift distribution, not a
  global constant. A tight passage and a rambling one need different sensitivity.
- **Honest degeneration** — layers are emitted coarse→fine and byte-identical nodes are dropped, so
  lattice depth *adapts to document length*. On the real corpus: 25 docs stratify to 4 layers, 123
  to 2, 25 to 1. Dedup rate **30.9 %**, and we report it rather than hiding it.
- **Shared sentence embeddings** — computed once per document and reused by the semantic chunker,
  the window chunker, the reranker **and** the answer composer. Four views cost far less than 4×.

📖 **[docs/CHUNKING.md](docs/CHUNKING.md)** — the full argument, with an ablation against a
`window-only-naive` baseline.

### 2 · Retrieval that fuses on rank, not on scale

A cosine of `0.81` and a BM25 score of `41.0` are not comparable. Adding them is a common bug where
whichever scale is larger silently wins.

- **Per-layer dense search** using precomputed boolean masks — a layer restriction is a matrix view,
  not a post-filter loop.
- **BM25 as a real inverted index** — document-side weights precomputed into a CSC sparse matrix, so
  a query touches only its own posting lists. `rank_bm25` would cost 40–80 ms here; this costs
  **0.1 ms**.
- **Reciprocal Rank Fusion** + normalised score blending + layer and cohesion priors.
- **Late-interaction reranking** — `maxsim(q, chunk) = maxⱼ cos(q, sentenceⱼ)` over sentence vectors
  cached at ingest. This is the sharp signal a pooled chunk embedding loses: a block whose *third*
  sentence answers the question has a mediocre mean and an excellent max. A cross-encoder would cost
  80–150 ms; this costs **0.4 ms**.
- **Span diversification** — five citations are five *different* pieces of evidence, not one passage
  sliced four ways.
- **Cross-lingual by construction** — a Hindi question retrieves the paired English passage, and the
  composer then prefers spans in the asker's language.

### 3 · An answer that cannot hallucinate a token

MS MARCO is an extractive machine-reading corpus: its passages were collected *because they contain
the answer*, and it ships a human `is_selected` label marking which one does.

So the **Reflex path** (default) is a span selector, not a paraphraser. Sentences are scored on
semantic similarity, lexical containment, retrieval prior, query-shape affinity, language match and
position; selected greedily with a redundancy veto; and stitched with inline citations.

Two consequences:

- **~0.4 ms** instead of 100–250 ms time-to-first-token.
- **Every token is copied from retrieved evidence.** Groundedness is structural, not hoped for.

The **Deliberate path** (`GROQ_API_KEY`) exists for answers that must fuse passages or be rephrased.
It runs under a strict JSON contract with mandatory citations and a first-class `abstain` field, with
retries, a circuit breaker, and a declared fallback to the reflex path. It is **measured separately
and excluded from the 200 ms SLO**, because nobody can bound somebody else's inference endpoint.

### 4 · Guardrails that know when *to* answer too

Seven gates across four stages. Every gate returns **a score and the threshold it had to clear** —
never a bare boolean, because "blocked" with no number is not something you can tune or test.

| stage | gate | catches |
|---|---|---|
| admission | `input.sanity` | non-questions, spam, degenerate repetition |
| admission | `input.injection` | 14 patterns × 6 attack families, incl. Devanagari |
| safety | `input.safety` | **7 semantic harm centroids** — cross-lingual, paraphrase-robust |
| retrieval | `retrieval.topicality` | out-of-corpus, via absolute score **+ score margin** + lexical anchoring + **temporal deixis** |
| output | `output.groundedness` | per-claim attribution, **numeric hallucination**, invented entities |
| output | `output.pii` | email/phone/Aadhaar/PAN/card — Verhoeff + Luhn validated |
| output | `output.trust-floor` | composite < 45/100 withheld even when no single gate blocked |

Three things most safety layers get wrong, that SETU handles:

- **Multilingual coverage.** Keyword blocklists leave 14 languages unguarded. Harm categories are
  seed phrases embedded into centroids **in the same multilingual space the retriever uses** — so a
  request in Hindi or Tamil is caught by seeds written in English, and paraphrasing doesn't help the
  attacker, because invariance to paraphrase is what an encoder is for.
- **Not breaking the product.** MS MARCO contains medical, historical and security passages. A gate
  that refuses *"what are the symptoms of an opioid overdose"* protects nothing. A **benign centroid
  set** of 12 deliberately-adjacent seeds means a query is only blocked when it's closer to harm than
  to benign. 12 red-team cases exist purely to catch this false-positive direction.
- **Score margin, not just score.** Top-1 minus top-2 must clear a floor. A *flat* score distribution
  means the retriever found no *distinguishing* evidence — which is what "not in the corpus" looks
  like from the inside, and it's the case an absolute threshold misses.
- **Failures retrieval cannot see.** *"What will the stock market do tomorrow?"* retrieves
  beautifully — "stock market" is an ordinary corpus topic — and is unanswerable, because the corpus
  is a fixed snapshot with no timestamps. No retrieval score can detect that: the evidence is
  topically right and temporally meaningless. A **temporal-deixis** check declines the whole class.
- **Max-over-seeds, not just centroid.** Averaging four seed phrases into one vector dilutes them: a
  query paraphrasing *one* seed almost exactly can sit below the centroid threshold because the
  centroid also points at three unrelated phrasings. SETU keeps both signals and takes the stronger.
  Both of these were found by the red-team suite, not by inspection.

Self-harm gets a **safe completion with India's Tele-MANAS (14416) and KIRAN helplines**, not a bare
refusal. Every refusal and abstention is **localised into all 15 corpus languages** — a guardrail
that refuses in English to someone who asked in Tamil has failed at the part the user experiences.

📖 **[docs/GUARDRAILS.md](docs/GUARDRAILS.md)**

### 5 · A real harness: the deadline is a resource

Nine capabilities registered as **typed tools** with Pydantic input *and* output models. Calling one
validates both directions, opens a trace span, applies retry/timeout/circuit-breaker policy, and
returns a structured `Outcome` that never raises at the call site.

**The budget governor** is the part most pipelines lack. Before any *optional* stage:

```python
if budget.can_afford(cost_ms, reserve_ms=still_to_come):
    ...run...
else:
    budget.skip("rerank.late-interaction", cost_ms)   # recorded, returned to the client
```

The pipeline is **anytime**: it degrades quality, never the deadline — and it tells you which stage
it dropped. There is no code path that blows the SLO to finish a refinement.

**Declared fallback chains**, every one surfaced in `warnings`:

| primary | falls back to |
|---|---|
| Deliberate (LLM) | Reflex extractive |
| ElevenLabs Scribe | browser speech recognition |
| Qdrant | in-process exact store |
| fastembed ONNX | sentence-transformers |

A turn degrades; it does not crash. And whatever degraded is *named* — silent fallback is worse than
failure, because you can't debug what you can't see.

📖 **[docs/HARNESS.md](docs/HARNESS.md)**

### 6 · Speculative prefetch — spending the STT latency

Voice adds hundreds of milliseconds of dead time while the retrieval side sits idle.

While you're still speaking, the browser streams interim transcripts over a websocket and the server
**embeds and retrieves against the partial**. By the time the final Scribe transcript arrives,
retrieval is frequently already done — the turn reports `speculative_hit: true` and drops to ~2 ms.

Reuse requires an **exact** normalised match. Approximate reuse would be faster and occasionally
answer a question you didn't ask; not a trade worth making. A miss costs nothing, because it happened
during speech. The benchmark **disables it entirely**, so reported numbers are the honest cost of
computing a turn from scratch.

---

## Latency

Three distributions, reported separately, because collapsing them would hide which part is bounded:

| metric | boundary | in the SLO? |
|---|---|---|
| **`pipeline_ms`** | transcript in → **verified** answer out | ✅ **this is the SLO** |
| `stt_ms` | ElevenLabs Scribe network round trip | ❌ third-party endpoint |
| `e2e_ms` | wall clock including STT | ❌ reported for honesty |

300 distinct queries from the corpus's own gold set, warm process, speculative prefetch disabled,
paced arrival — measured on the shipped 10-language index (2,361 passages / 10,841 pieces):

```
  P50      14.8 ms  [OK] ###
  P70      15.5 ms  [OK] ###
  P90      17.5 ms  [OK] ####
  P95      18.6 ms  [OK] ####
  P99      21.7 ms  [OK] ####
  P100     23.7 ms  [OK] #####
  SLO     200 ms         ─────────────────────────────────────────────────
  mean     14.5 ms  ·  min 7.4 ms  ·  stdev 2.8 ms
  compliance     285/285 inside (100.00%), 0 breaches
```

Where the time goes:

| stage | P50 ms | P100 ms | share |
|---|---|---|---|
| `retrieve` (dense + BM25 + fuse + expand + rerank) | 7.0 | 18.6 | ~47 % |
| ├─ dense search across lattice layers | 5.9 | 15.8 | |
| ├─ late-interaction rerank | 0.38 | 1.90 | |
| ├─ RRF fusion | 0.27 | 3.34 | |
| └─ BM25 posting lists | 0.18 | 0.68 | |
| `embed_query` (ONNX, 384-d) | 6.9 | 15.2 | ~46 % |
| `answer_reflex` | 0.31 | 1.49 | ~2 % |
| `guard` × 4 stages | 0.21 | 1.40 | ~1 % |
| `route` + `normalise` | 0.18 | 1.50 | ~1 % |

One unavoidable model call and no second one — which is why there is no cross-encoder reranker and no
LLM on the default path. All seven guardrails together cost **0.21 ms**.

<details>
<summary><b>⚠️ Read this before quoting the numbers: the host dominates, and it is not stable</b></summary>

Three runs of essentially the same pipeline on the same laptop, hours apart:

| | P50 | P70 | P100 | inside 200 ms |
|---|---|---|---|---|
| cool, 1.8k-piece index | 70.5 ms | 80.1 ms | 164.8 ms | 300/300 (100 %) |
| heat-soaked, 1.8k-piece index | 159.3 ms | 178.2 ms | 410.7 ms | 258/300 (86 %) |
| **cool, 10.8k-piece index (shipped)** | **14.8 ms** | **15.5 ms** | **23.7 ms** | **285/285 (100 %)** |

Note the third row is on an index **six times larger** and is the *fastest* — because retrieval is a
matmul over resident data, and index size barely matters at this scale. What moves the numbers by 10×
is the machine's sustained single-core clock. The same decay appeared during ingest, where embedding
throughput fell from 81/s to 11/s over a few minutes and recovered to 28/s later.

P50 and P70 were inside 200 ms in **all three** runs. P100 exceeded it only on the thermally
saturated one. All three are published because quoting only the best would be the easiest
misrepresentation available here.

</details>

<details>
<summary><b>A measurement finding worth reading: thread pools and tail latency</b></summary>

The encoder's thread count is set to **2**, and that is not a guess. ONNX Runtime's intra-op pool
spins briefly, then parks. Measured on a 16-core laptop, encoding one short query:

| threads | back-to-back | with a 40 ms gap between calls |
|---|---|---|
| default (one per core) | **5.8 ms** | **202.3 ms** |
| 1 | 53.0 ms | 49.7 ms |
| **2** | 39.9 ms | **42.8 ms** |

A large pool is superb while it stays hot and catastrophic once it has parked — and *real traffic
always has gaps*, because between two queries the pipeline is busy doing retrieval, verification
and I/O. Benchmarking in a tight loop hides this completely and reports the 5.8 ms figure, which no
user would ever experience. The first version of this benchmark did exactly that, and the gap between
it and the end-to-end number is what exposed the problem.

So SETU picks the configuration with the best **tail**, not the best median. `scripts/bench.py`
defaults to a paced arrival pattern for the same reason, and `--pace-ms 0` is available as a
separate throughput stress test rather than being passed off as latency.

</details>

<details>
<summary><b>A measurement finding worth reading: thread pools and tail latency</b></summary>

The encoder's thread count is set to **2**, and that is not a guess. ONNX Runtime's intra-op pool
spins briefly, then parks. Measured on a 16-core laptop, encoding one short query:

| threads | back-to-back | with a 40 ms gap between calls |
|---|---|---|
| default (one per core) | **5.8 ms** | **202.3 ms** |
| 1 | 53.0 ms | 49.7 ms |
| **2** | 39.9 ms | **42.8 ms** |

A large pool is superb while it stays hot and catastrophic once it has parked — and *real traffic
always has gaps*, because between two queries the pipeline is busy doing retrieval, verification
and I/O. Benchmarking in a tight loop hides this completely and reports the 5.8 ms figure, which no
user would ever experience.

So SETU picks the configuration with the best **tail**, not the best median. `scripts/bench.py`
defaults to a paced arrival pattern for the same reason, and `--pace-ms 0` is available as a
separate throughput stress test rather than being passed off as latency.

</details>

**Reproduce it:**

```bash
python scripts/bench.py --n 300                    # → docs/latency.json
python scripts/bench.py --n 300 --mode deliberate  # the LLM path, separately
```

Methodology — **distinct queries** every turn from the corpus's own gold set (repeating one query
hits the encoder LRU and reports a real number for an unreal workload), **warm process** with the
first 15 turns discarded, **prefetch disabled**, **paced arrival** (see the note above), and
**P100 as a true maximum**. `bench.py` exits non-zero on any SLO breach, so it gates CI.

Latency is also **always on**: every turn served — including every click in the UI — lands in a ring
buffer, and `GET /api/v1/metrics` reports the live distribution. The header badge is that endpoint.

📖 **[docs/LATENCY.md](docs/LATENCY.md)** — including the honest caveats.

---

## Quality — measured, not asserted

MS MARCO-XI's `is_selected` labels and gold `Answer` strings give supervision for free, so there's no
excuse for not measuring:

```bash
python scripts/evaluate.py --n 200     # recall@k, MRR@10, gold-answer token recall
python scripts/redteam.py --verbose    # 40 adversarial + benign cases, confusion matrix
python scripts/ablate.py               # what each chunking layer actually contributes
```

**Guardrails — `docs/redteam.json`:**

| | count | |
|---|---|---|
| adversarial cases stopped | **28 / 28** | injection, weapons, self-harm, cyber, drugs, fraud, hate, out-of-corpus, garbage |
| **false negatives** | **0** | nothing unsafe got through |
| **false positives** | **0** | no legitimate question refused |
| benign abstentions | 8 | passage genuinely absent from the sample corpus — not a guardrail failure |

**Retrieval — `docs/quality.json`**, 72 gold queries on the shipped 10-language index:
recall@1 0.14 · recall@5 0.33 · recall@10 0.43 · MRR@10 0.22 · gold-answer token recall 0.45 ·
answered 57 % · `abstain_wrong` 10.

> ⚠️ **These are weak, and the reason is a deliberate trade-off.** The shipped index spreads 2,361
> passages across ten languages, so each language gets ~236 passages — roughly 24 questions' worth of
> depth. The right passage often simply isn't there, and ten languages of near-identical translated
> content makes the ones that *are* there harder to separate. Widening language coverage at a fixed
> corpus size costs retrieval depth; I chose coverage so that Kannada, Telugu and Malayalam actually
> work, and `abstain_wrong: 10` is the honest price.
>
> `make ingest LANGS=hi,ta,bn ROWS=400` gives far better numbers on fewer languages. They are
> reported unflattering and un-cherry-picked, because a quality section that only appears when it
> looks good is not a quality section.

`evaluate.py` reports something most RAG demos can't: **`abstain_wrong`** — how often gold evidence
*was* retrieved and the system declined anyway. Tightening a guardrail always improves your red-team
score and always costs answer rate; that number is how you know a threshold has gone too far.

---

## The interface

One page, one job: a big microphone button, a text field, and an answer you can check. **Light theme
by default**, dark one click away and remembered — a public question box should read as calm and
approachable, not as an internal developer tool.

What a visitor sees, in order: a headline, the ten languages **in their own scripts** (ಕನ್ನಡ · தமிழ் ·
বাংলা · తెలుగు…), the mic, and three real questions from the library to try. After asking: the answer
with inline citation numbers, then a collapsed *"Where this came from"* section holding the actual
passages.

Deliberately **not** on the page: the chunk-lattice map, router internals, span waterfall, retrieval
score breakdowns, live percentile dashboard and the guardrail verdict table. All of it was accurate
and none of it helped someone who just wants an answer — it made the product look like a debugger.
The pipeline still reports every one of those things on the API (`/api/v1/metrics`, `/api/v1/trace/{id}`,
and every field of `TurnResult`), and `scripts/` measures them, which is where that information
belongs.

What survived from the telemetry, because it is genuinely user-facing:

- **Citation numbers** in the answer — hover one to highlight the passage it came from.
- **"Checked against its sources"** with the source count and the time taken.
- **A plain-language reason** whenever SETU declines: *"It found related passages, but none of them
  actually supported an answer."* The guardrails still run; they just no longer report as a scoreboard.

### Voice output: right voice, right language

This was the biggest fix. One English voice was reading all ten languages, which sounds absurd for
Kannada or Telugu. Synthesis is now routed per language:

| | how it's chosen |
|---|---|
| **Voice** | a different ElevenLabs voice per language, so switching language audibly switches speaker |
| **Model** | `eleven_flash_v2_5` where pronunciation can be pinned, `eleven_multilingual_v2` elsewhere |
| **Pronunciation** | `language_code` is sent when the API accepts it — verified: only `en`, `hi`, `ta` |
| **Local voices win** | if your device has a real voice for that language, the browser uses it, because native prosody beats any cross-lingual model |

Measured on the shipped build: 10 distinct voices across 10 languages, every answer spoken in the
language it was written in.

And it works with **no API key at all** — `window.speechSynthesis` is the fallback for both speaking
and listening, so a fresh clone is never mute.

| | with `ELEVENLABS_API_KEY` | without |
|---|---|---|
| **listening** | ElevenLabs Scribe v2 | browser speech recognition (Chrome/Edge) |
| **speaking** | per-language ElevenLabs voice | installed system voice for that language |

### Language picker

All 15 languages in their own script, searchable, in two honest groups: **Full support** (passages
indexed, answers come back in that language) and **Ask anyway** (understood via cross-lingual search,
answers arrive in English for now). A small speaker badge marks languages your device has a real
voice for, so you know before pressing play.

It replaces a native `<select>`, whose popup the browser rendered with the OS theme — white text on
white, effectively invisible.

Keyboard: **Space** to talk, **/** to type. Reduced-motion support and ARIA labels throughout.

---

## Languages

**Indexed in the shipped build** — English · Hindi · Bengali · Tamil · Telugu · Kannada · Malayalam ·
Marathi · Gujarati · Punjabi

**Also askable** (cross-lingual, answers in English until you index them) — Assamese · Nepali ·
Odia · Sanskrit · Urdu

```bash
make ingest LANGS=hi,ta,bn,kn,te,ml,mr,gu,pa   # what this build uses
make ingest LANGS=as,ne,or,sa,ur               # add the rest
```

One shared multilingual vector space, so cross-lingual retrieval works in both directions — a Kannada
question can retrieve an English passage. Script detection, Unicode tokenisation and sentence
splitting (danda, double danda, Arabic full stop) live in `server/setu/text.py`: five writing systems,
not an English pipeline with extra fonts.

Answering, though, is **language-locked**: the composer only selects spans in the language you asked
in, falling back to any language when the corpus has nothing. Cross-lingual *retrieval* is a feature;
cross-lingual *answering* is a defect, and it is worse in a voice system because the wrong language
gets read aloud.

---

## Project layout

```
setu-voice-rag/
├── server/setu/
│   ├── harness/          contracts · tracing · policy · registry     ← the harness
│   ├── chunking/         atomic · semantic · window · metadata · lattice
│   ├── index/            bm25 · store · qdrant_store · fusion · corpus
│   ├── retrieval/        router · engine · rerank
│   ├── answer/           extractive · llm · compose
│   ├── guardrails/       admission · topicality · groundedness · redaction · stack
│   ├── speech/           elevenlabs (Scribe + Flash)
│   ├── ingest/           msmarco_xi · build
│   ├── text.py           script-aware multilingual text layer
│   ├── embed.py          ONNX encoder, warm + LRU cache
│   ├── telemetry.py      always-on percentile ring buffer
│   ├── pipeline.py       the state machine
│   └── app.py            FastAPI + SPA hosting
├── web/src/              React console (custom CSS design system)
├── scripts/              bench · evaluate · redteam · ablate
├── tests/                chunking · retrieval · guardrails · harness
├── docs/                 ARCHITECTURE · CHUNKING · LATENCY · GUARDRAILS · HARNESS
└── data/sample/          offline corpus so the repo runs with no download
```

---

## Commands

```bash
make help        # all targets
make build       # index the sample corpus (no download)
make ingest      # download MS MARCO-XI shards + build   (LANGS=hi,ta ROWS=220)
make web         # build the frontend
make serve       # run API + UI on :8000
make bench       # P50/P70/P100 → docs/latency.json
make evaluate    # retrieval + answer quality vs gold labels
make redteam     # guardrail suite
make ablate      # chunking ablation
make test lint   # pytest + ruff + tsc
```

Single-question CLI, with the full trace printed:

```bash
cd server && python -m setu.cli ask "how many calories are in a banana"
cd server && python -m setu.cli ask "एक केले में कितनी कैलोरी होती हैं" --lang hi
```

---

## Deploy

One container, one URL. `Dockerfile` builds the React app, bakes in the encoder weights, and serves
everything from FastAPI.

- **Render** — `render.yaml` blueprint, 5 GB persistent disk for the index
- **Fly.io** — `fly.toml`, `min_machines_running = 1` (a suspending machine hands back the warm-up
  cost on every wake, which would make the latency story a lie)
- **Docker Compose** — with an optional Qdrant profile

Set `ELEVENLABS_API_KEY` for voice, `GROQ_API_KEY` for the deliberate path. Neither is needed for
the core pipeline.

> **Security note:** the API ships with no authentication, because it's a public demo of a
> read-only corpus. Before putting it anywhere real, add auth to `/api/v1/*` — `/tts` and `/voice`
> call a paid third-party API and are trivially abusable as an open proxy.

---

## Honest limitations

- **The 200 ms SLO covers the local pipeline**, not the third-party STT call. Both are measured and
  reported; neither is hidden. See [LATENCY.md §1](docs/LATENCY.md).
- **Numbers are host-dependent.** These come from a laptop-class x86 core. A smaller cloud instance
  will be slower — which is exactly what the budget governor is for.
- **Exact search doesn't scale forever.** At 43k nodes a brute-force matmul beats an HNSW round trip
  with zero recall loss. At millions of vectors, switch to Qdrant (`SETU_VECTOR_BACKEND=qdrant`) and
  dense retrieval becomes an approximate ~2–5 ms. The budget has room; the backend is one env var.
- **Extraction has a ceiling.** It cannot synthesise an answer spread across two passages in fluent
  prose. That's what the Deliberate path is for, and why `evaluate.py` reports gold-answer token
  recall rather than claiming a solved problem.
- **The corpus is a slice**, not all 55 GB. Abstention rate is therefore higher than it would be on a
  full ingest — and `evaluate.py` separates *correct* abstentions from *wrong* ones so that's visible
  rather than flattering.
- **Realtime Scribe tokens** (`/api/v1/stt/realtime-token`) are wired but not the primary path; the
  demo uses batch Scribe on the recorded blob plus browser partials for prefetch.

---

## Credits

**Corpus** — [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
(IndicRAGSuite, [arXiv:2506.01615](https://arxiv.org/abs/2506.01615)), derived from Microsoft
MS MARCO.
**Speech** — ElevenLabs Scribe v2 (STT) and Flash v2.5 (TTS).
**Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2`, served as ONNX via
[fastembed](https://github.com/qdrant/fastembed).
**Vector DB** — [Qdrant](https://qdrant.tech).

Licensed MIT (see [LICENSE](LICENSE) for third-party data terms).

<div align="center">



</div>
