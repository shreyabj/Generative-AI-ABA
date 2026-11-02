# Chunking: the Polymorphic Chunk Lattice

> The brief asks for a chunking strategy that is *vast* — not a single naive fixed-size split.
> This document is the argument, the implementation map, and the measured evidence.

## The problem with picking a chunk size

Every chunk size is a bet about the shape of future questions.

- **Small chunks** win on precision. `"A banana contains 89 calories per 100 g."` is the answer
  to a factoid question with nothing around it to dilute the embedding. They lose on context —
  a pronoun without its antecedent, a number without its unit.
- **Large chunks** win on context and lose on precision. A 1000-character passage whose seventh
  sentence answers the question has a pooled embedding dominated by the other six.

You cannot resolve this at ingest, because the right granularity depends on the question, and the
question has not arrived yet. So SETU does not choose. It builds **four views of every document
simultaneously** and chooses at query time, per query, in about 0.3 ms.

```
        L3  DOCUMENT   [hi | NUMERIC | rank 1/10] <whole passage>
             │                                     metadata-aware embedding
        L1  SEMANTIC   ▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓│▓▓▓▓     drift-bounded coherence blocks
             │                                     ↕ L2 WINDOW overlays this scale
        L0  ATOMIC     ▪▪│▪▪▪│▪▪│▪▪▪▪│▪▪│▪▪▪▪▪     one proposition per node
```

---

## L0 — Atomic / proposition
`server/setu/chunking/atomic.py`

One chunk, one assertion. This is what makes extractive answering and claim-level groundedness
verification possible: when the retriever returns an atom, that atom *is* the answer span.

Two behaviours keep it from degenerating into useless fragments:

**Forward accretion.** Sentences below `atomic_min_chars` (45) are glued to the following
sentence rather than indexed alone. `"Yes."` is not a retrievable unit, and indexing it pollutes
the vector space with a node that is weakly similar to everything.

**Clause fission.** Sentences above `atomic_max_chars` (240) are split at discourse boundaries —
semicolons, coordinating conjunctions, and their Devanagari / Bengali / Tamil equivalents — so a
600-character run-on sentence still yields precise atoms. Fragments produced by fission that fall
back below the minimum are re-merged, so fission cannot undo accretion.

## L1 — Semantic coherence (embedding-drift boundaries)
`server/setu/chunking/semantic.py`

Fixed-size splitting cuts wherever the character counter lands, which routinely severs a
definition from its example. This layer asks the encoder where the topic actually changes.

1. Embed every sentence (done once, shared by all four layers — see *Cost* below).
2. Build a drift signal `dᵢ = 1 − cos(sᵢ, sᵢ₊₁)` over adjacent sentences.
3. Smooth it with a width-3 kernel, so one parenthetical aside is not a topic shift.
4. Cut where `dᵢ ≥ μ + z·σ` of **that document's own** drift distribution.
5. Enforce `semantic_max_chars` by recursively bisecting oversized blocks at their strongest
   interior drift peak, biased toward the middle so a cut never shaves off a single sentence.
6. Merge blocks below a third of target back into their neighbour.

The threshold is **per document, not global**. A tightly written passage and a rambling one need
different sensitivities; a global constant is exactly the naive choice the brief warns about.
Each block records its measured `cohesion`, which the fusion stage later uses as a mild quality
prior — a semantically clean block is a better citation than one straddling a topic change.

## L2 — Sliding window with cohesion-adaptive overlap
`server/setu/chunking/window.py`

A fixed-window layer is worth keeping: it is the only view that *guarantees* no region of the
corpus is under-represented, and it survives the badly punctuated machine translation where
sentence splitting fails.

A fixed **overlap** is not worth keeping. Constant 20 % over-indexes tightly-written passages
(wasting space, diluting scores with near-duplicates) and under-protects rambling ones (where the
answer straddles a boundary). So overlap is a function of local cohesion:

```
overlap = lerp(overlap_max, overlap_min, cohesion)      0.50 → 0.14
```

High cohesion means a cut is cheap → small overlap. Low cohesion means a cut is expensive → large
overlap. Every chunk stores the `overlap_ratio` actually used, so the effect is auditable in the
UI rather than a claim in a README. Measured mean on the MS MARCO-XI slice: **23.2 %**, i.e. the
mechanism is genuinely moving within its range rather than pinning to an end.

Windows always snap to sentence boundaries — the character budget picks *which* boundary, it never
cuts mid-sentence.

## L3 — Document with metadata-aware embedding
`server/setu/chunking/metadata.py`

One chunk per passage, for broad questions whose answer is the gestalt of the passage.

The interesting part is what the *encoder* sees. MS MARCO-XI hands us structured context for
free, and we fold a compact header into the embedding text:

```
[hi | NUMERIC | rank 2/10 | answer-bearing] What is a corporation?
<passage text>
```

- **Disambiguation** — machine-translated passages on different topics can be lexically similar;
  a query-type token separates them in vector space.
- **Type matching** — a *how many* question embeds closer to a `NUMERIC`-tagged passage than a
  `DESCRIPTION`-tagged one, for free, with no extra model.

The header never appears in `text`, so citations and the spoken answer stay clean. Ablate it with
`SETU_METADATA_PREFIX=false`.

---

## Parent links: small-to-search, big-to-read

Retrieval runs on the fine layers, where signal is concentrated. Whatever wins is then **expanded
through `parent_id`** to its containing coherence block, so the answer composer reads real context
instead of an orphaned fragment.

```
L0 atom ──parent──▶ L1 semantic block ──parent──▶ L3 document
L2 window ─────────parent──────────────────────▶ L3 document
```

Parents are resolved by maximum character-span overlap, not by re-searching. Windows point
straight at the document because they straddle semantic blocks by design.

The router turns expansion on for fine-grained query shapes (`numeric`, `factoid`, `yesno`,
`entity`) and off for broad ones, where the coarse layers were already searched directly.

## Deduplication, and honest degeneration

A very short passage would otherwise produce four near-identical nodes. Layers are emitted coarse
to fine and a node is dropped when a coarser node already carries byte-identical text. L3 is
always retained, guaranteeing the router a backstop layer for every document.

The result is that **the lattice depth adapts to document length**, and we report it rather than
hiding it. On the measured MS MARCO-XI slice:

| layers built for a document | documents | interpretation |
|---|---|---|
| 4 | 25 | long, multi-topic passage — full stratification |
| 3 | 65 | medium passage |
| 2 | 123 | typical short passage: atoms + document |
| 1 | 25 | single-sentence passage — nothing to stratify |

Overall dedup rate **30.9 %**. That number is the point: the lattice is not four times the index
size, because a third of the nodes it would naively produce are redundant and get dropped.

## Cost

Building four views is *not* four times the cost of building one, because the expensive part is
shared. Sentence embeddings are computed **once per document** and reused by:

- the semantic chunker (drift detection),
- the window chunker (cohesion measurement),
- the reranker (late interaction at query time),
- the answer composer (span scoring at query time).

That last pair is why the reflex path is single-digit milliseconds: at query time no sentence is
ever re-embedded, because they are all sitting in one `(n_sentences, 384)` float32 matrix indexed
by document offset.

Measured on the ingested slice: **8,738 documents → 38k lattice nodes**, roughly 4.4 nodes per
document, with one embedding pass over chunks and one over sentences.

## Configuration, and why these numbers

MS MARCO passages have a median length around 290 characters and 3–5 sentences. Copying the
customary 1000-token chunk settings would make L1 and L2 byte-identical to L3 on *every single
passage*, and dedup would silently erase two of the four views — a four-layer lattice on paper and
a one-layer index in practice. The defaults put roughly one boundary decision inside a typical
passage and several inside a long one:

| setting | value | reason |
|---|---|---|
| `atomic_min_chars` | 45 | below this a node is not independently retrievable |
| `atomic_max_chars` | 240 | ~1–2 sentences; fission target |
| `semantic_max_chars` | 420 | ≈1.5× median passage → real boundary decisions |
| `semantic_z` | 0.45 | cuts at μ+0.45σ of per-document drift |
| `window_chars` | 300 | ≈median passage → windows exist on medium docs |
| `window_overlap` | 0.14 – 0.50 | cohesion-adaptive range |

Verify any change with `python scripts/ablate.py`, which rebuilds under reduced configurations and
reports recall@k against the dataset's own `is_selected` labels — including a
`window-only-naive` baseline (single 500-char layer, no overlap, no metadata, dense-only) so the
comparison is against something real rather than against nothing.
