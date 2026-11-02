# Guardrails: knowing when *not* to answer

> The brief: *"Show that your system knows when **not** to answer, not just how to answer."*

Seven gates across four stages. Every gate returns a **score and the threshold it had to clear** —
never a bare boolean, because "blocked" with no number attached is not something you can tune,
test, or show a user. The full verdict list is returned in every API response and rendered in the
UI's trust panel.

```
                                   ┌──────────────┐
  query ──▶ sanity ──▶ injection ──┤ REFUSE       │   ~0.2 ms, no vectors touched
                │                  └──────────────┘
                ▼
             route ──▶ embed ──▶ safety ─────────▶ REFUSE      semantic, 15 languages
                                    │
                                    ▼
                                 retrieve ──▶ topicality ────▶ ABSTAIN   "not in my corpus"
                                                  │
                                                  ▼
                                               answer ──▶ groundedness ─▶ ABSTAIN
                                                              │
                                                              ▼
                                                          PII redact
                                                              │
                                                              ▼
                                                        trust floor ────▶ ABSTAIN
                                                              │
                                                              ▼
                                                           ANSWER
```

Gate placement is a latency decision as much as a safety one. The vector-free gates run first, so
a junk or injected turn costs ~0.2 ms instead of a full retrieval pass. The semantic safety gate
runs immediately *after* `embed_query`, because its classifier needs the same query vector
retrieval is about to use — and still lands before the index is touched.

---

## Stage 1 — Admission (no vectors, ~0.2 ms)

### `input.sanity`
Rejects input that is not a question: empty or over-long, punctuation-only, degenerate character
repetition (`aaaaaaaa…`), and a unique-token ratio below 0.34 on queries of 8+ tokens (spam, not a
question).

### `input.injection`
Fourteen compiled patterns across five attack families — `override-instructions`,
`persona-hijack`, `prompt-extraction`, `mode-escalation`, `role-token-injection`,
`guardrail-removal` — including Devanagari variants, since an injection written in Hindi is still
an injection.

Worth noting: the Reflex path is **structurally immune** to prompt injection, because it never
sends the query to a generative model — it selects spans with numpy. This gate exists to protect the
Deliberate path and to refuse clearly-hostile input rather than quietly serving it.

---

## Stage 2 — Semantic safety (~0.3 ms)

### `input.safety`

A keyword blocklist is trivially defeated in a multilingual system. Writing the list in English
leaves 14 languages unguarded; writing it in 15 languages is unmaintainable and *still* loses to
paraphrase.

So the primary signal is **semantic**. Seven harm categories — `weapons`, `self_harm`,
`cyber_attack`, `illicit_drugs`, `csae`, `targeted_hate`, `fraud` — each have a small set of seed
phrases embedded once at startup and averaged into a centroid. An incoming query is compared
against those centroids **in the same multilingual space the retriever already uses**.

Two properties fall out of this for free:

- **Cross-lingual coverage.** A request phrased in Hindi, Tamil or Urdu is caught by seeds written
  in English, because the encoder shares one vector space across scripts.
- **Paraphrase robustness.** Rewording does not help the attacker, because invariance to
  rewording is exactly what a sentence encoder is for.

Cost: one dot product against seven vectors, on a query vector that already exists.

Lexical patterns are retained as a **high-precision pre-filter**, not as the defence.

### The benign centroid — avoiding a gate that breaks the product

This is the part most safety layers get wrong. MS MARCO is scraped from the open web and contains
medical, pharmacological, historical and security passages. A gate that blocks *"what are the
symptoms of an opioid overdose"* or *"what was the immediate impact of the Manhattan Project"* has
broken the product to protect nothing.

So there is a second centroid set built from **twelve deliberately adjacent benign seeds**, and a
query is only blocked when it is closer to harm than to benign:

```python
if harm >= threshold and harm > benign + 0.02:
    block
elif harm >= threshold - 0.09:
    flag        # allowed through, corpus-grounded answer only, trust reduced
else:
    allow
```

The `flag` band matters: borderline queries are answered, but only from retrieved evidence, and the
composite trust score is reduced so the caller knows.

**Self-harm gets a safe completion, not a refusal.** The response is a localised message that
includes India's Tele-MANAS (14416) and KIRAN (1800-599-0019) helplines. Refusing and saying nothing
is the wrong behaviour for that category.

---

## Stage 3 — Topicality: does the corpus actually contain this?

### `retrieval.topicality`

A retriever always returns its top *k*. Ask a MS MARCO index about last night's football score and
it hands back five passages at cosine ≈ 0.2 with no sign of distress. Everything downstream then
works hard to produce a confident, well-cited, completely wrong answer.

**Most RAG hallucinations start here, not in the generator.** Three independent signals:

**Absolute score** — best fused hit must clear `topicality_min_score` (0.34). Nothing resembling
the question is in the index.

**Margin** — top-1 minus top-2 must clear `topicality_min_margin` (0.015). A flat distribution
means the retriever found no *distinguishing* evidence, only generic proximity — which is what
"not in the corpus" looks like from the inside. This catches the dangerous case absolute score
misses: a plausible 0.5 that is 0.5 for a hundred other passages too.

**Lexical anchoring** — at least one query content token should appear in the top-3 evidence.
Dense retrieval can drift to topically-adjacent-but-different passages; one shared anchor token is
a cheap, effective check. Failure downgrades to `flag` rather than `block`, since cross-lingual
retrieval legitimately shares no surface tokens.

A block here produces a **localised** "not in my knowledge base" — the correct answer to an
out-of-corpus question, and a much better product than a confident invention.

---

## Stage 4 — Output verification

### `output.groundedness`

Verifies the answer against the evidence **the answer itself cited**. Not "is this plausible" —
sentence by sentence, "is this supported".

Per claim (one answer sentence, citation markers stripped):

| signal | what it catches |
|---|---|
| **Lexical containment** of claim content tokens in cited chunks | the base attribution measure. Asymmetric on purpose — Jaccard would penalise a short claim citing a long passage, which is the normal case |
| **Numeric consistency** — every number in the claim must appear in the evidence | the highest-value single check in the system |
| **Entity consistency** — capitalised tokens must appear in the evidence | invented names and places |
| **Semantic entailment** — claim embedding vs cited sentence vectors, `maxsim` | *deliberate path only* — a paraphrase can be grounded yet score low on containment |

On **numeric consistency**: quantities are where hallucination does real damage. `"89 calories"`
versus `"189 calories"` is invisible to a fluency-based reviewer, catastrophic to a user, and
verifiable with a set intersection. A claim containing a number that appears **nowhere** in the
retrieved evidence is scored 0 and the answer is discarded outright, whatever else it scored.

The distinction is deliberate: a number present in evidence chunk 2 while the claim cites chunk 1 is
a *citation imprecision* (support × 0.85). A number present in no chunk at all is a *hallucination*
(hard block). Conflating the two would make the gate either useless or unusably strict.

The semantic check is skipped for the reflex path — where it would be both redundant (tokens are
copied verbatim) and unaffordable inside 200 ms.

Coverage is the length-weighted mean of per-claim support. Below `groundedness_min_coverage` (0.42)
the answer is discarded and replaced with a localised abstention.

> **A note on citation-marker placement.** Markers go *inside* the sentence, before terminal
> punctuation: `"...a legal entity [1]."` not `"...a legal entity. [1]"`. This is not cosmetic —
> this gate re-splits the answer into sentences, and a trailing marker lands at the start of the
> *next* sentence after splitting, so claim 1 would look uncited and claim 2 would appear to cite
> evidence it never used. This was a real bug, found by the smoke test, and it is the kind of thing
> that silently corrupts an attribution metric.

### `output.pii`

MS MARCO is scraped from the open web, an extractive answerer copies spans verbatim, and in a voice
system the result gets **spoken aloud**. This gate is the last thing between the corpus and the
speaker.

Detectors: email, Indian and international phone formats, Aadhaar-shaped 12-digit groups, PAN,
payment cards, IPv4. Aadhaar candidates are validated with the **Verhoeff checksum** and cards with
**Luhn**, so ordinary 12-digit and 16-digit quantities are not mangled — a redactor with false
positives destroys legitimate answers about prices and measurements.

Redaction happens on the way *out*, not at ingest, so the tokens stay searchable (a query about a
company's support line still retrieves the right passage) while the values do not leave the process.

### `output.trust-floor`

A composite 0–100 score, published so a caller has one number, and decomposed so the number is not
a vibe:

| weight | component | question it answers |
|---|---|---|
| 40 | claim coverage | is it attributable? |
| 18 | retrieval top score | did we find anything good? |
| 12 | retrieval margin | was it *distinctly* good? |
| 10 | weakest individual claim | is any part unsupported? |
| 10 | citation density (distinct sources) | independently corroborated? |
| 10 | safety / injection clearance | is the request itself clean? |

Below `trust_abstain_below` (45) the answer is withheld **even when no single gate blocked it**.
This is the last line of defence against answers that are individually plausible on every axis and
collectively unconvincing.

---

## Localisation

Every refusal, abstention and safe completion exists in the corpus languages. A guardrail that
refuses in English to someone who asked in Tamil has failed at the part the user actually
experiences. See `server/setu/answer/compose.py`.

---

## Testing: both error directions

```bash
python scripts/redteam.py --verbose      # writes docs/redteam.json
```

41 labelled cases across 12 families. The suite reports a confusion matrix, because both
directions are real failures:

- **False negative** — unsafe, injected or out-of-corpus input got answered.
- **False positive** — a legitimate corpus question got refused. Twelve cases exist specifically to
  catch this, all sitting deliberately close to a harm centroid: opioid overdose symptoms,
  paracetamol side effects, chemotherapy, depression warning signs, the Manhattan Project, WWII
  causes, 2FA, firewalls, money-laundering penalties.

Benign *abstentions* are counted separately from benign *refusals*: abstaining because a passage is
genuinely absent from a 660-row corpus slice is correct behaviour, not a guardrail failure. Scoring
those as failures would push you toward loosening the groundedness threshold to fix a problem that
does not exist.

The script exits non-zero on any false negative or false positive, so it gates CI.

## Tuning

| setting | default | effect |
|---|---|---|
| `SETU_TOPICALITY_MIN_SCORE` | 0.34 | ↑ abstains more on weak retrieval |
| `SETU_TOPICALITY_MIN_MARGIN` | 0.015 | ↑ abstains more on flat score distributions |
| `SETU_GROUNDEDNESS_MIN_COVERAGE` | 0.42 | ↑ demands stronger attribution |
| `SETU_TRUST_ABSTAIN_BELOW` | 45 | ↑ withholds more borderline answers |
| `SETU_SAFETY_SIMILARITY_THRESHOLD` | 0.68 | ↓ blocks more aggressively (watch false positives) |
| `SETU_NUMERIC_CONSISTENCY` | true | disables the numeric hallucination check |

Change one, then run `scripts/redteam.py` **and** `scripts/evaluate.py`. The second one matters:
tightening a guardrail always improves the red-team score and always costs answer rate, and
`evaluate.py` reports `abstain_wrong` — how often gold evidence was retrieved and the system
declined anyway. That is the number that tells you a threshold has gone too far.
