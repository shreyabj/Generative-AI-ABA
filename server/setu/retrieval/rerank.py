"""Late-interaction reranking, without a cross-encoder.

A cross-encoder reranker is the standard second stage, and it is also completely
unaffordable here: even a MiniLM cross-encoder over 12 candidates is 80-150 ms on
CPU, which is most of the budget for one signal.

So we rerank with information we already paid for. Every sentence in the corpus
was embedded at ingest, and the corpus index can hand back the sentence vectors
covered by any chunk in O(1). That gives a ColBERT-flavoured late interaction at
matmul cost:

    maxsim(q, chunk)  = max_j  cos(q, sentence_j)
    meansim(q, chunk) = mean_j cos(q, sentence_j)

``maxsim`` is the sharp signal a single pooled chunk embedding loses: a 900-character
coherence block whose *third* sentence answers the question has a mediocre mean
vector but an excellent max. Pooling averages the answer away; late interaction
does not.

Three more signals, all cheap:

* **Lexical containment** of query content tokens - catches exact identifiers,
  years and transliterated names that the encoder smooths over.
* **Numeric overlap** - a numeric query should prefer a chunk that actually
  contains numbers.
* **Brevity prior** - among equally relevant candidates, the shorter one is the
  better citation and the better thing to read aloud.

Total cost for 12-24 candidates: about 1-3 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..index.corpus import ChunkRecord, CorpusIndex
from ..text import containment, content_tokens, numbers_in, tokenize


@dataclass(slots=True)
class RerankSignal:
    maxsim: float
    meansim: float
    lexical: float
    numeric: float
    brevity: float
    score: float
    best_sentence: int


class LateInteractionReranker:
    def __init__(
        self,
        w_max: float = 0.46,
        w_mean: float = 0.16,
        w_lex: float = 0.22,
        w_num: float = 0.06,
        w_brev: float = 0.10,
        ideal_chars: int = 260,
    ) -> None:
        self.w_max = w_max
        self.w_mean = w_mean
        self.w_lex = w_lex
        self.w_num = w_num
        self.w_brev = w_brev
        self.ideal_chars = ideal_chars

    def score_one(
        self,
        index: CorpusIndex,
        rec: ChunkRecord,
        qvec: np.ndarray,
        q_content: set[str],
        q_numbers: set[str],
        *,
        numeric_query: bool,
    ) -> RerankSignal:
        sent_vecs = index.sentence_vectors(rec)
        if sent_vecs.shape[0]:
            sims = sent_vecs @ qvec
            best = int(np.argmax(sims))
            maxsim = float(sims[best])
            meansim = float(sims.mean())
        else:
            row = index.row_of(rec.chunk_id)
            fallback = float(index.vectors[row] @ qvec) if row is not None else 0.0
            best, maxsim, meansim = 0, fallback, fallback

        chunk_tokens = set(tokenize(rec.text))
        lexical = containment(q_content, chunk_tokens) if q_content else 0.0

        numeric = 0.0
        if numeric_query:
            chunk_nums = set(numbers_in(rec.text))
            if chunk_nums:
                numeric = 1.0 if (q_numbers & chunk_nums) else 0.55

        n = max(len(rec.text), 1)
        brevity = float(np.exp(-abs(n - self.ideal_chars) / (self.ideal_chars * 2.6)))

        score = (
            self.w_max * maxsim
            + self.w_mean * meansim
            + self.w_lex * lexical
            + self.w_num * numeric
            + self.w_brev * brevity
        )
        return RerankSignal(maxsim, meansim, lexical, numeric, brevity, float(score), best)

    def rerank(
        self,
        index: CorpusIndex,
        rows: list[int],
        query: str,
        qvec: np.ndarray,
        *,
        numeric_query: bool = False,
    ) -> list[tuple[int, RerankSignal]]:
        q_content = set(content_tokens(query))
        q_numbers = set(numbers_in(query))
        out: list[tuple[int, RerankSignal]] = []
        for row in rows:
            rec = index.record(row)
            sig = self.score_one(
                index, rec, qvec, q_content, q_numbers, numeric_query=numeric_query
            )
            out.append((row, sig))
        out.sort(key=lambda t: -t[1].score)
        return out


__all__ = ["LateInteractionReranker", "RerankSignal"]
