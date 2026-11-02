"""Retrieval-stage gate: does the corpus actually contain this answer?

A retriever always returns its top k. Ask a MS MARCO index about last night's football
score and it will hand back five passages with cosine similarities around 0.2 and no
sign of distress. Everything downstream then works hard to produce a confident,
well-cited, completely wrong answer. Most RAG hallucinations start here, not in the
generator.

This gate refuses the turn on three independent signals:

**Absolute score.** The best fused hit must clear ``topicality_min_score``. Nothing
resembling the question is in the index.

**Margin.** Top-1 minus top-2 must clear ``topicality_min_margin``. A flat distribution
means the retriever found no *distinguishing* evidence, only generic proximity - which is
what "not in the corpus" looks like from the inside. This catches the dangerous case that
absolute score misses: a plausible-looking 0.5 that is 0.5 for a hundred other passages too.

**Lexical anchoring.** At least one content token of the query should appear in the
retrieved text. Dense retrieval alone can drift to topically-adjacent-but-different
passages; requiring one shared anchor token is a cheap, effective grounding check.

Blocking here produces a localised "not in my knowledge base" - which is the correct
answer to an out-of-corpus question, and a much better product than a confident invention.
"""

from __future__ import annotations

import re

from ..config import settings
from ..harness.contracts import GuardVerdict
from ..harness.tracing import now_ms
from ..text import content_tokens, tokenize
from .base import BaseGate, GateContext

# Time-relative and predictive phrasing. The corpus is a fixed snapshot with no timestamps,
# so a question anchored to *now* cannot be grounded in it no matter how well it retrieves -
# and it will retrieve well, because "prime minister" and "stock market" are perfectly
# ordinary corpus topics. Retrieval score cannot detect this class of failure at all: the
# evidence is topically right and temporally meaningless. Cheapest correct answer is to
# recognise the phrasing and decline.
_TEMPORAL_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|right\s+now|just\s+now|currently|at\s+the\s+moment|"
    r"this\s+(?:morning|afternoon|evening|week|month|year)|"
    r"last\s+(?:night|week|month)|next\s+(?:week|month|year)|"
    r"latest|most\s+recent|breaking|so\s+far\s+this|"
    r"who\s+is\s+(?:the\s+)?current|what\s+is\s+the\s+current\s+(?:price|score|rate|value)|"
    r"will\s+.{0,30}\b(tomorrow|next\s+week|next\s+year|in\s+the\s+future))\b"
    r"|\b(आज|कल|अभी|इस\s+समय|वर्तमान\s+में)\b"
    r"|\b(இன்று|நாளை|இப்போது|தற்போது)\b"
    r"|\b(আজ|আগামীকাল|এখন|বর্তমানে)\b",
    re.IGNORECASE | re.UNICODE,
)


class TopicalityGate(BaseGate):
    name = "retrieval.topicality"
    stage = "retrieval"

    def __init__(
        self,
        min_score: float | None = None,
        min_margin: float | None = None,
        require_anchor: bool = True,
    ) -> None:
        self.min_score = settings.topicality_min_score if min_score is None else min_score
        self.min_margin = settings.topicality_min_margin if min_margin is None else min_margin
        self.require_anchor = require_anchor

    def check(self, ctx: GateContext) -> GuardVerdict:
        t0 = now_ms()
        res = ctx.retrieval
        if res is None or not res.hits:
            return self.block(0.0, reason="retriever returned nothing", started=t0)

        top = float(res.top_score)
        margin = float(res.margin)

        temporal = _TEMPORAL_RE.search(ctx.query)
        if temporal and ctx.extras.get("dated_corpus"):
            # Knowledge collections carry publication dates, so "latest" or "this year"
            # can be answered relative to the newest document - provided the answer says
            # which date it is relative to. Flag (and let the composer cite dates)
            # instead of declining the whole class as the undated MS MARCO corpus must.
            pass
        elif temporal:
            return self.block(
                0.0,
                reason=(
                    f"time-relative question ('{temporal.group(0)}') - the corpus is a fixed "
                    "snapshot with no timestamps, so no retrieval score can ground it"
                ),
                started=t0, detector="temporal-deixis", top_score=round(top, 4),
            )

        if top < self.min_score:
            return self.block(
                top, reason=f"best evidence scores {top:.3f}, below the {self.min_score} floor",
                threshold=self.min_score, started=t0, margin=round(margin, 4),
            )

        if margin < self.min_margin and len(res.hits) > 1:
            return self.block(
                margin,
                reason=(
                    f"score distribution is flat (margin {margin:.4f}) - the corpus has "
                    "nothing distinctly relevant"
                ),
                threshold=self.min_margin, started=t0, top_score=round(top, 4),
            )

        anchors: list[str] = []
        if self.require_anchor:
            # ``content_tokens`` drops interrogatives as well as stop words, so a shared
            # "what is" / "ಎಂದರೇನು" can never present itself as agreement about the
            # subject. It used to: a Kannada "what is X" query anchored on a passage
            # titled "What is ringworm?" and this gate reported two lexical anchors.
            q_content = set(content_tokens(ctx.query))

            # Lexical overlap is only evidence when both sides are written in the same
            # language. Cross-lingual retrieval is a feature here - a Kannada question is
            # meant to be able to answer from an English passage - and two different
            # scripts share no tokens by construction, so demanding an anchor would flag
            # every correct cross-lingual turn as drift.
            same_lang = [h for h in res.hits[:3] if not h.lang or not ctx.lang or h.lang == ctx.lang]
            if q_content and same_lang:
                evidence = set()
                for h in same_lang:
                    evidence |= set(tokenize(h.text))
                anchors = sorted(q_content & evidence)
                if not anchors:
                    return self.flag(
                        top,
                        reason="no query content token appears in the evidence - possible topical drift",
                        threshold=self.min_score, started=t0, margin=round(margin, 4),
                        checked_langs=sorted({h.lang for h in same_lang if h.lang}),
                    )

        return self.allow(
            top, reason=f"top {top:.3f}, margin {margin:.4f}, {len(anchors)} lexical anchor(s)",
            threshold=self.min_score, started=t0, margin=round(margin, 4), anchors=anchors[:6],
        )


__all__ = ["TopicalityGate"]
