"""The Reflex path - deterministic extractive answering.

Why this is the default path, not a fallback
--------------------------------------------
The brief asks for the whole pipeline to finish under 200 ms. A generative call is
structurally incapable of that: the fastest hosted 8B model has a time-to-first-token
around 100-250 ms *before* it emits a sentence, and that is network latency we do not
control. Building the primary path on it would mean either missing the SLO or reporting
a number that quietly excludes the slowest stage.

MS MARCO is an extractive machine-reading corpus. Its passages were collected because
they *contain* the answer, and the dataset ships a human ``is_selected`` label marking
which one does. So the right primary answerer is a span selector, not a paraphraser -
and it happens to be about 400x faster.

The property that matters most is not speed, though. It is that an extractive answer
**cannot hallucinate**: every token it emits was copied from a retrieved chunk, so
groundedness is structural rather than something we check for afterwards and hope. The
verification layer in :mod:`setu.guardrails.groundedness` still runs, because the
*selection* can be wrong even when the tokens are real - but the failure mode is
"cited the wrong sentence", which is detectable, rather than "invented a plausible
number", which often is not.

The Deliberate path (:mod:`setu.answer.llm`) exists for fluent multi-hop synthesis and
is measured separately, outside the 200 ms SLO. See docs/LATENCY.md.

Scoring
-------
Every sentence inside the retrieved chunks is a candidate, scored on:

  0.42  semantic similarity to the query   (cached sentence vector - free)
  0.24  lexical containment of query content tokens
  0.14  the retrieval score of its parent chunk
  0.08  shape affinity (numeric query prefers sentences with numbers, definition
        queries prefer copular/defining constructions)
  0.07  language match with the query
  0.05  position prior (MS MARCO answers skew towards passage openings)

Selection is greedy with a redundancy veto (Jaccard > 0.55 against anything already
chosen), so a three-sentence answer says three things instead of one thing three times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..harness.contracts import AnswerDraft, ChunkHit, Layer, QueryShape, RoutePlan
from ..index.corpus import CorpusIndex
from ..text import Sentence, containment, content_tokens, jaccard, numbers_in, tokenize
from .compose import build_citations, stitch

_DEFINING_RE = re.compile(
    r"\b(is|are|was|were|refers to|means|defined as|known as|called)\b"
    r"|\b(है|हैं|कहलाता|अर्थात्)\b"
    r"|\b(হয়|বলা হয়)\b"
    r"|\b(ஆகும்|எனப்படும்)\b",
    re.IGNORECASE | re.UNICODE,
)

# ------------------------------------------------------------------ span quality
#
# Retrieval scores tell you which *passage* is relevant. They say nothing about whether a
# given sentence inside it is a usable answer, and web text is full of sentences that are
# highly on-topic and completely useless. Two real failures from the MS MARCO slice:
#
#   "Examples of corporation in a Sentence."          <- pure scaffolding, zero information
#   "1: a government-owned corporation (as a ..."     <- a dictionary sub-sense, not a definition
#
# Both scored above "A corporation is a company or group of people authorized to act as a
# single entity", which is the answer a person wants. So span quality is scored explicitly
# instead of being left to semantic similarity, which cannot see the difference.

# Dictionary / SEO / navigation scaffolding.
_BOILERPLATE_RE = re.compile(
    r"examples?\s+of\s+.{0,40}\s+in\s+a\s+sentence"
    r"|^\s*(see|see\s+more|read\s+more|learn\s+more|more\s+examples|related\s+words"
    r"|synonyms?|antonyms?|how\s+to\s+use|first\s+known\s+use|recent\s+examples"
    r"|word\s+history|other\s+words|share\s+this|click\s+here|sign\s+up|log\s+in)\b"
    r"|\bin\s+a\s+sentence\s*$"
    r"|^\s*(definition\s+of|meaning\s+of)\s+\w+\s*$",
    re.IGNORECASE | re.UNICODE,
)

# A leading numeral: dictionary sense markers ("1:") and example indices ("2 a substantial…").
# These are fragments of a list, not sentences that stand on their own.
_ENUM_LEAD_RE = re.compile(r"^\s*\(?\d{1,2}\)?\s*[:.)\]]?\s+\S", re.UNICODE)

# A sentence that opens by naming its subject and then defining it. High precision, and
# exactly the shape a "what is X" question wants back.
_DEF_LEAD_TEMPLATE = (
    r"^\s*(?:the\s+|a\s+|an\s+)?{subj}s?\b[^.!?]{{0,40}}?"
    r"\b(?:is|are|was|were|refers?\s+to|means|denotes|describes|"
    r"can\s+be\s+defined\s+as|is\s+defined\s+as|definition)\b"
)

W_SEM = 0.40
W_LEX = 0.22
W_HIT = 0.13
W_SHAPE = 0.13  # raised: definitional form is the strongest signal for "what is X"
W_LANG = 0.07
W_POS = 0.05

# Multiplicative, so a scaffolding sentence cannot win on similarity alone.
P_BOILERPLATE = 0.25
P_ENUM_LEAD = 0.55

# How many sentences to emit, by query shape. A definition answered in three sentences reads
# like a scrape; a "why" question answered in one reads like a fragment.
SENTENCES_BY_SHAPE: dict[QueryShape, int] = {
    QueryShape.DEFINITION: 2,
    QueryShape.NUMERIC: 1,
    QueryShape.FACTOID: 2,
    QueryShape.YESNO: 2,
    QueryShape.ENTITY: 2,
    QueryShape.LIST: 3,
    QueryShape.DESCRIPTION: 3,
    QueryShape.UNKNOWN: 2,
}

# Stop-words that can never be the subject of a "what is X" question.
_SUBJECT_STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "for", "to",
    "do", "does", "did", "define", "definition", "meaning", "means", "tell", "me",
    "about", "explain", "describe",
}


def query_subject(query: str) -> str | None:
    """The head noun a definition question is about: "what is a corporation" -> "corporation".

    Used to bind the definitional pattern to the actual subject, so
    "A corporation is a company..." is rewarded but "The corporation is then governed by the
    laws of incorporation in that state" - same words, not a definition - is not.
    """
    toks = [t for t in tokenize(query) if t not in _SUBJECT_STOP and len(t) > 2]
    return toks[-1] if toks else None


@dataclass(slots=True)
class Span:
    sentence: Sentence
    source_index: int
    chunk_id: str
    doc_id: str
    lang: str
    score: float
    semantic: float
    lexical: float
    doc_position: float


@dataclass(slots=True)
class _Source:
    """A passage the answer may be read out of, and what its citation should say.

    Retrieved hits become sources directly. Parallel passages - the same evidence in the
    language the user asked in - are added alongside them, which is what lets the language
    lock below be a rule rather than a preference.
    """

    rec: object  # ChunkRecord; kept loose to avoid a circular import in the annotation
    chunk_id: str
    doc_id: str
    layer: object
    text: str
    lang: str
    score: float
    order: int
    translated: bool = False


# A parallel passage is the *same* passage, so it inherits its sibling's measured relevance
# in full. A different passage from the same source question is only inferred to be
# relevant, so it enters discounted and has to earn its place on span score alone.
PARALLEL_SCORE = 1.00
GROUP_SCORE = 0.60


class ExtractiveAnswerer:
    """Selects and stitches answer spans out of retrieved evidence."""

    def __init__(self, index: CorpusIndex) -> None:
        self.index = index

    # ==================================================================== public
    def answer(
        self,
        query: str,
        qvec: np.ndarray,
        hits: list[ChunkHit],
        plan: RoutePlan,
        *,
        max_sentences: int | None = None,
        max_chars: int | None = None,
    ) -> AnswerDraft:
        # Length follows the question. A definition wants one or two sentences; a "why"
        # question needs three. Emitting a fixed three every time is what made short
        # answers read like a scrape of the source page.
        if max_sentences is None:
            max_sentences = min(
                SENTENCES_BY_SHAPE.get(plan.shape, 2), settings.answer_max_sentences
            )
        max_chars = max_chars or settings.answer_max_chars
        if not hits:
            return AnswerDraft(text="", lang=plan.lang, path="reflex", confidence=0.0)

        sources = self._sources(hits, plan.lang)
        spans = self._candidates(query, qvec, sources, plan)
        if not spans:
            return AnswerDraft(text="", lang=plan.lang, path="reflex", confidence=0.0)

        spans = self._prefer_query_language(spans, plan.lang)
        chosen = self._select(spans, max_sentences)
        if not chosen:
            return AnswerDraft(text="", lang=plan.lang, path="reflex", confidence=0.0)

        # Read in source order so the answer flows like prose, not like a ranking.
        chosen.sort(key=lambda s: (s.source_index, s.sentence.start))

        # One citation marker per distinct source chunk.
        marker_of: dict[str, int] = {}
        cite_rows: list[tuple[int, str, str, object, str, float]] = []
        markers: list[int] = []
        for sp in chosen:
            m = marker_of.get(sp.chunk_id)
            if m is None:
                m = len(marker_of) + 1
                marker_of[sp.chunk_id] = m
                src = sources[sp.source_index]
                cite_rows.append((m, src.chunk_id, src.doc_id, src.layer, src.text, src.score))
            markers.append(m)

        text = stitch([s.sentence.text for s in chosen], markers, max_chars=max_chars)
        answer_lang = chosen[0].lang or plan.lang

        top = max(s.score for s in chosen)
        spread = float(np.mean([s.semantic for s in chosen]))
        confidence = float(np.clip(0.58 * top + 0.30 * spread + 0.12 * min(len(marker_of), 3) / 3.0, 0.0, 1.0))

        return AnswerDraft(
            text=text,
            lang=answer_lang,
            citations=build_citations(cite_rows),  # type: ignore[arg-type]
            path="reflex",
            model="extractive-span-fusion",
            confidence=round(confidence, 4),
        )

    # ================================================================= internals
    def _widest_row(self, doc_id: str) -> int | None:
        """The chunk row covering the most of a document - its whole-passage view."""
        rows = self.index.rows_of_doc(doc_id)
        if not rows:
            return None
        best, best_span = None, -1
        for row in rows:
            rec = self.index.record(row)
            lo, hi = rec.sent_span
            width = (hi or 0) - lo
            if rec.layer is Layer.DOCUMENT:
                return row
            if width > best_span:
                best, best_span = row, width
        return best

    def _sources(self, hits: list[ChunkHit], lang: str) -> list[_Source]:
        """Retrieved hits, plus the same evidence in the language of the question.

        Retrieval is cross-lingual on purpose, so the best hit for a Kannada question is
        very often the English original. Without this step the answerer had to choose
        between answering in the wrong language or not answering; with it, the same passage
        is available in Kannada and the choice disappears.
        """
        out: list[_Source] = []
        seen: set[str] = set()

        for i, hit in enumerate(hits):
            row = self.index.row_of(hit.chunk_id)
            if row is None:
                continue
            rec = self.index.record(row)
            out.append(
                _Source(
                    rec=rec, chunk_id=hit.chunk_id, doc_id=rec.doc_id, layer=hit.layer,
                    text=hit.text, lang=rec.lang, score=float(hit.score), order=i,
                )
            )
            seen.add(rec.doc_id)

        if not lang:
            return out

        for src in list(out):
            if src.lang == lang:
                continue
            targets: list[tuple[str, float]] = []
            parallel = self.index.parallel_doc_id(src.doc_id, lang)
            if parallel:
                targets.append((parallel, src.score * PARALLEL_SCORE))
            else:
                for did in self.index.group_doc_ids(src.doc_id, lang)[:2]:
                    targets.append((did, src.score * GROUP_SCORE))

            for doc_id, score in targets:
                if doc_id in seen:
                    continue
                row = self._widest_row(doc_id)
                if row is None:
                    continue
                rec = self.index.record(row)
                seen.add(doc_id)
                out.append(
                    _Source(
                        rec=rec, chunk_id=rec.chunk_id, doc_id=doc_id, layer=rec.layer,
                        text=rec.text, lang=rec.lang, score=score, order=src.order,
                        translated=True,
                    )
                )
        return out

    def _candidates(
        self, query: str, qvec: np.ndarray, sources: list[_Source], plan: RoutePlan
    ) -> list[Span]:
        q_content = set(content_tokens(query))
        q_numbers = set(numbers_in(query))
        numeric = plan.shape is QueryShape.NUMERIC
        definition = plan.shape is QueryShape.DEFINITION

        # Bind the definitional pattern to the question's actual subject.
        subject = query_subject(query)
        def_lead = (
            re.compile(_DEF_LEAD_TEMPLATE.format(subj=re.escape(subject)), re.IGNORECASE | re.UNICODE)
            if subject
            else None
        )
        hit_scores = np.asarray([s.score for s in sources], dtype=np.float32)
        hi = float(hit_scores.max()) if hit_scores.size else 1.0
        lo = float(hit_scores.min()) if hit_scores.size else 0.0
        rng = max(hi - lo, 1e-6)

        out: list[Span] = []
        for src_idx, src in enumerate(sources):
            rec = src.rec
            pairs = self.index.sentences_of(rec)  # type: ignore[arg-type]
            if not pairs:
                continue
            doc = self.index.docs.get(rec.doc_id)  # type: ignore[union-attr]
            doc_len = max(len(doc.text), 1) if doc else 1
            hit_prior = (src.score - lo) / rng

            for sent, svec in pairs:
                if len(sent.text) < 24:
                    continue
                semantic = float(svec @ qvec) if svec.size else 0.0
                s_tokens = set(tokenize(sent.text))
                lexical = containment(q_content, s_tokens) if q_content else 0.0

                shape_bonus = 0.0
                if numeric:
                    s_nums = set(numbers_in(sent.text))
                    if s_nums:
                        shape_bonus = 1.0 if (q_numbers & s_nums) else 0.7
                elif definition:
                    # A leading "<subject> is ..." is worth far more than the word "is"
                    # appearing somewhere in the sentence.
                    if def_lead and def_lead.search(sent.text):
                        shape_bonus = 1.0
                    elif _DEFINING_RE.search(sent.text):
                        shape_bonus = 0.35

                lang_bonus = 1.0 if src.lang == plan.lang else 0.0
                position = 1.0 - min(sent.start / doc_len, 1.0)

                score = (
                    W_SEM * semantic
                    + W_LEX * lexical
                    + W_HIT * hit_prior
                    + W_SHAPE * shape_bonus
                    + W_LANG * lang_bonus
                    + W_POS * position
                )

                # Quality gates, applied last and multiplicatively: no amount of topical
                # similarity should let scaffolding or a list fragment become the answer.
                if _BOILERPLATE_RE.search(sent.text):
                    score *= P_BOILERPLATE
                if _ENUM_LEAD_RE.match(sent.text):
                    score *= P_ENUM_LEAD
                out.append(
                    Span(
                        sentence=sent, source_index=src_idx, chunk_id=src.chunk_id,
                        doc_id=src.doc_id, lang=src.lang, score=float(score),
                        semantic=semantic, lexical=lexical, doc_position=position,
                    )
                )
        out.sort(key=lambda s: -s.score)
        return out

    @staticmethod
    def _prefer_query_language(spans: list[Span], lang: str) -> list[Span]:
        """Answer in the language the question was asked in.

        The corpus stores each passage once per language, so a Kannada question routinely
        retrieves the English original as its single best span. Cross-lingual *retrieval* is
        the feature; a Kannada question answered in English is not, and it is worse out
        loud, because the wrong voice then reads the wrong script.

        This used to be a preference with two escape hatches - no same-language span at all,
        or the best one scoring under half the global best - and both leaked English answers
        to Indic questions. They are gone, because ``_sources`` now supplies the same
        evidence in the target language before scoring, so a same-language span existing is
        the normal case rather than a lucky one.

        The remaining fallback is deliberate and narrow: if this corpus genuinely holds no
        passage in that language, answer from what there is. Being right in the wrong
        language still beats silence, and ``AnswerDraft.lang`` reports what actually came
        out so the caller - and the voice - follow the answer rather than the intent.
        """
        if not lang:
            return spans
        same = [s for s in spans if s.lang == lang]
        return same or spans

    @staticmethod
    def _select(spans: list[Span], limit: int) -> list[Span]:
        """Greedy selection: redundancy veto, one language, and one sentence per source first.

        The per-source preference is why an answer no longer reads as three consecutive
        sentences lifted off one page, all citing ``[1]``. Taking the next sentence from a
        *different* passage reads better and means the groundedness gate is checking
        independent evidence instead of the same passage three times. If nothing else
        qualifies, a second pass allows more from the same source rather than truncating.
        """
        best = spans[0].score if spans else 0.0
        chosen: list[Span] = []
        seen: list[set[str]] = []
        per_chunk: dict[str, int] = {}

        def consider(sp: Span, allowance: int) -> None:
            if len(chosen) >= limit or sp in chosen:
                return
            if chosen and sp.score < best * 0.55:
                return
            if per_chunk.get(sp.chunk_id, 0) >= allowance:
                return
            # Never mix languages inside one answer - it reads as broken, and it makes the
            # spoken output unusable because TTS gets one voice and two scripts.
            if chosen and sp.lang != chosen[0].lang:
                return
            toks = set(tokenize(sp.sentence.text))
            if any(jaccard(toks, prev) > 0.5 for prev in seen):
                return
            chosen.append(sp)
            seen.append(toks)
            per_chunk[sp.chunk_id] = per_chunk.get(sp.chunk_id, 0) + 1

        for allowance in (1, limit):
            for sp in spans:
                consider(sp, allowance)
            if len(chosen) >= limit:
                break
        return chosen


__all__ = ["ExtractiveAnswerer", "Span"]
