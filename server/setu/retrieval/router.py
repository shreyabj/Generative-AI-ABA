"""Query router - decides *which* lattice layers to search, per query.

Building four chunk views is only useful if something chooses between them. That
is this module. It classifies the query's *shape* from cheap surface features and
emits a plan: which layers, how deep into each, how much to trust dense versus
sparse retrieval, and whether the optional stages are worth their milliseconds.

It is deliberately rule-based rather than a model:

* **Cost.** It runs in roughly 0.2-0.5 ms. A classifier model would eat 10-20 ms
  of a 200 ms budget to make a decision that six features already make well.
* **Explainability.** Every plan carries a human-readable ``reason``, which the UI
  displays. "Routed to atomic because the query is numeric and 5 tokens" is
  auditable; a softmax over four classes is not.
* **No training data.** We are not given a routing-labelled set, and inventing one
  would be worse than encoding the linguistics directly.

Feature set: token count, wh-word presence, digits, numeric/definition/list cue
lexicons across five scripts, entity density (rare or capitalised tokens), and
detected script. MS MARCO's own ``query_type`` taxonomy (NUMERIC, DESCRIPTION,
ENTITY, LOCATION, PERSON) informs the shape vocabulary, so the router's classes
line up with how the corpus is actually annotated.
"""

from __future__ import annotations

import re

from ..config import settings
from ..harness.contracts import Layer, LayerPlan, QueryShape, RoutePlan
from ..text import (
    DEFINITION_CUES,
    DEFINITION_PHRASES,
    LIST_CUES,
    LIST_PHRASES,
    NUMERIC_CUES,
    NUMERIC_PHRASES,
    WH_WORDS,
    YESNO_CUES,
    content_tokens,
    detect_lang,
    detect_script,
    tokenize,
)

_DIGIT_RE = re.compile(r"\d")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}")

# Explanatory intent. These veto the "contains digits -> numeric" heuristic: a question
# like "explain why the Bengal famine of 1943 happened" contains a year but wants a
# coherence block, not an atom. Without this veto any question mentioning a date gets
# routed to the precision layer and loses the context that actually answers it.
_DESCRIPTIVE_RE = re.compile(
    r"\b(explain|describe|why|how\s+(?:does|did|do|is|are|was|were)|"
    r"what\s+(?:happened|caused|led\s+to)|causes?\s+of|reasons?\s+for|"
    r"impact\s+of|effects?\s+of|significance\s+of|in\s+detail|overview|summar)"
    r"|क्यों|कैसे\s+होता|व्याख्या|कारण"
    r"|கேன்|ஏன்|எப்படி\s+நடந்தது|காரணம்"
    r"|কেন|কীভাবে\s+ঘটে|কারণ",
    re.IGNORECASE | re.UNICODE,
)

# layer -> weight. Weights are relative depth *and* fusion priors.
_PROFILES: dict[QueryShape, dict[Layer, float]] = {
    QueryShape.NUMERIC: {Layer.ATOMIC: 0.52, Layer.SEMANTIC: 0.24, Layer.WINDOW: 0.10, Layer.DOCUMENT: 0.14},
    QueryShape.FACTOID: {Layer.ATOMIC: 0.48, Layer.SEMANTIC: 0.30, Layer.WINDOW: 0.10, Layer.DOCUMENT: 0.12},
    QueryShape.YESNO: {Layer.ATOMIC: 0.44, Layer.SEMANTIC: 0.34, Layer.WINDOW: 0.10, Layer.DOCUMENT: 0.12},
    QueryShape.DEFINITION: {Layer.ATOMIC: 0.22, Layer.SEMANTIC: 0.32, Layer.WINDOW: 0.10, Layer.DOCUMENT: 0.36},
    QueryShape.DESCRIPTION: {Layer.ATOMIC: 0.12, Layer.SEMANTIC: 0.42, Layer.WINDOW: 0.26, Layer.DOCUMENT: 0.20},
    QueryShape.LIST: {Layer.ATOMIC: 0.14, Layer.SEMANTIC: 0.38, Layer.WINDOW: 0.28, Layer.DOCUMENT: 0.20},
    QueryShape.ENTITY: {Layer.ATOMIC: 0.34, Layer.SEMANTIC: 0.28, Layer.WINDOW: 0.12, Layer.DOCUMENT: 0.26},
    QueryShape.UNKNOWN: {Layer.ATOMIC: 0.30, Layer.SEMANTIC: 0.32, Layer.WINDOW: 0.18, Layer.DOCUMENT: 0.20},
}

# Sparse retrieval matters more for identifiers and numbers, less for prose.
_SPARSE_BIAS: dict[QueryShape, float] = {
    QueryShape.NUMERIC: 0.46,
    QueryShape.ENTITY: 0.50,
    QueryShape.FACTOID: 0.40,
    QueryShape.YESNO: 0.36,
    QueryShape.DEFINITION: 0.34,
    QueryShape.LIST: 0.30,
    QueryShape.DESCRIPTION: 0.26,
    QueryShape.UNKNOWN: settings.sparse_weight,
}


class QueryRouter:
    def __init__(self, dense_k: int | None = None) -> None:
        self.dense_k = dense_k or settings.dense_k

    # ----------------------------------------------------------------- shaping
    def classify(self, query: str) -> tuple[QueryShape, list[str]]:
        toks = tokenize(query)
        low = query.casefold()
        n = len(toks)
        tokset = set(toks)
        reasons: list[str] = []

        has_wh = bool(tokset & WH_WORDS)
        has_digit = bool(_DIGIT_RE.search(query))

        # Token cues match tokens; phrase cues match the raw string. Mixing the two is
        # how "how many" turns into a bare "how" cue and mis-routes every "how does…".
        numeric_cue = bool(tokset & NUMERIC_CUES) or any(p in low for p in NUMERIC_PHRASES)
        # An explicit enumeration noun ("symptoms", "types", "causes") is a stronger signal
        # than the phrase "what are the", which opens plenty of non-list questions.
        list_token = bool(tokset & LIST_CUES)
        list_phrase = any(p in low for p in LIST_PHRASES)
        # A definition phrase only counts when it *opens* the query: "what is a corporation"
        # is a definition, "explain what is happening in the diagram" is not.
        def_cue = bool(tokset & DEFINITION_CUES) or low.startswith(DEFINITION_PHRASES)

        yesno = n > 0 and toks[0] in YESNO_CUES and not has_wh
        # Explanatory intent. This vetoes the "contains digits -> numeric" heuristic:
        # "explain why the Bengal famine of 1943 happened" carries a year but wants a
        # coherence block, not an atom. Without the veto, any question mentioning a date
        # is routed to the precision layer and loses the context that answers it.
        descriptive_cue = bool(_DESCRIPTIVE_RE.search(query))
        caps = len(_CAP_RE.findall(query))
        content = content_tokens(query)
        entity_density = (caps / max(n, 1)) if n else 0.0

        if descriptive_cue and not numeric_cue:
            reasons.append(
                "explanatory cue" + (" (digits present but not the ask)" if has_digit else "")
            )
            shape = QueryShape.DESCRIPTION
        elif numeric_cue or (has_digit and n <= 12):
            reasons.append("numeric cue" if numeric_cue else "contains digits")
            shape = QueryShape.NUMERIC
        elif list_token:
            reasons.append("enumeration noun")
            shape = QueryShape.LIST
        elif def_cue:
            reasons.append("definition cue")
            shape = QueryShape.DEFINITION
        elif list_phrase:
            reasons.append("enumeration phrasing")
            shape = QueryShape.LIST
        elif yesno:
            reasons.append("leading auxiliary verb")
            shape = QueryShape.YESNO
        elif has_wh and n >= 9:
            reasons.append(f"open question, {n} tokens")
            shape = QueryShape.DESCRIPTION
        elif has_wh:
            reasons.append(f"wh-question, {n} tokens")
            shape = QueryShape.FACTOID
        elif entity_density >= 0.34 or (not has_wh and len(content) <= 3 and n <= 4):
            reasons.append("entity-dominant, no interrogative")
            shape = QueryShape.ENTITY
        elif n >= 10:
            reasons.append(f"long statement, {n} tokens")
            shape = QueryShape.DESCRIPTION
        else:
            reasons.append("no decisive signal")
            shape = QueryShape.UNKNOWN
        return shape, reasons

    # ------------------------------------------------------------------ routing
    def route(
        self,
        query: str,
        *,
        lang_hint: str | None = None,
        available_layers: set[Layer] | None = None,
        dense_k: int | None = None,
    ) -> RoutePlan:
        shape, reasons = self.classify(query)
        toks = tokenize(query)
        script = detect_script(query)
        lang = detect_lang(query, lang_hint)

        profile = _PROFILES[shape]
        if available_layers:
            profile = {lay: w for lay, w in profile.items() if lay in available_layers} or profile
        total = sum(profile.values()) or 1.0

        budget_k = dense_k or self.dense_k
        layers: list[LayerPlan] = []
        for layer, weight in sorted(profile.items(), key=lambda kv: -kv[1]):
            norm = weight / total
            # Everything gets a floor of 4 so no layer is silently unreachable.
            k = max(4, int(round(budget_k * norm)))
            layers.append(LayerPlan(layer=layer, k=k, weight=round(norm, 4)))

        sparse_w = _SPARSE_BIAS[shape]
        if script != "latin":
            # Indic morphology inflates surface forms, so lexical matching is a
            # weaker signal than in English; lean a little harder on the encoder.
            sparse_w = max(0.18, sparse_w - 0.08)
            reasons.append(f"{script} script - dense-leaning fusion")
        dense_w = round(1.0 - sparse_w, 4)

        rerank = len(toks) >= 2
        expand = shape in {
            QueryShape.NUMERIC, QueryShape.FACTOID, QueryShape.YESNO, QueryShape.ENTITY
        }
        if expand:
            reasons.append("fine-grained shape - parent expansion on")

        return RoutePlan(
            shape=shape,
            script=script,
            lang=lang,
            layers=layers,
            dense_weight=dense_w,
            sparse_weight=round(sparse_w, 4),
            rerank=rerank,
            expand_parents=expand,
            token_count=len(toks),
            reason="; ".join(reasons),
        )


__all__ = ["QueryRouter"]
