"""The guardrail stack: gate ordering, the trust score, and safe completions.

Gates are grouped by stage and short-circuit on the first BLOCK, so an unsafe query
never reaches the index and an unverifiable answer never reaches the user.

The trust score
---------------
A single 0-100 number is not a guardrail, but a stack of five verdicts with different
units is not something a user can read at a glance either. So the stack also publishes
a composite, built from the signals that independently predict whether an answer is
worth believing:

    40  claim coverage from the groundedness gate     (is it attributable?)
    18  retrieval top score                           (did we find anything good?)
    12  retrieval margin                              (was it distinctly good?)
    10  weakest individual claim                      (is any part unsupported?)
    10  citation density                              (multiple independent sources?)
    10  safety and injection clearance                (is the request itself clean?)

Every component is shown in the UI next to the composite, so the number is
decomposable rather than a vibe. Below ``trust_abstain_below`` the answer is withheld
even when no single gate blocked it - the composite is the last line of defence for
answers that are individually plausible on every axis and collectively unconvincing.
"""

from __future__ import annotations

import numpy as np

from ..answer.compose import (
    INJECTION,
    LOW_GROUNDING,
    NO_EVIDENCE,
    OFF_TOPIC,
    SELF_HARM,
    UNSAFE,
    localized,
)
from ..config import settings
from ..harness.contracts import Decision, GuardReport, GuardVerdict
from ..index.corpus import CorpusIndex
from .admission import InjectionGate, SafetyGate, SanityGate
from .base import GateContext
from .groundedness import GroundednessGate
from .redaction import PIIGate
from .topicality import TopicalityGate


class GuardrailStack:
    def __init__(self, index: CorpusIndex) -> None:
        self.index = index
        self.sanity = SanityGate()
        self.injection = InjectionGate()
        self.safety = SafetyGate()
        self.topicality = TopicalityGate()
        self.groundedness = GroundednessGate(index)
        self.pii = PIIGate()

    def warm(self) -> None:
        self.safety.warm()

    @property
    def gate_names(self) -> list[str]:
        return [
            self.sanity.name, self.injection.name, self.safety.name,
            self.topicality.name, self.groundedness.name, self.pii.name,
        ]

    # ==================================================================== stages
    def run_admission(self, ctx: GateContext, report: GuardReport) -> GuardVerdict | None:
        """Vector-free gates. Run first so junk and injection cost well under a millisecond."""
        for gate in (self.sanity, self.injection):
            v = report.add(gate.check(ctx))
            if v.decision is Decision.BLOCK:
                report.safe_message = self._safe_message(v, ctx.lang)
                return v
        return None

    def run_safety(self, ctx: GateContext, report: GuardReport) -> GuardVerdict | None:
        """Semantic harm classification. Requires the query vector, so it runs after embedding.

        Ordering matters here. The lexical patterns could run earlier, but the centroid
        classifier is the part that generalises across 15 languages and across paraphrase,
        and it needs the same query vector retrieval is about to use. Placing this gate
        immediately after ``embed_query`` and before ``retrieve`` costs one dot product
        against seven vectors and still blocks an unsafe request before it touches the index.
        """
        v = report.add(self.safety.check(ctx))
        if v.decision is Decision.BLOCK:
            report.safe_message = self._safe_message(v, ctx.lang)
            return v
        return None

    def run_retrieval(self, ctx: GateContext, report: GuardReport) -> GuardVerdict | None:
        v = report.add(self.topicality.check(ctx))
        if v.decision is Decision.BLOCK:
            report.safe_message = localized(NO_EVIDENCE, ctx.lang)
            return v
        return None

    def run_output(self, ctx: GateContext, report: GuardReport) -> GuardVerdict | None:
        v = report.add(self.groundedness.check(ctx))
        report.groundedness = v.score
        if ctx.answer is not None:
            ctx.answer.claims = list(self.groundedness.claims)
        if v.decision is Decision.BLOCK:
            report.safe_message = localized(LOW_GROUNDING, ctx.lang)
            return v

        p = report.add(self.pii.check(ctx))
        if p.decision is not Decision.ALLOW and self.pii.redacted_text and ctx.answer is not None:
            ctx.answer.text = self.pii.redacted_text
        return None

    # =============================================================== trust score
    def score(self, ctx: GateContext, report: GuardReport) -> float:
        res = ctx.retrieval
        draft = ctx.answer

        coverage = float(report.groundedness)
        top = float(res.top_score) if res else 0.0
        margin = float(res.margin) if res else 0.0
        weakest = min((c.support for c in (draft.claims if draft else [])), default=coverage)
        n_sources = len({c.doc_id for c in draft.citations}) if draft else 0

        safety_clear = 1.0
        for v in report.verdicts:
            if v.gate in {"input.safety", "input.injection"} and v.decision is not Decision.ALLOW:
                safety_clear = 0.35 if v.decision is Decision.FLAG else 0.0

        composite = (
            40.0 * float(np.clip(coverage, 0.0, 1.0))
            + 18.0 * float(np.clip(top / 0.85, 0.0, 1.0))
            + 12.0 * float(np.clip(margin / 0.18, 0.0, 1.0))
            + 10.0 * float(np.clip(weakest, 0.0, 1.0))
            + 10.0 * float(np.clip(n_sources / 2.0, 0.0, 1.0))
            + 10.0 * safety_clear
        )
        report.trust_score = round(float(np.clip(composite, 0.0, 100.0)), 2)
        return report.trust_score

    def components(self, ctx: GateContext, report: GuardReport) -> dict[str, float]:
        res = ctx.retrieval
        draft = ctx.answer
        weakest = min((c.support for c in (draft.claims if draft else [])), default=report.groundedness)
        return {
            "groundedness": round(float(report.groundedness), 4),
            "retrieval_top": round(float(res.top_score) if res else 0.0, 4),
            "retrieval_margin": round(float(res.margin) if res else 0.0, 4),
            "weakest_claim": round(float(weakest), 4),
            "distinct_sources": len({c.doc_id for c in draft.citations}) if draft else 0,
        }

    # ================================================================= internals
    @staticmethod
    def _safe_message(v: GuardVerdict, lang: str) -> str:
        if v.gate == "input.injection":
            return localized(INJECTION, lang)
        if v.gate == "input.safety":
            if v.detail.get("category") == "self_harm":
                return localized(SELF_HARM, lang)
            return localized(UNSAFE, lang)
        if v.gate == "input.sanity":
            return localized(OFF_TOPIC, lang)
        return localized(OFF_TOPIC, lang)

    @staticmethod
    def abstain_below_trust(report: GuardReport, lang: str) -> bool:
        if report.trust_score >= settings.trust_abstain_below:
            return False
        report.safe_message = localized(LOW_GROUNDING, lang)
        report.verdicts.append(
            GuardVerdict(
                gate="output.trust-floor",
                stage="output",
                decision=Decision.BLOCK,
                score=report.trust_score / 100.0,
                threshold=settings.trust_abstain_below / 100.0,
                reason=(
                    f"composite trust {report.trust_score:.1f} below the "
                    f"{settings.trust_abstain_below:.0f} floor - withholding a technically "
                    "citable but unconvincing answer"
                ),
            )
        )
        report.blocked = True
        report.blocked_by = "output.trust-floor"
        return True


__all__ = ["GuardrailStack"]
