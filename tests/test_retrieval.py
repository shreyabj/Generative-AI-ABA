"""Router, BM25, fusion and the anytime retrieval engine."""

from __future__ import annotations

import numpy as np
import pytest
from setu.harness.contracts import Layer, QueryShape
from setu.harness.tracing import LatencyBudget, Trace
from setu.index.bm25 import BM25Index
from setu.index.fusion import RankFuser
from setu.retrieval.router import QueryRouter


# ------------------------------------------------------------------------ router
@pytest.mark.parametrize(
    "query,expected",
    [
        ("how many calories in a banana", QueryShape.NUMERIC),
        ("what is the average salary of a nurse", QueryShape.NUMERIC),
        ("what is a corporation", QueryShape.DEFINITION),
        ("define inflation", QueryShape.DEFINITION),
        ("list the types of renewable energy", QueryShape.LIST),
        ("is photosynthesis an endothermic reaction", QueryShape.YESNO),
        ("why did the roman empire collapse and what replaced it afterwards", QueryShape.DESCRIPTION),
        ("who invented the telephone", QueryShape.FACTOID),
    ],
)
def test_router_classifies_query_shape(query, expected):
    assert QueryRouter().classify(query)[0] is expected


def test_explanatory_intent_beats_an_incidental_year():
    """Regression: "…famine of 1943…" used to route as NUMERIC because it contains digits."""
    plan = QueryRouter().route("explain in detail why the bengal famine of 1943 happened")
    assert plan.shape is QueryShape.DESCRIPTION, plan.reason


def test_router_prefers_fine_layers_for_numeric_and_coarse_for_descriptive():
    r = QueryRouter()
    numeric = r.route("how many calories in a banana")
    descriptive = r.route("explain in detail why the bengal famine of 1943 happened")

    def weight(plan, layer):
        return next(lp.weight for lp in plan.layers if lp.layer is layer)

    assert weight(numeric, Layer.ATOMIC) > weight(descriptive, Layer.ATOMIC)
    assert weight(descriptive, Layer.SEMANTIC) > weight(numeric, Layer.SEMANTIC)


def test_router_always_plans_every_available_layer():
    """No layer may be silently unreachable - every plan needs a document backstop."""
    plan = QueryRouter().route("how many calories in a banana")
    assert {lp.layer for lp in plan.layers} == set(Layer)
    assert all(lp.k >= 4 for lp in plan.layers)


def test_router_leans_dense_for_indic_scripts():
    latin = QueryRouter().route("how many calories in a banana")
    deva = QueryRouter().route("एक केले में कितनी कैलोरी होती हैं")
    assert deva.sparse_weight < latin.sparse_weight
    assert deva.script == "devanagari"
    assert deva.lang == "hi"


def test_router_gives_a_reason_for_every_decision():
    for q in ["how many calories", "what is a corporation", "Mount Everest"]:
        assert QueryRouter().route(q).reason.strip()


def test_router_weights_are_normalised():
    plan = QueryRouter().route("what causes high blood pressure")
    assert abs(sum(lp.weight for lp in plan.layers) - 1.0) < 1e-3
    assert abs(plan.dense_weight + plan.sparse_weight - 1.0) < 1e-3


# -------------------------------------------------------------------------- bm25
def test_bm25_ranks_the_exact_lexical_match_first():
    corpus = [
        "A banana contains about 89 calories per 100 grams of fruit.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "A corporation is a legal entity separate from its owners.",
        "Potassium and magnesium are essential dietary minerals.",
    ]
    idx = BM25Index().fit(corpus)
    ids, scores = idx.search("banana calories", k=3)
    assert ids.size > 0
    assert ids[0] == 0
    assert scores[0] > 0


def test_bm25_returns_nothing_for_out_of_vocabulary_queries():
    idx = BM25Index().fit(["alpha beta gamma", "delta epsilon zeta"])
    ids, scores = idx.search("kryptonite unobtainium", k=5)
    assert ids.size == 0 and scores.size == 0


def test_bm25_handles_indic_tokens():
    idx = BM25Index().fit(
        ["एक केले में लगभग 89 कैलोरी होती हैं", "प्रकाश संश्लेषण पौधों में होता है"]
    )
    ids, _ = idx.search("केले कैलोरी", k=2)
    assert ids.size > 0 and ids[0] == 0


def test_bm25_candidate_mask_restricts_scoring():
    idx = BM25Index().fit(["banana calories", "banana potassium", "corporation law"])
    mask = np.array([False, True, True])
    ids, _ = idx.search("banana", k=3, candidate_mask=mask)
    assert 0 not in ids.tolist()


def test_bm25_survives_a_save_load_round_trip(tmp_path):
    corpus = ["banana calories energy", "corporation legal entity", "photosynthesis chlorophyll"]
    a = BM25Index().fit(corpus)
    a.save(tmp_path)
    b = BM25Index.load(tmp_path)
    ia, sa = a.search("banana energy", k=3)
    ib, sb = b.search("banana energy", k=3)
    assert ia.tolist() == ib.tolist()
    assert np.allclose(sa, sb)


# ------------------------------------------------------------------------ fusion
def test_rrf_is_scale_free():
    """A BM25 score of 41 must not swamp a cosine of 0.81 - fusion is on rank.

    The invariant is that the *magnitude* of the sparse score is irrelevant: multiplying
    every BM25 score by 1000 must not change the ordering. That is the property naive
    score addition violates.
    """
    dense = [(np.array([1, 2, 3]), np.array([0.81, 0.79, 0.70], dtype=np.float32), 1.0)]
    small = (np.array([9, 8, 7]), np.array([41.0, 22.0, 11.0], dtype=np.float32))
    huge = (np.array([9, 8, 7]), np.array([41000.0, 22000.0, 11000.0], dtype=np.float32))

    order_small = [c.idx for c in RankFuser().fuse(dense, small, top_k=6)]
    order_huge = [c.idx for c in RankFuser().fuse(dense, huge, top_k=6)]
    assert order_small == order_huge, "fusion order changed with sparse score magnitude"

    # The dense top-1 leads (dense is weighted higher by default), and the sparse-only
    # top-1 is still retained in the pool rather than being crushed out of it.
    assert order_small[0] == 1
    assert 9 in order_small


def test_fusion_rewards_agreement_between_retrievers():
    """A document both retrievers rank first must outrank one only one of them found."""
    dense = [(np.array([5, 1]), np.array([0.9, 0.85], dtype=np.float32), 1.0)]
    sparse = (np.array([5, 2]), np.array([30.0, 28.0], dtype=np.float32))
    cands = RankFuser().fuse(dense, sparse, top_k=4)
    assert cands[0].idx == 5
    assert {"dense", "sparse"} == cands[0].sources


def test_fusion_preserves_per_retriever_provenance():
    dense = [(np.array([3]), np.array([0.77], dtype=np.float32), 1.0)]
    sparse = (np.array([3]), np.array([12.5], dtype=np.float32))
    c = RankFuser().fuse(dense, sparse, top_k=1)[0]
    assert c.dense_score == pytest.approx(0.77, abs=1e-4)
    assert c.sparse_score == pytest.approx(12.5, abs=1e-4)
    assert c.rank_dense == 1 and c.rank_sparse == 1


def test_fusion_of_nothing_is_empty():
    assert RankFuser().fuse([], None) == []


# ------------------------------------------------------------------------ engine
def test_engine_returns_diverse_evidence(index, encoder):
    """Five citations should be five different pieces of evidence, not one passage sliced."""
    from setu.retrieval.engine import RetrievalEngine

    q = "how many calories are in a banana"
    plan = QueryRouter().route(q)
    res = RetrievalEngine(index).search(
        q, encoder.encode_query(q), plan,
        budget=LatencyBudget(10_000), trace=Trace(), top_k=5,
    )
    assert res.hits
    spans = [(h.doc_id, h.char_start, h.char_end) for h in res.hits]
    assert len(spans) == len(set(spans))
    # No document may dominate the whole result set.
    from collections import Counter

    assert max(Counter(h.doc_id for h in res.hits).values()) <= 2


def test_engine_finds_the_relevant_document(index, encoder):
    from setu.retrieval.engine import RetrievalEngine

    q = "how many calories are in a banana"
    plan = QueryRouter().route(q)
    res = RetrievalEngine(index).search(
        q, encoder.encode_query(q), plan,
        budget=LatencyBudget(10_000), trace=Trace(), top_k=5,
    )
    assert any(h.doc_id.startswith("1:") for h in res.hits)


def test_engine_retrieves_across_languages(index, encoder):
    """A Hindi question must be able to reach the paired English passage."""
    from setu.retrieval.engine import RetrievalEngine

    q = "एक केले में कितनी कैलोरी होती हैं"
    plan = QueryRouter().route(q)
    res = RetrievalEngine(index).search(
        q, encoder.encode_query(q), plan,
        budget=LatencyBudget(10_000), trace=Trace(), top_k=8,
    )
    langs = {h.lang for h in res.hits}
    assert "hi" in langs
    assert any(h.doc_id.startswith("1:") for h in res.hits)


def test_engine_skips_optional_stages_when_the_budget_is_exhausted(index, encoder):
    """The anytime property: an impossible deadline degrades quality, not the deadline."""
    from setu.retrieval.engine import RetrievalEngine

    q = "how many calories are in a banana"
    plan = QueryRouter().route(q)
    budget = LatencyBudget(0.001)  # already expired
    trace = Trace()
    res = RetrievalEngine(index).search(
        q, encoder.encode_query(q), plan, budget=budget, trace=trace, top_k=3
    )
    assert budget.skipped, "no stage was skipped despite an exhausted budget"
    assert budget.degraded
    assert res.truncated_by_budget
    # Crucially, it still produced an answerable result set.
    assert res.hits


def test_engine_reports_top_score_and_margin(index, encoder):
    from setu.retrieval.engine import RetrievalEngine

    q = "what is a corporation"
    plan = QueryRouter().route(q)
    res = RetrievalEngine(index).search(
        q, encoder.encode_query(q), plan,
        budget=LatencyBudget(10_000), trace=Trace(), top_k=5,
    )
    assert res.top_score > 0
    assert res.margin >= 0
    assert res.dense_candidates > 0
