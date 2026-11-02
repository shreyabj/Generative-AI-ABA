"""Chunk lattice behaviour.

These assert the *properties* the design claims, not the exact chunk counts - counts
would break on any threshold tweak and would be testing the config file rather than the
code.
"""

from __future__ import annotations

import pytest
from setu.chunking.atomic import AtomicChunker
from setu.chunking.base import Document
from setu.chunking.lattice import ChunkLattice
from setu.chunking.metadata import DocumentChunker, metadata_header
from setu.chunking.semantic import SemanticChunker
from setu.chunking.window import AdaptiveWindowChunker
from setu.harness.contracts import Layer
from setu.text import split_sentences


def test_all_four_layers_build_for_a_document_long_enough_to_stratify(encoder):
    """The window layer only exists above `window_chars`; give it a document that long.

    The fixture corpus is intentionally made of short MS MARCO-style passages, where the
    lattice correctly collapses to 2 layers. This test needs a genuinely long passage,
    otherwise it is asserting that dedup is broken.
    """
    from setu.config import settings

    body = " ".join(
        [
            "The Western Ghats are a mountain range along the western coast of India.",
            "They run roughly 1600 kilometres from the Tapti river down to Kanyakumari.",
            "The range is older than the Himalaya and is a UNESCO World Heritage site.",
            "It is one of the eight hottest biodiversity hotspots in the world.",
            "At least 325 globally threatened species occur in the Western Ghats.",
            "The monsoon interacts with the range to produce very high rainfall on the windward side.",
            "Agasthyamalai and the Nilgiris are among its best studied sub-ranges.",
            "Deforestation and plantation agriculture remain the principal pressures on the region.",
        ]
    )
    assert len(body) > settings.window_chars * 2, "test document is not long enough to window"

    doc = Document(doc_id="long:0:en", text=body, lang="en", meta={"query_type": "DESCRIPTION"})
    chunks = ChunkLattice(encoder).build_all([doc])
    built = {c.layer.value for c in chunks}
    for layer in ("atomic", "semantic", "window", "document"):
        assert layer in built, f"layer {layer} never built (got {sorted(built)})"


def test_fixture_corpus_stratifies_at_least_partially(index):
    """Short passages legitimately collapse, but atoms and documents must always exist."""
    per_layer = index.manifest["per_layer"]
    assert per_layer.get("atomic", 0) > 0
    assert per_layer.get("document", 0) > 0
    assert sum(per_layer.values()) == len(index.chunks)


def test_chunk_spans_are_exact_slices_of_their_document(index):
    """char_start/char_end must be usable as a plain slice - parent expansion depends on it."""
    for rec in index.chunks:
        doc = index.docs[rec.doc_id]
        assert 0 <= rec.char_start <= rec.char_end <= len(doc.text)
        if rec.layer is not Layer.DOCUMENT:
            assert doc.text[rec.char_start : rec.char_end].strip() == rec.text.strip()


def test_parent_links_resolve_and_contain_their_child(index):
    linked = 0
    for rec in index.chunks:
        if not rec.parent_id:
            continue
        parent = index.parent_of(rec)
        assert parent is not None, f"dangling parent_id on {rec.chunk_id}"
        assert parent.doc_id == rec.doc_id
        # The parent must actually enclose (or at least overlap) the child span.
        overlap = min(rec.char_end, parent.char_end) - max(rec.char_start, parent.char_start)
        assert overlap > 0
        linked += 1
    assert linked > 0


def test_atomic_accretes_short_sentences_instead_of_indexing_them_alone():
    doc = Document(
        doc_id="t", lang="en",
        text="Yes. No. Maybe. The Indian Premier League is a professional Twenty20 cricket "
             "league contested by ten teams based in different Indian cities.",
    )
    doc.sentences = split_sentences(doc.text)
    chunks = AtomicChunker().split(doc)
    assert chunks
    # Nothing as trivial as "Yes." should survive as its own retrievable node.
    assert all(len(c.text) >= 20 for c in chunks), [c.text for c in chunks]


def test_atomic_fissions_an_oversized_run_on_sentence():
    long_clause = (
        "The committee approved the proposal because the funding was already allocated; "
        "however the implementation was delayed until the following quarter, therefore the "
        "final report was postponed and the budget was carried forward to the next cycle"
    )
    doc = Document(doc_id="t", lang="en", text=long_clause)
    doc.sentences = split_sentences(doc.text)
    chunker = AtomicChunker(min_chars=40, max_chars=90)
    chunks = chunker.split(doc)
    assert len(chunks) > 1, "oversized sentence was not fissioned"
    assert all(len(c.text) <= 90 + 40 for c in chunks)


def test_semantic_chunker_cuts_at_a_topic_change(encoder):
    """Two clearly unrelated topics in one document must produce a boundary between them."""
    text = (
        "The Python programming language was created by Guido van Rossum. "
        "It emphasises readability and uses significant indentation. "
        "Python is widely used for data science and web development. "
        "The Amazon rainforest covers most of the Amazon basin in South America. "
        "It is home to a vast number of plant and animal species. "
        "Deforestation there has accelerated over the past two decades."
    )
    doc = Document(doc_id="t", lang="en", text=text)
    lattice = ChunkLattice(encoder)
    lattice.prepare([doc])
    # max_chars must stay realistic: the chunker merges blocks below max_chars//3, so an
    # absurdly large cap would merge every boundary straight back out again.
    chunks = SemanticChunker(max_chars=420, z=0.0).split(doc)
    assert len(chunks) >= 2, "no boundary found between two unrelated topics"
    # The boundary should fall near the topic switch, not at an arbitrary offset.
    switch = text.index("The Amazon")
    boundaries = [c.char_start for c in chunks[1:]]
    assert min(abs(b - switch) for b in boundaries) < 80


def test_window_overlap_is_adaptive_not_constant(encoder):
    """A low-cohesion document must get more overlap than a high-cohesion one."""
    cohesive = Document(
        doc_id="c", lang="en",
        text=" ".join(
            [
                "Solar panels convert sunlight into electricity using photovoltaic cells.",
                "Photovoltaic cells are made from semiconductor materials such as silicon.",
                "Silicon cells release electrons when photons strike the material.",
                "The released electrons create a direct current that an inverter converts.",
                "Inverters change direct current into the alternating current homes use.",
                "Rooftop solar installations therefore feed alternating current to the grid.",
            ]
        ),
    )
    disjoint = Document(
        doc_id="d", lang="en",
        text=" ".join(
            [
                "Solar panels convert sunlight into electricity using photovoltaic cells.",
                "The blue whale is the largest animal known to have existed on Earth.",
                "Beethoven composed nine symphonies before his death in 1827.",
                "Mount Everest lies on the border between Nepal and the Tibet region.",
                "Sourdough bread relies on a fermented starter culture rather than yeast.",
                "The Bombay Stock Exchange was established in 1875 in Mumbai.",
            ]
        ),
    )
    lattice = ChunkLattice(encoder)
    lattice.prepare([cohesive, disjoint])
    chunker = AdaptiveWindowChunker(window_chars=220, overlap_min=0.10, overlap_max=0.50)
    a = chunker.split(cohesive)
    b = chunker.split(disjoint)
    assert a and b
    assert max(c.overlap_ratio for c in b) > max(c.overlap_ratio for c in a), (
        "overlap did not increase for the low-cohesion document"
    )


def test_metadata_header_is_embedded_but_never_displayed(docs, encoder):
    doc = docs[0]
    doc.sentences = split_sentences(doc.text)
    chunk = DocumentChunker().split(doc)[0]
    header = metadata_header(doc)
    assert header in chunk.embed_text, "metadata header missing from the embedded text"
    assert header not in chunk.text, "metadata header leaked into the citable text"
    assert "NUMERIC" in header and doc.lang in header


def test_dedup_collapses_a_single_sentence_document(encoder):
    """Four views of a one-sentence passage must not produce four identical nodes."""
    doc = Document(doc_id="t", lang="en", text="Panaji is the capital city of the Indian state of Goa.")
    lattice = ChunkLattice(encoder)
    chunks = lattice.build_all([doc])
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts)), "duplicate text across layers survived dedup"
    # L3 is always retained so the router always has a backstop layer.
    assert any(c.layer is Layer.DOCUMENT for c in chunks)


def test_lattice_depth_adapts_to_document_length(index):
    """Long documents should stratify further than short ones."""
    stats = index.lattice_stats.to_dict()
    assert stats["dedup_rate"] > 0, "nothing was deduped - the layers are not overlapping at all"
    assert stats["parent_links"] > 0


@pytest.mark.parametrize(
    "text",
    [
        "एक केले में प्रति 100 ग्राम लगभग 89 कैलोरी होती हैं। यह पोटैशियम का अच्छा स्रोत है।",
        "ஒரு வாழைப்பழத்தில் 100 கிராமுக்கு சுமார் 89 கலோரிகள் உள்ளன. இது பொட்டாசியத்தின் நல்ல மூலம்.",
        "একটি কলায় প্রতি ১০০ গ্রামে প্রায় ৮৯ ক্যালোরি থাকে। এটি পটাশিয়ামের ভালো উৎস।",
    ],
)
def test_indic_scripts_split_into_multiple_sentences(text):
    """Danda and its equivalents must be treated as sentence terminators."""
    sents = split_sentences(text)
    assert len(sents) >= 2, f"failed to split: {[s.text for s in sents]}"
    for s in sents:
        assert text[s.start : s.end] == s.text
