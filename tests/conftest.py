"""Test fixtures.

The tests are built around a tiny synthetic corpus rather than the real index, so the
suite runs in seconds without a download and without depending on which slice of MS
MARCO-XI happens to be ingested. The encoder is real (it has to be, for the semantic
chunker and the safety centroids to mean anything) and is shared across the session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))


@pytest.fixture(scope="session")
def encoder():
    from setu.embed import get_encoder

    enc = get_encoder()
    enc.warm()
    return enc


@pytest.fixture(scope="session")
def docs():
    """Bilingual, multi-sentence, with a numeric fact and a PII string to redact."""
    from setu.chunking.base import Document

    def d(doc_id, text, lang, qtype, selected=False, rank=0, query=""):
        return Document(
            doc_id=doc_id,
            text=text,
            lang=lang,
            meta={
                "query_id": doc_id.split(":")[0],
                "query": query,
                "query_type": qtype,
                "is_selected": selected,
                "passage_rank": rank,
                "passage_count": 3,
                "source": "test",
            },
        )

    return [
        d(
            "1:0:en",
            "A banana contains about 89 calories per 100 grams. Most of that energy comes from "
            "carbohydrates, mainly natural sugars and starch. Bananas are also a good source of "
            "potassium and vitamin B6. They are one of the most widely consumed fruits worldwide.",
            "en", "NUMERIC", True, 0, "how many calories in a banana",
        ),
        d(
            "1:0:hi",
            "एक केले में प्रति 100 ग्राम लगभग 89 कैलोरी होती हैं। इसकी अधिकांश ऊर्जा कार्बोहाइड्रेट से "
            "आती है, मुख्य रूप से प्राकृतिक शर्करा और स्टार्च से। केला पोटैशियम और विटामिन बी6 का भी "
            "अच्छा स्रोत है। यह दुनिया में सबसे अधिक खाया जाने वाला फल है।",
            "hi", "NUMERIC", True, 0, "एक केले में कितनी कैलोरी होती हैं",
        ),
        d(
            "2:0:en",
            "A corporation is a legal entity that is separate and distinct from its owners. "
            "Corporations enjoy most of the rights and responsibilities that individuals possess: "
            "they can enter contracts, borrow money, sue and be sued, and own assets. "
            "The key feature is limited liability, which means shareholders are not personally "
            "responsible for company debts.",
            "en", "DESCRIPTION", True, 0, "what is a corporation",
        ),
        d(
            "3:0:en",
            "Photosynthesis is the process by which green plants convert light energy into chemical "
            "energy. Chlorophyll in the leaves absorbs sunlight, and the plant combines carbon "
            "dioxide with water to produce glucose and oxygen. The oxygen is released into the "
            "atmosphere as a by-product.",
            "en", "DESCRIPTION", True, 0, "how does photosynthesis work",
        ),
        d(
            "4:0:en",
            "For billing questions contact support at billing@example.com or call 9876543210 during "
            "business hours. Our office is located in Panaji, Goa. Refunds are processed within "
            "seven working days of an approved request.",
            "en", "ENTITY", True, 0, "how do i contact billing support",
        ),
        d(
            "5:0:en",
            "Naloxone is used to reverse an opioid overdose. Warning signs of an overdose include "
            "slow or stopped breathing, pinpoint pupils, and unresponsiveness. Anyone who suspects "
            "an overdose should call emergency services immediately.",
            "en", "DESCRIPTION", True, 0, "what are the symptoms of an opioid overdose",
        ),
    ]


@pytest.fixture(scope="session")
def index(docs, encoder):
    from setu.chunking.lattice import ChunkLattice
    from setu.index.corpus import CorpusIndex

    lattice = ChunkLattice(encoder)
    groups = list(lattice.build(list(docs)))
    idx = CorpusIndex().build(list(docs), groups, encoder)
    idx.lattice_stats = lattice.stats  # type: ignore[attr-defined]
    return idx


@pytest.fixture(scope="session")
def pipeline(index):
    from setu.pipeline import SetuPipeline

    p = SetuPipeline(index)
    p.warm()
    return p


@pytest.fixture
def ask(pipeline):
    """Convenience: await ask("question") -> TurnResult."""
    from setu.harness.contracts import AnswerMode, AskRequest

    async def _ask(text: str, lang: str | None = None, mode: str = "reflex", **kw):
        return await pipeline.ask(
            AskRequest(text=text, lang=lang, mode=AnswerMode(mode), **kw)
        )

    return _ask
