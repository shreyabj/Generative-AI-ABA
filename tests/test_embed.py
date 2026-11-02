"""The encoder.

The ordering test here is the important one. ``Encoder.encode`` sorts by length internally to
avoid padding waste, and if the inverse permutation were wrong every chunk in the index would be
paired with the wrong vector - a failure that produces plausible-looking nonsense rather than an
exception, and would be invisible in every other test.
"""

from __future__ import annotations

import numpy as np


class TestNormalisation:
    def test_vectors_are_unit_length(self, encoder):
        v = encoder.encode(["alpha beta gamma", "a much longer passage about corporations"])
        norms = np.linalg.norm(v, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_query_vector_is_one_dimensional(self, encoder):
        q = encoder.encode_query("what is a corporation")
        assert q.ndim == 1
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-4


class TestBucketedOrder:
    def test_order_is_preserved_across_length_buckets(self, encoder):
        """Deliberately adversarial ordering: long, short, long, short..."""
        texts = []
        for i in range(40):
            texts.append(
                f"short {i}" if i % 2 == 0
                else f"a considerably longer passage number {i} " + ("padding words " * 30)
            )
        batched = encoder.encode(texts)
        assert batched.shape[0] == len(texts)

        # Each row must match the vector obtained by encoding that text alone.
        for i in (0, 1, 7, 20, 39):
            solo = encoder.encode([texts[i]])[0]
            assert float(solo @ batched[i]) > 0.999, f"row {i} misaligned after bucketing"

    def test_distinct_texts_get_distinct_vectors(self, encoder):
        v = encoder.encode(["bananas contain potassium", "corporations have limited liability"])
        assert float(v[0] @ v[1]) < 0.95

    def test_empty_and_single(self, encoder):
        assert encoder.encode([]).shape[0] == 0
        assert encoder.encode(["one text"]).shape == (1, encoder.dim)


class TestCache:
    def test_repeat_query_hits_the_cache(self, encoder):
        before = encoder.stats()["cache"]["hits"]
        q = "a distinctive query string for the cache test"
        a = encoder.encode_query(q)
        b = encoder.encode_query(q)
        after = encoder.stats()["cache"]["hits"]
        assert after > before
        assert np.allclose(a, b)

    def test_cache_is_case_and_space_insensitive(self, encoder):
        a = encoder.encode_query("What Is A Corporation")
        b = encoder.encode_query("  what is a corporation  ")
        assert np.allclose(a, b)


class TestMultilingual:
    def test_translation_pairs_are_close(self, encoder):
        """Cross-lingual retrieval depends entirely on this holding."""
        en = encoder.encode_query("how many calories are in a banana")
        hi = encoder.encode_query("एक केले में कितनी कैलोरी होती है")
        unrelated = encoder.encode_query("the history of naval architecture in Portugal")
        assert float(en @ hi) > float(en @ unrelated)

    def test_scripts_share_one_space(self, encoder):
        vecs = encoder.encode(
            [
                "a corporation is a legal entity",
                "निगम एक कानूनी संस्था है",
                "ஒரு நிறுவனம் ஒரு சட்ட அமைப்பு",
            ]
        )
        assert vecs.shape == (3, encoder.dim)
        assert float(vecs[0] @ vecs[1]) > 0.2
        assert float(vecs[0] @ vecs[2]) > 0.2


class TestStats:
    def test_reports_backend_and_dim(self, encoder):
        s = encoder.stats()
        assert s["dim"] == encoder.dim
        assert s["warm"] is True
        assert "cache" in s
