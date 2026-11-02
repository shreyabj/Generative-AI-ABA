"""Percentile reporting - the numbers the submission is judged on."""

from __future__ import annotations

from setu.telemetry import MetricsCollector, TurnRecord, histogram, percentiles


class TestPercentiles:
    def test_p100_is_the_true_maximum(self):
        """Not P99.9 rounded up - the brief asks for P100."""
        values = list(range(1, 101))
        p = percentiles(values)
        assert p["p100"] == 100.0

    def test_p50_is_the_median(self):
        p = percentiles([10, 20, 30, 40, 50])
        assert p["p50"] == 30.0

    def test_p70_is_reported(self):
        p = percentiles(list(range(1, 101)))
        assert 65 <= p["p70"] <= 75

    def test_ordering_is_monotonic(self):
        p = percentiles([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        assert p["p50"] <= p["p70"] <= p["p90"] <= p["p95"] <= p["p99"] <= p["p100"]

    def test_empty_input_is_zeros(self):
        assert percentiles([])["p50"] == 0.0

    def test_single_sample(self):
        p = percentiles([42.0])
        assert p["p50"] == p["p100"] == 42.0


class TestHistogram:
    def test_buckets_and_overflow(self):
        h = histogram([5, 15, 15, 250, 900])
        assert sum(b["count"] for b in h) == 5
        assert h[-1]["hi"] is None
        assert h[-1]["count"] >= 1  # 900 ms lands in overflow

    def test_empty(self):
        assert all(b["count"] == 0 for b in histogram([]))


class TestCollector:
    def _rec(self, ms: float, status: str = "answered", **kw) -> TurnRecord:
        return TurnRecord(
            turn_id=f"t{ms}", pipeline_ms=ms, e2e_ms=ms, status=status,
            slo_met=ms <= 200.0, stages={"embed_query": ms * 0.6}, **kw
        )

    def test_empty_report_is_well_formed(self):
        r = MetricsCollector().report()
        assert r["samples"] == 0
        assert "note" in r
        assert r["pipeline_ms"]["p50"] == 0.0

    def test_reports_percentiles_and_compliance(self):
        c = MetricsCollector(window=100)
        for ms in (5, 6, 7, 8, 9, 10):
            c.record(self._rec(ms))
        r = c.report()
        assert r["samples"] == 6
        assert r["slo_compliance"] == 1.0
        assert r["slo_breaches"] == 0
        assert r["pipeline_ms"]["p100"] == 10.0

    def test_counts_breaches(self):
        c = MetricsCollector()
        c.record(self._rec(10))
        c.record(self._rec(500))
        r = c.report()
        assert r["slo_breaches"] == 1
        assert r["slo_compliance"] == 0.5

    def test_per_stage_percentiles(self):
        c = MetricsCollector()
        for ms in (10, 20, 30):
            c.record(self._rec(ms))
        stages = c.report()["stages"]
        assert "embed_query" in stages
        assert stages["embed_query"]["samples"] == 3
        assert stages["embed_query"]["p100"] == 18.0

    def test_window_is_bounded(self):
        c = MetricsCollector(window=5)
        for i in range(50):
            c.record(self._rec(float(i + 1)))
        r = c.report()
        assert r["samples"] == 5
        assert r["total_turns"] == 50  # lifetime count survives the ring buffer

    def test_status_and_answer_rate(self):
        c = MetricsCollector()
        c.record(self._rec(10, "answered"))
        c.record(self._rec(11, "abstained"))
        c.record(self._rec(12, "refused"))
        r = c.report()
        assert r["status_counts"]["answered"] == 1
        # The report rounds to 4 dp, so compare at that precision rather than exactly.
        assert abs(r["answer_rate"] - 1 / 3) < 1e-4

    def test_headline_shape(self):
        c = MetricsCollector()
        c.record(self._rec(10))
        h = c.headline()
        assert set(h) == {"p50", "p70", "p100", "samples"}

    def test_clear(self):
        c = MetricsCollector()
        c.record(self._rec(10))
        c.clear()
        assert c.report()["samples"] == 0
