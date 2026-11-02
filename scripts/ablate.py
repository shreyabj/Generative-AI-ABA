"""Chunking and retrieval ablation.

The brief asks for real thought about chunking rather than one naive strategy. A design is
only "thought through" if you can show what each part contributes, so this script rebuilds
the index under reduced configurations and measures recall@5 against the gold labels for
each one.

Configurations
--------------
window-only-naive   A single fixed 500-character window layer, no overlap, no metadata
                    header, dense retrieval only. This is the baseline the brief warns
                    against, included so the comparison is against something real.
atomic-only         Proposition layer alone.
document-only       Whole passages alone.
no-metadata-prefix  Full lattice, but L3 embeds raw text with no metadata header.
dense-only          Full lattice, BM25 disabled.
sparse-only         Full lattice, dense retrieval disabled.
no-rerank           Full lattice, late-interaction rerank disabled.
full-lattice        Everything on.

Each run is a genuine rebuild (chunk, embed, index), so it is slow. Use --n to keep the
evaluation slice small while iterating.

    python scripts/ablate.py --docs 1200 --n 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from _common import ROOT, hr, load_gold


async def evaluate_config(
    name: str, docs, queries, *, n: int, patch: dict, log
) -> dict:
    """Rebuild the index with ``patch`` applied to settings, then measure recall@k."""
    import copy

    from setu.chunking.lattice import ChunkLattice
    from setu.config import settings
    from setu.embed import get_encoder
    from setu.harness.contracts import AskRequest
    from setu.index.corpus import CorpusIndex
    from setu.pipeline import SetuPipeline

    original = {k: getattr(settings, k) for k in patch}
    for k, v in patch.items():
        setattr(settings, k, v)

    try:
        enc = get_encoder()
        # Fresh Document objects each run: chunkers mutate `sentences` / `sent_vecs`.
        fresh = copy.deepcopy(docs)
        t0 = time.perf_counter()
        lattice = ChunkLattice(enc)
        groups = list(lattice.build(fresh))
        index = CorpusIndex().build(fresh, groups, enc)
        build_s = time.perf_counter() - t0

        pipe = SetuPipeline(index)
        pipe.warm()

        ks = (1, 3, 5, 10)
        hits = {k: 0 for k in ks}
        rr: list[float] = []
        lat: list[float] = []
        answered = 0

        usable = [q for q in queries if any(d in index.docs for d in q.gold_doc_ids)][:n]
        for g in usable:
            r = await pipe.ask(AskRequest(text=g.query, lang=g.lang, top_k=10))
            retrieved = [h.doc_id for h in (r.retrieval.hits if r.retrieval else [])]
            goldset = set(g.gold_doc_ids)
            rank = next((j + 1 for j, d in enumerate(retrieved) if d in goldset), None)
            for k in ks:
                if rank is not None and rank <= k:
                    hits[k] += 1
            rr.append(1.0 / rank if rank else 0.0)
            lat.append(r.pipeline_ms)
            if r.status.value in {"answered", "degraded"}:
                answered += 1

        m = max(len(usable), 1)
        out = {
            "config": name,
            "chunks": len(index.chunks),
            "per_layer": {k: v for k, v in index.manifest["per_layer"].items() if v},
            "build_s": round(build_s, 1),
            "queries": len(usable),
            **{f"recall@{k}": round(hits[k] / m, 4) for k in ks},
            "mrr@10": round(statistics.mean(rr), 4) if rr else 0.0,
            "answer_rate": round(answered / m, 4),
            "p50_ms": round(statistics.median(lat), 2) if lat else 0.0,
        }
        log(f"  {name:<20} chunks={out['chunks']:<7} recall@5={out['recall@5']:.3f} "
            f"mrr={out['mrr@10']:.3f} p50={out['p50_ms']:.1f}ms")
        return out
    finally:
        for k, v in original.items():
            setattr(settings, k, v)


CONFIGS: list[tuple[str, dict]] = [
    ("window-only-naive", {
        "enabled_layers": "window", "window_chars": 500,
        "window_overlap_min": 0.0, "window_overlap_max": 0.0,
        "metadata_prefix": False, "sparse_weight": 0.0, "dense_weight": 1.0,
    }),
    ("atomic-only", {"enabled_layers": "atomic"}),
    ("document-only", {"enabled_layers": "document"}),
    ("no-metadata-prefix", {"metadata_prefix": False}),
    ("dense-only", {"sparse_weight": 0.0, "dense_weight": 1.0}),
    ("sparse-only", {"sparse_weight": 1.0, "dense_weight": 0.0}),
    ("no-rerank", {"rerank_k": 0}),
    ("full-lattice", {}),
]


async def run(args: argparse.Namespace) -> int:
    from setu.config import settings
    from setu.ingest.msmarco_xi import read_corpus

    path = settings.corpus_file if settings.corpus_file.exists() else settings.sample_corpus_file
    if not path.exists():
        print("no corpus - run: python -m setu.cli ingest --langs hi,ta --rows 220")
        return 1

    docs = read_corpus(path)[: args.docs]
    queries = load_gold()
    print(f"ablation over {len(docs)} documents, {min(args.n, len(queries))} gold queries")
    print("(each row is a full rebuild: chunk + embed + index)")
    print()

    wanted = set(args.only.split(",")) if args.only else None
    results = []
    for name, patch in CONFIGS:
        if wanted and name not in wanted:
            continue
        results.append(
            await evaluate_config(name, docs, queries, n=args.n, patch=patch, log=print)
        )

    hr("ABLATION")
    print(f"  {'config':<20}{'chunks':>8}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@10':>7}"
          f"{'MRR':>7}{'ans':>7}{'P50':>8}")
    base = next((r for r in results if r["config"] == "full-lattice"), None)
    for r in results:
        print(f"  {r['config']:<20}{r['chunks']:>8}{r['recall@1']:>7.3f}{r['recall@3']:>7.3f}"
              f"{r['recall@5']:>7.3f}{r['recall@10']:>7.3f}{r['mrr@10']:>7.3f}"
              f"{r['answer_rate']:>7.3f}{r['p50_ms']:>8.1f}")
    if base:
        hr("DELTA vs full-lattice  (recall@5)")
        for r in results:
            if r["config"] == "full-lattice":
                continue
            d = r["recall@5"] - base["recall@5"]
            print(f"  {r['config']:<22}{d:+.3f}  {'#' * int(abs(d) * 120)}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"documents": len(docs), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Chunking / retrieval ablation")
    ap.add_argument("--docs", type=int, default=1500, help="documents per rebuild")
    ap.add_argument("--n", type=int, default=120, help="gold queries per config")
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--out", default="docs/ablation.json")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
