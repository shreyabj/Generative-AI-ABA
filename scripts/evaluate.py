"""Retrieval and answer quality against MS MARCO-XI's own gold labels.

Latency without quality is meaningless - a retriever that returns nothing is very fast. The
dataset gives us two supervision signals for free, so there is no excuse for not measuring:

``is_selected``  A human marked which of the ten candidate passages answers the query. That
                 makes recall@k and MRR@10 computable with no annotation work.
``Answer``       A gold answer string, so we can measure token recall of the gold answer
                 inside what SETU actually said.

Reported metrics
----------------
recall@1/3/5/10   fraction of queries where a gold passage appears in the top k
mrr@10            mean reciprocal rank of the first gold passage
answer recall     token recall of the gold answer within the produced answer
abstention        how often the system declined, split into correct (no gold passage was
                  retrievable) and incorrect (gold was right there and we still refused)

That last split is the one worth reading. Abstaining is only a virtue when the answer really
was not available; abstaining on an answerable question is a failure that a groundedness
threshold set too aggressively will happily produce.

    python scripts/evaluate.py --n 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from _common import ROOT, hr, load_gold, load_pipeline


def token_recall(gold: str, produced: str) -> float:
    from setu.text import content_tokens

    g = set(content_tokens(gold))
    if not g:
        return 0.0
    p = set(content_tokens(produced))
    return len(g & p) / len(g)


async def run(args: argparse.Namespace) -> int:
    from setu.harness.contracts import AnswerMode, AskRequest

    pipe, index, _ = load_pipeline()
    gold = load_gold()
    if not gold:
        print("No gold queries found. Run:  python -m setu.cli ingest --langs hi,ta --rows 220")
        return 1

    # Only evaluate queries whose gold passages actually made it into this index slice.
    usable = [
        g for g in gold
        if g.gold_doc_ids
        and any(d in index.docs for d in g.gold_doc_ids)
        # Skip machine-translation artefacts masquerading as queries.
        and 8 < len(g.query) <= 300
    ]
    print(f"gold queries: {len(gold)} total, {len(usable)} with an indexed gold passage")
    sample = usable[: args.n]
    if not sample:
        print("no evaluable queries")
        return 1

    ks = (1, 3, 5, 10)
    hits = {k: 0 for k in ks}
    rr: list[float] = []
    ans_recall: list[float] = []
    abstain_correct = 0
    abstain_wrong = 0
    answered = 0
    per_lang: dict[str, dict[str, float]] = {}
    rows: list[dict] = []

    for i, g in enumerate(sample):
        r = await pipe.ask(
            AskRequest(text=g.query, lang=g.lang, mode=AnswerMode(args.mode), top_k=10)
        )
        retrieved = [h.doc_id for h in (r.retrieval.hits if r.retrieval else [])]
        goldset = set(g.gold_doc_ids)

        rank = next((j + 1 for j, d in enumerate(retrieved) if d in goldset), None)
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        rr.append(1.0 / rank if rank else 0.0)

        produced = r.answer if r.status.value in {"answered", "degraded"} else ""
        if produced:
            answered += 1
            ar = token_recall(g.gold_answer, produced)
            ans_recall.append(ar)
        else:
            ar = 0.0
            if rank is None:
                abstain_correct += 1
            else:
                abstain_wrong += 1

        lang = per_lang.setdefault(g.lang, {"n": 0, "hit5": 0, "recall": 0.0})
        lang["n"] += 1
        if rank is not None and rank <= 5:
            lang["hit5"] += 1
        lang["recall"] += ar

        rows.append(
            {
                "query": g.query[:70], "lang": g.lang, "type": g.query_type,
                "gold_rank": rank, "status": r.status.value, "trust": r.trust_score,
                "answer_recall": round(ar, 3), "pipeline_ms": r.pipeline_ms,
            }
        )
        if args.progress and (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(sample)}")

    n = len(sample)
    hr("RETRIEVAL  (gold = human is_selected passage)")
    for k in ks:
        pctv = hits[k] / n
        print(f"  recall@{k:<3} {pctv:6.3f}   {'#' * int(pctv * 44)}")
    print(f"  MRR@10     {statistics.mean(rr):6.3f}")

    hr("ANSWERING")
    print(f"  answered            {answered}/{n}  ({answered / n * 100:.1f}%)")
    print(f"  abstained (correct) {abstain_correct:>3}   gold passage was not retrievable")
    print(f"  abstained (wrong)   {abstain_wrong:>3}   gold was retrieved but we still declined")
    if ans_recall:
        print(f"  gold-answer token recall  mean {statistics.mean(ans_recall):.3f}  "
              f"median {statistics.median(ans_recall):.3f}")

    hr("BY LANGUAGE")
    print(f"  {'lang':<6}{'n':>5}{'recall@5':>11}{'ans recall':>13}")
    for lang, d in sorted(per_lang.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {lang:<6}{int(d['n']):>5}{d['hit5'] / d['n']:>11.3f}{d['recall'] / d['n']:>13.3f}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "queries": n,
                    "mode": args.mode,
                    "recall": {f"@{k}": round(hits[k] / n, 4) for k in ks},
                    "mrr@10": round(statistics.mean(rr), 4),
                    "answered": answered,
                    "abstain_correct": abstain_correct,
                    "abstain_wrong": abstain_wrong,
                    "answer_token_recall_mean": round(statistics.mean(ans_recall), 4) if ans_recall else 0.0,
                    "by_language": {
                        k: {
                            "n": int(v["n"]),
                            "recall@5": round(v["hit5"] / v["n"], 4),
                            "answer_recall": round(v["recall"] / v["n"], 4),
                        }
                        for k, v in per_lang.items()
                    },
                    "rows": rows,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval + answer quality vs MS MARCO-XI gold labels")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", default="reflex", choices=["reflex", "deliberate", "auto"])
    ap.add_argument("--out", default="docs/quality.json")
    ap.add_argument("--progress", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
