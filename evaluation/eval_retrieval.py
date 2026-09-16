"""Compare retrieval strategies on the benchmark built by build_eval_set.

For every question, the same Retriever is scored four ways:
  bm25             keyword search only
  dense            embedding search only
  hybrid           BM25 + dense fused with RRF
  hybrid+rerank    hybrid, then the cross-encoder (what the app uses)

Metrics (a retrieved chunk is correct if its page is a gold page):
  Hit@1, Hit@5   share of questions with a correct chunk in the top k
  MRR@10         mean of 1/rank of the first correct chunk (0 if none
                 in the top 10) -- rewards putting it near the top

No LLM is involved, so this runs in minutes and costs nothing.

Run from the project root:
    python -m evaluation.eval_retrieval [--limit 200]
"""

import argparse
import json
import logging
import time
from pathlib import Path

from evaluation.build_eval_set import load_eval_set
from src.vervemint.config import settings
from src.vervemint.logging_config import setup_logging
from src.vervemint.retrieve import Retriever, load_models

logger = logging.getLogger(__name__)

RESULTS_FILE = Path(__file__).parent / "results" / "retrieval.json"
METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank")


def _rankings(retriever: Retriever, question: str) -> dict[str, list[int]]:
    """Chunk rows ranked best-first by each method."""
    bm25 = retriever.bm25_search(question, settings.bm25_top_k)
    dense = retriever.dense_search(question, settings.dense_top_k)
    hybrid = retriever.fuse_rrf([bm25, dense])
    reranked = [row for row, _ in retriever.rerank(question, hybrid)]
    return {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid+rerank": reranked,
    }


def _first_hit_rank(
    retriever: Retriever, rows: list[int], gold: set[tuple[str, int]]
) -> int | None:
    """1-based rank of the first chunk from a gold page, within top 10."""
    for rank, row in enumerate(rows[:10], start=1):
        if retriever.location(row) in gold:
            return rank
    return None


def evaluate(retriever: Retriever, items: list[dict]) -> dict[str, dict]:
    ranks: dict[str, list[int | None]] = {m: [] for m in METHODS}
    for n, item in enumerate(items, start=1):
        gold = {tuple(p) for p in item["gold_pages"]}
        for method, rows in _rankings(retriever, item["question"]).items():
            ranks[method].append(_first_hit_rank(retriever, rows, gold))
        if n % 100 == 0:
            logger.info("Evaluated %d/%d questions", n, len(items))

    def share(values: list[bool]) -> float:
        return round(sum(values) / len(values), 3)

    return {
        method: {
            "hit@1": share([r == 1 for r in r_list]),
            "hit@5": share([r is not None and r <= 5 for r in r_list]),
            "mrr@10": round(
                sum(1 / r for r in r_list if r) / len(r_list), 3
            ),
        }
        for method, r_list in ranks.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    args = parser.parse_args()

    items = load_eval_set()[: args.limit]
    retriever = Retriever.from_disk(*load_models())

    start = time.perf_counter()
    metrics = evaluate(retriever, items)
    elapsed = time.perf_counter() - start

    print(f"\nRetrieval on {len(items)} questions ({elapsed:.0f}s)\n")
    print("| method | Hit@1 | Hit@5 | MRR@10 |")
    print("|---|---|---|---|")
    for method, m in metrics.items():
        print(f"| {method} | {m['hit@1']:.3f} | {m['hit@5']:.3f} "
              f"| {m['mrr@10']:.3f} |")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(
        {"n_questions": len(items), "metrics": metrics}, indent=2
    ))
    logger.info("Saved %s", RESULTS_FILE)


if __name__ == "__main__":
    setup_logging()
    main()
