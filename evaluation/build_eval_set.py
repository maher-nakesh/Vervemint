"""Build a retrieval benchmark from the dataset's own QA pairs.

Each test question in panasonic_qa_v1 ships with the "gold" document
page(s) it was written from. This script finds those pages in our
corpus, giving every question a set of (source, page_no) pairs that
count as a correct retrieval.

Output: evaluation/eval_set.jsonl, one question per line:
  {"id", "question", "options", "answer", "gold_pages": [[source, page]]}

Run from the project root:
    python -m evaluation.build_eval_set
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.vervemint.config import settings
from src.vervemint.ingest import load_corpus, strip_images
from src.vervemint.logging_config import setup_logging

logger = logging.getLogger(__name__)

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"
_KEY_CHARS = 200  # matching on a normalized prefix tolerates OCR noise


def _page_key(text: str) -> str:
    """Whitespace-normalized prefix of a page's cleaned text."""
    return " ".join(strip_images(text).split())[:_KEY_CHARS]


def split_question(raw: str) -> tuple[str, str]:
    """Split the dataset's MCQ text into (question, options).

    Retrieval uses only the question part: a real technician types a
    question, not a list of answer options.
    """
    question, _, options = raw.partition("Options:")
    return question.strip(), options.strip()


def build() -> None:
    corpus = load_corpus()
    page_index: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in corpus.itertuples(index=False):
        # Identical pages can appear in several PDFs; all count as gold.
        page_index[_page_key(row.md_content)].add(
            (Path(row.file_path).stem, int(row.page_no))
        )

    qa = pd.read_parquet(settings.qa_dir / "test-00000-of-00001.parquet")
    items, unmatched = [], 0
    for i, row in enumerate(qa.itertuples(index=False)):
        gold: set[tuple[str, int]] = set()
        for doc in row.documents:
            gold |= page_index.get(_page_key(doc), set())
        if not gold:
            unmatched += 1
            continue
        question, options = split_question(row.question)
        items.append({
            "id": i,
            "question": question,
            "options": options,
            "answer": sorted(row.answer),
            "gold_pages": sorted(gold),
        })

    with EVAL_FILE.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(
        "Eval set: %d questions matched to corpus pages, %d skipped "
        "(gold page not found) -> %s",
        len(items), unmatched, EVAL_FILE,
    )


def load_eval_set() -> list[dict]:
    with EVAL_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    setup_logging()
    build()
