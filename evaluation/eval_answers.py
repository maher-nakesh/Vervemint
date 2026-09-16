"""End-to-end answer accuracy: does retrieval actually help the LLM?

Each multiple-choice question is asked twice with the same model:
  closed-book   question + options only (the model's own knowledge)
  rag           question + options + the top retrieved passages
Score = exact match between the predicted letters and the gold letters
(all correct options, no extra ones).

This calls the LLM 2x per question, so it uses a fixed random sample.
With Ollama it is free; with Claude/OpenAI it costs API credits and the
key is read from ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.

Run from the project root:
    python -m evaluation.eval_answers [--n 50] [--provider ollama]
"""

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

from evaluation.build_eval_set import load_eval_set
from src.vervemint.config import settings
from src.vervemint.llm import LLMConfig, chat
from src.vervemint.logging_config import setup_logging
from src.vervemint.retrieve import Retriever, load_models

logger = logging.getLogger(__name__)

RESULTS_FILE = Path(__file__).parent / "results" / "answers.json"
_LETTER_RE = re.compile(r"\b[A-J]\b")

_SYSTEM = (
    "You answer multiple-choice questions about industrial electronics. "
    "Some questions have more than one correct option. Reply with the "
    "letters of all correct options only, comma-separated, e.g. 'B' or "
    "'A, C'. No explanation."
)


def parse_letters(reply: str) -> list[str]:
    """Letters from the first line of the reply, e.g. 'A, C' -> [A, C]."""
    first_line = reply.strip().splitlines()[0] if reply.strip() else ""
    return sorted(set(_LETTER_RE.findall(first_line)))


def _ask_mcq(llm: LLMConfig, item: dict, context: str | None) -> list[str]:
    prompt = ""
    if context:
        prompt += f"Documentation:\n{context}\n\n"
    prompt += (
        f"Question: {item['question']}\n"
        f"Options: {item['options']}\n\nAnswer:"
    )
    return parse_letters(chat(llm, _SYSTEM, prompt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "claude", "openai", "gemini"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    default_model = {
        "ollama": settings.ollama_model,
        "claude": settings.claude_model,
        "openai": settings.openai_model,
        "gemini": settings.gemini_model,
    }[args.provider]
    key_env = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    llm = LLMConfig(
        args.provider,
        args.model or default_model,
        os.environ.get(key_env.get(args.provider, "")),
    )

    items = [i for i in load_eval_set() if i["options"]]
    sample = random.Random(42).sample(items, min(args.n, len(items)))
    retriever = Retriever.from_disk(*load_models())

    correct = {"closed_book": 0, "rag": 0}
    rows = []
    for n, item in enumerate(sample, start=1):
        hits = retriever.search(item["question"])
        context = "\n\n".join(h.text for h in hits)
        pred_closed = _ask_mcq(llm, item, None)
        pred_rag = _ask_mcq(llm, item, context)
        correct["closed_book"] += pred_closed == item["answer"]
        correct["rag"] += pred_rag == item["answer"]
        rows.append({"id": item["id"], "gold": item["answer"],
                     "closed_book": pred_closed, "rag": pred_rag})
        logger.info("%d/%d gold=%s closed=%s rag=%s", n, len(sample),
                    item["answer"], pred_closed, pred_rag)

    accuracy = {k: round(v / len(sample), 3) for k, v in correct.items()}
    print(f"\nAnswer accuracy on {len(sample)} questions "
          f"({llm.provider} / {llm.model}, exact match)\n")
    print("| setting | accuracy |")
    print("|---|---|")
    print(f"| closed-book (no retrieval) | {accuracy['closed_book']:.3f} |")
    print(f"| RAG (top-{settings.rerank_top_k} passages) "
          f"| {accuracy['rag']:.3f} |")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({
        "provider": llm.provider, "model": llm.model,
        "n_questions": len(sample), "accuracy": accuracy, "rows": rows,
    }, indent=2))
    logger.info("Saved %s", RESULTS_FILE)


if __name__ == "__main__":
    setup_logging()
    main()
