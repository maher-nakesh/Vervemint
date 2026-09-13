"""Turn retrieved chunks into a grounded, cited answer -- or an honest
abstention when the evidence is too weak.

This is where "faithfulness" (does the answer actually rest on the
retrieved evidence, or did the model make something up?) gets enforced,
not just hoped for. Two concrete mechanisms:

1. Abstention: if the best reranker score is below
   `settings.min_rerank_score`, we never call the LLM at all -- we
   return a fixed "not enough evidence" response. A false "I don't
   know" is cheap; a confident wrong answer about machine maintenance
   is not.
2. Citation enforcement: the prompt requires every claim to be tagged
   with a source id like [S1], and `validate_citations` checks, after
   generation, that every tag the model used actually exists in the
   context we gave it. A citation to a source that was never provided
   is a strong signal of hallucination and is stripped out rather than
   shown to the user.
"""

import re
from dataclasses import dataclass

import ollama

from src.copilot.config import settings
from src.copilot.retrieve import RetrievedChunk

_CITATION_RE = re.compile(r"\[S(\d+)\]")

ABSTAIN_MESSAGE = (
    "I don't have reliable documentation to answer this confidently. "
    "The closest matches in the manual set were not a strong enough "
    "match to justify an answer. Please rephrase, or consult a "
    "qualified technician."
)

_SYSTEM_PROMPT = """You are a technical assistant for industrial \
equipment documentation. Answer ONLY using the numbered sources below. \
Every factual sentence must end with a citation tag like [S1] pointing \
to the source it came from. If the sources do not contain the answer, \
say so explicitly instead of guessing."""


@dataclass
class Answer:
    text: str
    citations: list[str]  # source_pdf#page strings actually used
    abstained: bool


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as numbered sources the model can cite by index."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[S{i}] (source: {chunk.source_pdf}, page {chunk.page_no})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def _validate_citations(
    answer_text: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[str]]:
    """Strip citation tags that don't map to a real provided source,
    and return the list of sources that were legitimately cited.

    This is the output guardrail for generation: the model can still
    hallucinate a fact, but it cannot hallucinate a *citation* that
    points somewhere real if that pointer doesn't exist in what we gave
    it -- any [S7] when only 5 sources were provided gets removed.
    """
    used_sources: list[str] = []

    def _replace(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks):
            chunk = chunks[idx - 1]
            label = f"{chunk.source_pdf}#p{chunk.page_no}"
            used_sources.append(label)
            return f"[{label}]"
        return ""  # drop invalid citation silently

    cleaned = _CITATION_RE.sub(_replace, answer_text)
    return cleaned, used_sources


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> Answer:
    """The main entry point: decide to abstain, or call the LLM and
    validate its citations.
    """
    if not chunks or chunks[0].score < settings.min_rerank_score:
        return Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True)

    context = _build_context(chunks)
    user_prompt = f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = ollama.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_text = response["message"]["content"]
    cleaned_text, used_sources = _validate_citations(raw_text, chunks)

    return Answer(text=cleaned_text, citations=used_sources, abstained=False)


if __name__ == "__main__":
    from src.copilot.retrieve import Retriever

    retriever = Retriever()
    query = "What should I check if a motor is overheating?"
    hits = retriever.search(query)
    result = generate_answer(query, hits)
    print(result.text)
    print("Citations used:", result.citations)
