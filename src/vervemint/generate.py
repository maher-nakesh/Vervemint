"""Turn retrieved chunks into a grounded, cited answer -- or an honest
abstention when the evidence is too weak.

1. Abstention: if the best reranker score is below
   `settings.min_rerank_score`, the LLM is never called. A false
   "I don't know" is cheap; a confident wrong maintenance answer is not.
2. Citation enforcement: the prompt requires a tag like [S1] on every
   claim, and `_validate_citations` removes any tag that doesn't point
   to a source we actually provided.

Which LLM answers (Ollama, Claude, OpenAI) is decided by the LLMConfig
passed in; this module doesn't care.
"""

import logging
import re
from dataclasses import dataclass

from src.vervemint.config import settings
from src.vervemint.llm import LLMConfig, chat
from src.vervemint.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

# Matches [S1] and grouped forms like [S1, S2] or [S1;S3].
_CITATION_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
NOT_IN_SOURCES = "NOT_IN_SOURCES"
_NOT_IN_SOURCES_RE = re.compile(rf"\b{NOT_IN_SOURCES}\b[:.]?")

ABSTAIN_MESSAGE = (
    "I don't have reliable documentation to answer this confidently. "
    "None of the retrieved passages were a strong enough match. "
    "Try rephrasing, or check that the right documents are loaded."
)

NOT_FOUND_MESSAGE = (
    "The retrieved documents don't contain this information. The "
    "passages that were checked are listed below the answer."
)

UNCITED_NOTE = (
    "\n\n*Unverified: this answer cites no source. Check it against the "
    "passages under Sources before relying on it.*"
)

_SYSTEM_PROMPT = f"""You are a technical assistant for industrial \
equipment documentation. Answer ONLY using the numbered sources below.
- End every factual sentence with a citation tag like [S1] naming the \
source it came from.
- Sources may describe different products. Use a source only if it is \
about the exact product or model in the question.
- If the sources only partly answer the question, answer with what \
they say and state what is missing.
- Only if no source addresses the question at all, reply with exactly \
{NOT_IN_SOURCES} and nothing else. Never guess.
- Be brief: at most 5 sentences or a short list."""


@dataclass
class Answer:
    text: str
    citations: list[str]  # "source#pN" labels actually cited
    abstained: bool


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as numbered sources the model can cite by index."""
    return "\n\n".join(
        f"[S{i}] (source: {c.source}, page {c.page_no})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


def _validate_citations(
    answer_text: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[str]]:
    """Map [S#] tags to real source labels; drop tags with no source."""
    used_sources: list[str] = []

    def _replace(match: re.Match) -> str:
        labels = []
        for tag in re.split(r"\s*[,;]\s*", match.group(1)):
            idx = int(tag[1:])
            if not 1 <= idx <= len(chunks):
                logger.warning("Dropped invalid citation [S%d]", idx)
                continue
            chunk = chunks[idx - 1]
            label = f"{chunk.source}#p{chunk.page_no}"
            if label not in used_sources:
                used_sources.append(label)
            labels.append(f"[{label}]")
        return "".join(labels)

    cleaned = _CITATION_RE.sub(_replace, answer_text)
    return cleaned, used_sources


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMConfig,
    system: str = _SYSTEM_PROMPT,
    request: str | None = None,
) -> Answer:
    """Abstain on weak evidence, otherwise ask the LLM and check citations.

    `system` and `request` let other workflows (log analysis) reuse the
    same abstention, "not in sources" and citation checks with their own
    instructions. Raises llm.LLMError if the provider call fails.
    """
    if not chunks or chunks[0].score < settings.min_rerank_score:
        top = chunks[0].score if chunks else None
        logger.info("Abstained | top_score=%s", top)
        return Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True)

    user_prompt = (
        f"Sources:\n{_build_context(chunks)}\n\n"
        f"{request or f'Question: {query}'}\n\nAnswer:"
    )
    raw_text = chat(llm, system, user_prompt)
    cleaned_text, used_sources = _validate_citations(raw_text, chunks)
    if NOT_IN_SOURCES in raw_text:
        if not used_sources:
            # The model read the passages and found no answer: a clean
            # abstention, not an uncited refusal.
            logger.info("Abstained | model found no answer in sources")
            return Answer(text=NOT_FOUND_MESSAGE, citations=[],
                          abstained=True)
        # A cited answer that marks only some points as not covered
        # (e.g. one of several problems in a log): keep the answer.
        cleaned_text = _NOT_IN_SOURCES_RE.sub(
            "Not covered by the sources.", cleaned_text
        )

    if not used_sources:
        # The model ignored the citation rule, so nothing ties this
        # answer to the documents. Say so instead of presenting it as
        # grounded.
        logger.warning("Answer has no valid citations | model=%s", llm.model)
        cleaned_text += UNCITED_NOTE
    return Answer(text=cleaned_text, citations=used_sources, abstained=False)
