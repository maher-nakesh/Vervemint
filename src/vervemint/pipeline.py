"""The one function the UI, the API and the agent call to answer a
question from the documents.

Order of operations -- each step can stop the request early:
  1. guardrails.check_input      -> block empty / oversized / injection
  2. retriever.search            -> hybrid search + rerank
  3. guardrails.filter_retrieved -> drop chunks carrying injected text
  4. generate.generate_answer    -> abstain, or LLM answer + citations

Every request writes one human-readable log line and one structured
trace record (tracing.py), both tagged with the same request id.
"""

import logging
import time
from dataclasses import dataclass, field

from src.vervemint.generate import generate_answer
from src.vervemint.guardrails import (
    BLOCKED_MESSAGES,
    check_input,
    filter_retrieved,
    redact_secrets,
)
from src.vervemint.llm import LLMConfig
from src.vervemint.retrieve import RetrievedChunk, Retriever
from src.vervemint.tracing import (
    current_request_id,
    new_request_id,
    write_trace,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    request_id: str
    answer: str
    citations: list[str] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    blocked_reason: str | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _trace(question: str, llm: LLMConfig, result: PipelineResult) -> None:
    write_trace({
        "request_id": result.request_id,
        "question": redact_secrets(question),
        "provider": llm.provider,
        "model": llm.model,
        "blocked_reason": result.blocked_reason,
        "abstained": result.abstained,
        "citations": result.citations,
        "top_score": result.sources[0].score if result.sources else None,
        "sources": [f"{s.source}#p{s.page_no}" for s in result.sources],
        "latency_ms": result.latency_ms,
    })


def ask(question: str, retriever: Retriever, llm: LLMConfig) -> PipelineResult:
    """Answer one question end to end. Raises llm.LLMError on provider
    failure so the caller can show the message to the user."""
    request_id = current_request_id() or new_request_id()
    t_start = time.perf_counter()

    guard = check_input(question)
    if not guard.allowed:
        logger.warning("[%s] Blocked input | reason=%s",
                       request_id, guard.reason)
        result = PipelineResult(
            request_id=request_id,
            answer=BLOCKED_MESSAGES.get(guard.reason, "Request blocked."),
            blocked_reason=guard.reason,
        )
        _trace(question, llm, result)
        return result

    t_retrieve = time.perf_counter()
    hits = retriever.search(question)
    retrieve_ms = _ms_since(t_retrieve)

    safe_hits, removed = filter_retrieved(hits)
    if removed:
        logger.warning("[%s] Removed %d retrieved chunk(s) with injected "
                       "text", request_id, removed)

    t_generate = time.perf_counter()
    answer = generate_answer(question, safe_hits, llm)
    generate_ms = _ms_since(t_generate)

    result = PipelineResult(
        request_id=request_id,
        answer=answer.text,
        citations=answer.citations,
        sources=safe_hits,
        abstained=answer.abstained,
        latency_ms={
            "retrieve_ms": retrieve_ms,
            "generate_ms": generate_ms,
            "total_ms": _ms_since(t_start),
        },
    )
    logger.info(
        "[%s] Answered | q=%r | provider=%s model=%s | top_score=%.3f | "
        "abstained=%s | citations=%d | %s",
        request_id,
        redact_secrets(question)[:120],
        llm.provider,
        llm.model,
        safe_hits[0].score if safe_hits else 0.0,
        answer.abstained,
        len(answer.citations),
        result.latency_ms,
    )
    _trace(question, llm, result)
    return result
