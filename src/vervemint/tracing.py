"""Structured per-request traces: one JSON line per answered question.

Logs (logs/vervemint.log) are for humans reading what happened. Traces
(logs/traces.jsonl) are for measuring it: every line has the same
fields, so they can be loaded into pandas or a dashboard to answer
"what is our abstention rate?" or "what is p95 latency?".

Never stored: API keys. Stored: the question text, so keep this file
private if questions can contain sensitive data.
"""

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.vervemint.config import settings

logger = logging.getLogger(__name__)

TRACE_FILE = settings.log_dir / "traces.jsonl"

# The id of the API request being handled. The API middleware sets it,
# and it follows the request into the worker thread, so the access log,
# the pipeline log, logs/llm.log and the trace all carry the same id.
_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> None:
    _REQUEST_ID.set(request_id)


def current_request_id() -> str | None:
    """The API request's id, or None outside a request (scripts, tests)."""
    return _REQUEST_ID.get()


def write_trace(record: dict[str, Any]) -> None:
    """Append one record. A tracing failure must never break a request,
    so errors are logged and swallowed."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Could not write trace")


def load_traces() -> list[dict[str, Any]]:
    if not TRACE_FILE.exists():
        return []
    with TRACE_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers for the /stats endpoint."""
    if not traces:
        return {"requests": 0}
    total_ms = [t["latency_ms"]["total_ms"] for t in traces
                if t.get("latency_ms", {}).get("total_ms") is not None]
    answered = [t for t in traces if not t.get("blocked_reason")]

    def pct(q: int) -> float | None:
        return float(np.percentile(total_ms, q)) if total_ms else None

    return {
        "requests": len(traces),
        "blocked": len(traces) - len(answered),
        "abstention_rate": round(
            sum(t["abstained"] for t in answered) / max(len(answered), 1), 3
        ),
        "uncited_rate": round(
            sum(1 for t in answered
                if not t["abstained"] and not t["citations"])
            / max(len(answered), 1), 3
        ),
        "latency_ms_p50": pct(50),
        "latency_ms_p95": pct(95),
    }
