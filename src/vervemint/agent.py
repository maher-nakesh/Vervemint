"""Maintenance agent: an LLM that decides which tools to call to handle a
technician's request, instead of answering in a single step.

Tools:
- get_machine_health : sensor readings for an asset. SIMULATED in this
                       demo (stands in for a condition-monitoring feed).
- search_manuals     : the full RAG pipeline (guardrails, retrieval,
                       abstention, citations) wrapped as a tool.
- create_work_order  : drafts a work order. It is NEVER executed by the
                       agent -- the loop stops and a human must approve.

The model only *requests* tool calls; this module executes them. That
split is what makes least privilege and human approval enforceable:
the model cannot do anything the code doesn't allow.
"""

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.vervemint.config import settings
from src.vervemint.guardrails import BLOCKED_MESSAGES, check_input
from src.vervemint.llm import (
    LLMConfig,
    ToolCall,
    chat_with_tools,
    tool_result_messages,
)
from src.vervemint.pipeline import ask
from src.vervemint.retrieve import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

WORK_ORDER_FILE = settings.data_dir / "work_orders.jsonl"
PRIORITIES = ("low", "medium", "high", "critical")
_CITATION_RE = re.compile(r"\[([^\[\]]+?#p\d+)\]")

SYSTEM_PROMPT = """You are Vervemint, a maintenance assistant for an \
industrial plant. You help technicians diagnose equipment and decide \
what to do.

How to work:
1. If the request names an asset (like P-07), call get_machine_health \
first.
2. Use search_manuals for technical facts. Copy its citations into \
your answer exactly as returned. Never write a citation that \
search_manuals did not return.
3. If a reading is in WARNING or ALARM, call create_work_order with a \
short, specific summary. A human approves it; you never create it \
yourself.
4. Never invent sensor values or documentation content.

Answer briefly: findings first, then the recommended action."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_machine_health",
        "description": (
            "Current condition-monitoring readings for one asset: "
            "vibration, temperature, detected fault pattern, 7-day trend "
            "and status (OK, WARNING, ALARM)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "Asset ID, for example P-07.",
                }
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "search_manuals",
        "description": (
            "Search the technical documentation. Returns an answer with "
            "citations, or says the documents don't cover the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific technical question.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_work_order",
        "description": (
            "Draft a maintenance work order for an asset. The draft is "
            "shown to a human for approval; it is not created until then."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "summary": {
                    "type": "string",
                    "description": "Problem and requested action.",
                },
            },
            "required": ["asset_id", "priority", "summary"],
        },
    },
]

# Simulated condition-monitoring data. In a real deployment this would
# be an API call to the plant's monitoring system.
SIMULATED_ASSETS: dict[str, dict[str, Any]] = {
    "P-07": {
        "asset": "Centrifugal pump, cooling water loop, Hall 2",
        "vibration_rms_mm_s": 7.8,
        "vibration_alarm_limit_mm_s": 7.1,
        "bearing_temp_c": 71,
        "fault_pattern": "Peak at bearing outer-race defect frequency",
        "trend_7d": "Vibration up 45%",
        "status": "ALARM",
    },
    "M-12": {
        "asset": "55 kW electric motor, conveyor line 3",
        "vibration_rms_mm_s": 2.1,
        "vibration_alarm_limit_mm_s": 7.1,
        "winding_temp_c": 96,
        "winding_temp_limit_c": 90,
        "fault_pattern": "None detected",
        "trend_7d": "Temperature up 12 C",
        "status": "WARNING",
    },
    "F-03": {
        "asset": "Exhaust fan, paint shop",
        "vibration_rms_mm_s": 1.2,
        "vibration_alarm_limit_mm_s": 7.1,
        "bearing_temp_c": 38,
        "fault_pattern": "None detected",
        "trend_7d": "Stable",
        "status": "OK",
    },
}


@dataclass
class WorkOrderDraft:
    asset_id: str
    priority: str
    summary: str


@dataclass
class AgentResult:
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    pending_work_order: WorkOrderDraft | None = None
    # answered | awaiting_approval | max_steps | blocked
    stopped_reason: str = "answered"
    # Passages search_manuals retrieved, so the UI can show what the
    # agent's citations point to.
    sources: list[RetrievedChunk] = field(default_factory=list)


# --- Tools -------------------------------------------------------


def get_machine_health(asset_id: str) -> dict[str, Any]:
    asset_id = asset_id.strip().upper()
    if asset_id not in SIMULATED_ASSETS:
        return {
            "error": f"Unknown asset {asset_id}",
            "known_assets": sorted(SIMULATED_ASSETS),
        }
    return {"asset_id": asset_id, "simulated": True,
            **SIMULATED_ASSETS[asset_id]}


def _search_manuals(
    query: str,
    retriever: Retriever,
    llm: LLMConfig,
    cited: set[str],
    found: list[RetrievedChunk],
) -> dict:
    result = ask(query, retriever, llm)
    cited.update(result.citations)
    found.extend(result.sources)
    return {
        "answer": result.answer,
        "citations": result.citations,
        "abstained": result.abstained,
    }


def _draft_from_call(call: ToolCall) -> WorkOrderDraft:
    priority = str(call.arguments.get("priority", "medium")).lower()
    if priority not in PRIORITIES:
        priority = "medium"
    return WorkOrderDraft(
        asset_id=str(call.arguments.get("asset_id", "")).strip().upper(),
        priority=priority,
        summary=str(call.arguments.get("summary", "")).strip(),
    )


def _run_tool(
    call: ToolCall,
    retriever: Retriever,
    llm: LLMConfig,
    cited: set[str],
    found: list[RetrievedChunk],
) -> str:
    """Execute one read-only tool; errors go back to the model as text
    so it can correct itself instead of the whole request failing."""
    try:
        if call.name == "get_machine_health":
            output = get_machine_health(call.arguments["asset_id"])
        elif call.name == "search_manuals":
            output = _search_manuals(
                call.arguments["query"], retriever, llm, cited, found
            )
        else:
            output = {"error": f"Unknown tool {call.name}"}
    except KeyError as exc:
        output = {"error": f"Missing argument {exc}"}
    return json.dumps(output)


def _keep_known_citations(text: str, cited: set[str]) -> str:
    """Output guardrail for the agent: a citation is kept only if
    search_manuals returned it during this run."""
    def _check(match: re.Match) -> str:
        if match.group(1) in cited:
            return match.group(0)
        logger.warning("Removed unsupported citation [%s]", match.group(1))
        return ""

    return _CITATION_RE.sub(_check, text)


# --- Agent loop -------------------------------------------------------


def run_agent(
    question: str, retriever: Retriever, llm: LLMConfig
) -> AgentResult:
    """Loop: model picks tools -> we run them -> results go back, until
    the model answers, drafts a work order, or hits max_agent_steps."""
    guard = check_input(question)
    if not guard.allowed:
        logger.warning("Agent input blocked | reason=%s", guard.reason)
        return AgentResult(
            answer=BLOCKED_MESSAGES.get(guard.reason, "Request blocked."),
            stopped_reason="blocked",
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[dict[str, Any]] = []
    cited: set[str] = set()
    found: list[RetrievedChunk] = []

    for step in range(1, settings.max_agent_steps + 1):
        turn = chat_with_tools(llm, SYSTEM_PROMPT, messages, TOOLS)
        text = _keep_known_citations(turn.text, cited)
        if not turn.tool_calls:
            logger.info("Agent answered after %d step(s)", step)
            return AgentResult(text, steps, sources=found)

        messages.append(turn.assistant_message)
        results: list[tuple[ToolCall, str]] = []
        for call in turn.tool_calls:
            if call.name == "create_work_order":
                draft = _draft_from_call(call)
                steps.append({"tool": call.name, "arguments": asdict(draft)})
                logger.info("Agent drafted work order | %s", draft)
                answer = text or (
                    f"I drafted a {draft.priority}-priority work order "
                    f"for {draft.asset_id}. It needs your approval."
                )
                return AgentResult(
                    answer, steps, draft, "awaiting_approval", found
                )

            output = _run_tool(call, retriever, llm, cited, found)
            steps.append({
                "tool": call.name,
                "arguments": call.arguments,
                "result": output[:1000],
            })
            results.append((call, output))

        messages.extend(tool_result_messages(llm, results))

    logger.warning("Agent hit max_agent_steps=%d", settings.max_agent_steps)
    return AgentResult(
        "I couldn't finish within the step limit. The passages I found "
        "so far are listed below.",
        steps,
        stopped_reason="max_steps",
        sources=found,
    )


def approve_work_order(draft: WorkOrderDraft, approved_by: str) -> dict:
    """Called only after a human clicks Approve. Stands in for a call to
    the plant's maintenance system (e.g. SAP PM)."""
    record = {
        "work_order_id": f"WO-{uuid.uuid4().hex[:8].upper()}",
        **asdict(draft),
        "approved_by": approved_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    WORK_ORDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with WORK_ORDER_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Work order created | %s", record)
    return record
