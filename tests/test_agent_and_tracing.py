"""Agent tools, guardrails, tracing and LLM logging -- no LLM calls
needed."""

import contextvars
import json
import logging

import ollama
from google.genai import types as genai_types

from src.vervemint import agent, llm
from src.vervemint.llm import LLMConfig, _recover_text_tool_calls
from src.vervemint.logging_config import LLM_IO_LOGGER
from src.vervemint.tracing import set_request_id, summarize


def test_machine_health_known_and_unknown_asset():
    assert agent.get_machine_health(" p-07 ")["status"] == "ALARM"
    unknown = agent.get_machine_health("X-99")
    assert "error" in unknown and "P-07" in unknown["known_assets"]


def test_approved_work_order_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ORDER_FILE", tmp_path / "wo.jsonl")
    draft = agent.WorkOrderDraft("P-07", "high", "Replace bearing")
    record = agent.approve_work_order(draft, approved_by="tester")
    saved = json.loads((tmp_path / "wo.jsonl").read_text())
    assert saved["work_order_id"] == record["work_order_id"]
    assert saved["approved_by"] == "tester"


def test_tool_call_written_as_text_is_recovered():
    text = '{"name": "get_machine_health", "arguments": {"asset_id": "P-07"}}'
    calls = _recover_text_tool_calls(text, {"get_machine_health"})
    assert [(c.name, c.arguments) for c in calls] == [
        ("get_machine_health", {"asset_id": "P-07"})
    ]
    assert _recover_text_tool_calls('{"name": "rm_rf"}', {"x"}) == []


def test_agent_drops_citations_it_never_retrieved():
    text = "Bearing fault [manual.pdf#p3]. Also [made_up#p10]."
    kept = agent._keep_known_citations(text, {"manual.pdf#p3"})
    assert "[manual.pdf#p3]" in kept and "made_up" not in kept


def test_trace_summary():
    traces = [
        {"blocked_reason": None, "abstained": False, "citations": ["a#p1"],
         "latency_ms": {"total_ms": 100}},
        {"blocked_reason": None, "abstained": True, "citations": [],
         "latency_ms": {"total_ms": 300}},
        {"blocked_reason": "prompt_injection", "abstained": False,
         "citations": [], "latency_ms": {}},
    ]
    stats = summarize(traces)
    assert stats["requests"] == 3 and stats["blocked"] == 1
    assert stats["abstention_rate"] == 0.5


def test_llm_prompt_and_reply_are_logged(monkeypatch, caplog):
    usage = {"prompt_tokens": 12, "output_tokens": 5, "tokens_per_s": 40.0}
    monkeypatch.setattr(llm, "_chat_raw",
                        lambda *a: ("Oil is VG 10 [S1].", usage))
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger=LLM_IO_LOGGER)
    # setup_logging() stops llm.log records from reaching the root
    # logger, where caplog listens, so listen on that logger directly.
    io_logger = logging.getLogger(LLM_IO_LOGGER)
    io_logger.addHandler(caplog.handler)

    def call() -> str:
        set_request_id("req123")  # what the API middleware does
        return llm.chat(LLMConfig("ollama", "m"), "RULES", "Which oil?")

    try:
        # A copied context keeps the request id out of other tests.
        assert contextvars.copy_context().run(call) == "Oil is VG 10 [S1]."
    finally:
        io_logger.removeHandler(caplog.handler)
    assert "[req123] >>> ollama / m (chat)" in caplog.text
    assert "RULES" in caplog.text and "Which oil?" in caplog.text
    assert "[req123] <<< ollama / m" in caplog.text
    assert "Oil is VG 10 [S1]." in caplog.text
    assert "[req123] LLM call ok" in caplog.text
    assert "tokens_per_s=40.0" in caplog.text


def test_usage_fields_match_the_sdks():
    ollama_reply = ollama.ChatResponse(
        model="m", message=ollama.Message(role="assistant", content="x"),
        prompt_eval_count=50, eval_count=100,
        eval_duration=2_000_000_000, load_duration=5_000_000,
    )
    assert llm._usage("ollama", ollama_reply) == {
        "prompt_tokens": 50, "output_tokens": 100,
        "tokens_per_s": 50.0, "load_ms": 5,
    }
    gemini_reply = genai_types.GenerateContentResponse(
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=3,
            thoughts_token_count=7,
        )
    )
    assert llm._usage("gemini", gemini_reply) == {
        "prompt_tokens": 10, "output_tokens": 3, "thinking_tokens": 7,
    }


def test_sdk_messages_render_for_the_log():
    content = genai_types.Content(role="model", parts=[
        genai_types.Part(text="Checking", thought_signature=b"\x00\x01")
    ])
    text = llm._render([{"role": "user", "content": "Is P-07 ok?"}, content,
                        {"role": "user", "content": "key AIza" + "C" * 35}])
    assert "----- user -----\nIs P-07 ok?" in text
    assert "----- model -----" in text and "<2 bytes>" in text
    assert "AIza" not in text and "[REDACTED-SECRET]" in text
