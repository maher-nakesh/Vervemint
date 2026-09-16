"""Citation parsing, "not in sources" abstention, chunk headers and the
Gemini message format -- no LLM or network calls."""

from src.vervemint import generate
from src.vervemint.ingest import _chunk_page
from src.vervemint.llm import LLMConfig, ToolCall, tool_result_messages
from src.vervemint.retrieve import RetrievedChunk


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(f"c{i}", "text", "manual.pdf", i, "x", 0.9)
        for i in range(1, n + 1)
    ]


def test_grouped_and_invalid_citations_are_parsed():
    text = "Oil is FV50S [S1, S2]. Charge 250 cm3 [S2]. Fake [S9]."
    cleaned, used = generate._validate_citations(text, _chunks(2))
    assert used == ["manual.pdf#p1", "manual.pdf#p2"]
    assert "[manual.pdf#p1][manual.pdf#p2]" in cleaned
    assert "S9" not in cleaned


def test_not_in_sources_reply_becomes_abstention(monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda *a: "NOT_IN_SOURCES")
    answer = generate.generate_answer(
        "q", _chunks(1), LLMConfig("ollama", "m")
    )
    assert answer.abstained and answer.text == generate.NOT_FOUND_MESSAGE


def test_partly_covered_answer_is_kept(monkeypatch):
    # A log diagnosis where only the last point is not in the sources:
    # the cited part must reach the user, not a "not found" message.
    reply = ("Supply voltage 99 V is below the 103 V minimum [S1].\n"
             "NOT_IN_SOURCES: the approved oil is not stated.")
    monkeypatch.setattr(generate, "chat", lambda *a: reply)
    answer = generate.generate_answer(
        "q", _chunks(1), LLMConfig("ollama", "m")
    )
    assert not answer.abstained
    assert answer.citations == ["manual.pdf#p1"]
    assert "103 V minimum [manual.pdf#p1]" in answer.text
    assert "NOT_IN_SOURCES" not in answer.text
    assert "Not covered by the sources. the approved oil" in answer.text


def test_chunks_carry_the_page_title():
    page = (
        "# SPECIFICATION OF COMPRESSOR\n\nModel: DJK51C73RAU\n\n"
        "<table><tr><td>Oil charge</td><td>250</td></tr></table>\n\n"
        "Approved oils: FREOL alpha10"
    )
    chunks = _chunk_page(page, "spec.pdf", 0, "x")
    assert "DJK51C73RAU" in chunks[0].text
    assert all("Model: DJK51C73RAU" in c.text for c in chunks[1:])


def test_gemini_tool_results_are_function_responses():
    call = ToolCall("abc", "get_machine_health", {"asset_id": "P-07"})
    [content] = tool_result_messages(
        LLMConfig("gemini", "m", "k"), [(call, '{"status": "ALARM"}')]
    )
    part = content.parts[0].function_response
    assert content.role == "user" and part.name == "get_machine_health"
    assert part.id == "abc"
    assert part.response == {"result": '{"status": "ALARM"}'}


def test_provider_errors_keep_the_providers_reason():
    from src.vervemint.llm import _error_detail, _status_message

    class FakeError(Exception):
        body = {"type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"}}

    assert _error_detail(FakeError()) == "Overloaded"
    msg = _status_message("Gemini", 503, "The model is overloaded.")
    assert "temporarily unavailable (503)" in msg
    assert "The model is overloaded." in msg
    assert "provider's side" in msg
