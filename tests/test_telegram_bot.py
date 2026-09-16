"""Telegram bot: formatting, handlers and the token lifecycle, with fake
Telegram objects and a fake API client -- no network, no bot token."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from telegram.error import InvalidToken

from src.vervemint import bot_status, credentials
from src.vervemint.config import settings
from ui import telegram_bot as bot
from ui.telegram_format import (
    format_answer,
    markdown_to_html,
    route_document,
    split_message,
)

USER_ID = 42
ANSWER = {
    "request_id": "r1",
    "answer": "Supply is below the 103 V minimum [spec#p1]. Use oil VG 10 "
              "[spec#p0].",
    "citations": ["spec#p1", "spec#p0"],
    "abstained": False,
    "blocked_reason": None,
    "latency_ms": {},
    "sources": [
        {"chunk_id": "c1", "source": "spec", "page_no": 1, "score": 0.99},
        {"chunk_id": "c0", "source": "spec", "page_no": 0, "score": 0.98},
    ],
    "pages": {
        "spec#p1": [{"chunk_id": "c1",
                     "text": "# Compressor DJK51C73RAU\n"
                             "<table>Voltage range 103V to 127V</table>"}],
        "spec#p0": [{"chunk_id": "c0", "text": "<table>Oil VG 10</table>"}],
    },
}
DRAFT = {"asset_id": "P-07", "priority": "high",
         "summary": "Replace the bearing"}


# --- Formatting --------------------------------------------------------------


def test_markdown_becomes_safe_telegram_html():
    html = markdown_to_html("### **Problem**\n* voltage < 103 V\n`P-07`")
    assert html.splitlines() == ["<b>Problem</b>", "• voltage &lt; 103 V",
                                 "<code>P-07</code>"]


def test_answer_gets_numbered_sources():
    text = format_answer(ANSWER, "gemini · 2.0s")
    assert "103 V minimum [1]. Use oil VG 10 [2]." in text
    assert "[1] Compressor DJK51C73RAU (spec, page 1)" in text
    assert "[2] spec, page 0" in text
    assert text.endswith("<i>gemini · 2.0s</i>")


def test_long_answers_are_split_under_the_limit():
    text = "\n\n".join(f"<b>Step {i}</b> " + "x" * 300 for i in range(40))
    parts = split_message(text, limit=1000)
    assert len(parts) > 1 and all(len(p) <= 1000 for p in parts)
    assert all(p.count("<b>") == p.count("</b>") for p in parts)
    assert "\n\n".join(parts) == text
    assert all(len(p) <= 100 for p in split_message("y" * 250, limit=100))


def test_attachments_are_routed_by_extension():
    assert route_document("cr2.LOG") == "log"
    assert route_document("readings.csv") == "log"
    assert route_document("manual.pdf") == "document"
    assert route_document("photo.jpg") is None


# --- Handlers ----------------------------------------------------------------


class FakeMessage:
    def __init__(self, text=None, document=None, caption=None):
        self.text, self.document, self.caption = text, document, caption
        self.replies: list[tuple[str, dict]] = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


class FakeQuery:
    def __init__(self, data, message):
        self.data, self.message = data, message
        self.answers, self.buttons_removed = [], False

    async def answer(self, text=None, show_alert=False):
        self.answers.append(text)

    async def edit_message_reply_markup(self, reply_markup=None):
        self.buttons_removed = True


class FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    def ask(self, llm, question, scope):
        self.calls.append(("ask", llm, question, scope))
        return ANSWER

    def agent(self, llm, question, scope):
        self.calls.append(("agent", llm, question, scope))
        return {**ANSWER, "steps": [], "stopped_reason": "awaiting_approval",
                "pending_work_order": DRAFT}

    def analyze_log(self, llm, question, scope, log_text, log_name):
        self.calls.append(("analyze_log", question, log_text, log_name))
        return {**ANSWER, "steps": []}

    def upload(self, files):
        self.calls.append(("upload", [name for name, _ in files]))
        return {"documents": [{"doc_id": "d1", "chunks": 7, "cached": False}],
                "errors": []}

    def documents(self):
        return [{"doc_id": "d1"}]

    def approve(self, draft, approved_by):
        self.calls.append(("approve", draft, approved_by))
        return {"work_order_id": "WO-1", **draft}


class FakeBot:
    async def send_chat_action(self, **kwargs):
        pass


class FakeDocument:
    def __init__(self, name, data):
        self.file_name, self.file_size, self._data = name, len(data), data

    async def get_file(self):
        data = self._data

        class _File:
            async def download_as_bytearray(self):
                return bytearray(data)
        return _File()


def _update(message=None, query=None, user_id=USER_ID):
    user = SimpleNamespace(id=user_id, username="tech", full_name="Tech One")
    return SimpleNamespace(
        effective_user=user, effective_chat=SimpleNamespace(id=user_id),
        effective_message=message or (query.message if query else None),
        message=message, callback_query=query,
    )


@pytest.fixture
def context(monkeypatch):
    # What the web UI saved: allowed users and the provider choice.
    monkeypatch.setattr(credentials, "telegram_allowed_users",
                        lambda: [USER_ID])
    monkeypatch.setattr(credentials, "active_provider", lambda: "gemini")
    monkeypatch.setattr(credentials, "model_for", lambda provider: "model-x")
    bot._busy.clear()
    return SimpleNamespace(bot=FakeBot(), chat_data={},
                           bot_data={"client": FakeClient()})


def test_unknown_user_is_told_their_id(context):
    message = FakeMessage(text="What oil?")
    asyncio.run(bot.on_text(_update(message, user_id=7), context))
    assert "user id is 7" in message.replies[0][0]
    assert context.bot_data["client"].calls == []


def test_question_gets_cited_answer(context):
    message = FakeMessage(text="What oil does DJK51C73RAU use?")
    asyncio.run(bot.on_text(_update(message), context))
    [(kind, llm, question, scope)] = context.bot_data["client"].calls
    assert kind == "ask" and question == message.text
    assert llm == {"provider": "gemini", "model": None, "key": None}
    assert scope == {"scope": "library", "document_ids": []}
    text, kwargs = message.replies[0]
    assert "Use oil VG 10 [2]" in text and "<b>Sources</b>" in text
    assert kwargs["parse_mode"] == "HTML"


def test_one_request_at_a_time_per_user(context):
    bot._busy.add(USER_ID)
    message = FakeMessage(text="second question")
    asyncio.run(bot.on_text(_update(message), context))
    assert "still working" in message.replies[0][0]
    assert context.bot_data["client"].calls == []


def test_agent_work_order_needs_one_approval(context):
    context.chat_data["mode"] = "agent"
    message = FakeMessage(text="Check P-07")
    asyncio.run(bot.on_text(_update(message), context))
    text, kwargs = message.replies[-1]
    assert "awaiting your approval" in text
    approve = kwargs["reply_markup"].inline_keyboard[0][0].callback_data
    assert approve.startswith("wo:approve:")

    query = FakeQuery(approve, FakeMessage())
    asyncio.run(bot.on_work_order_button(_update(query=query), context))
    kind, draft, approved_by = context.bot_data["client"].calls[-1]
    assert kind == "approve" and draft == DRAFT
    assert approved_by == f"Tech One (Telegram {USER_ID})"
    assert query.buttons_removed
    assert "WO-1 created" in query.message.replies[0][0]

    again = FakeQuery(approve, FakeMessage())
    asyncio.run(bot.on_work_order_button(_update(query=again), context))
    assert again.answers == ["This work order was already handled."]
    assert len(context.bot_data["client"].calls) == 2  # no second approve


def test_log_file_is_diagnosed_with_the_caption(context):
    message = FakeMessage(document=FakeDocument("cr2.log", b"WARN 99 V"),
                          caption="Why did it trip?")
    asyncio.run(bot.on_document(_update(message), context))
    call = context.bot_data["client"].calls[0]
    assert call == ("analyze_log", "Why did it trip?", "WARN 99 V", "cr2.log")
    assert "log analysis" in message.replies[0][0]


def test_pdf_is_stored_and_searched(context):
    message = FakeMessage(document=FakeDocument("manual.pdf", b"%PDF-1.4"))
    asyncio.run(bot.on_document(_update(message), context))
    assert context.bot_data["client"].calls == [("upload", ["manual.pdf"])]
    assert context.chat_data["scope"] == "documents"
    assert "added (7 passages)" in message.replies[0][0]


def test_application_wires_every_handler():
    app = bot.build_application("123456:TEST-TOKEN", FakeClient())
    assert len(app.handlers[0]) == 6
    assert isinstance(app.bot_data["client"], FakeClient)


def test_bot_tokens_never_reach_the_logs(monkeypatch):
    # Both the old and the new token, after a change in Settings.
    old, new = "123456:OLD-SECRET-TOKEN", "654321:NEW-SECRET-TOKEN"
    monkeypatch.setattr(bot, "_tokens", {old, new})
    formatter = bot._RedactTokens(logging.Formatter("%(message)s"))
    record = logging.LogRecord(
        "x", logging.ERROR, "", 0,
        f"POST https://api.telegram.org/bot{old}/x then bot{new}/y",
        None, None)
    line = formatter.format(record)
    assert old not in line and new not in line and "[BOT-TOKEN]" in line


def test_settings_button_changes_the_shared_provider(context):
    saved = []
    context.bot_data["client"].update_settings = saved.append
    query = FakeQuery("set:provider:claude", FakeMessage())
    query.edited = None

    async def edit(text, **kwargs):
        query.edited = text
    query.edit_message_text = edit
    asyncio.run(bot.on_settings_button(_update(query=query), context))
    assert saved == [{"provider": "claude"}]  # the web UI's choice too
    assert "provider" not in context.chat_data


def test_bot_waits_for_a_token_and_follows_changes(tmp_path, monkeypatch):
    """No token: waiting. Token saved in Settings: connects. Token
    changed: reconnects. Bad token: says so. Token removed: waits."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(bot, "POLL_SECONDS", 0.01)
    saved = {"token": None}
    monkeypatch.setattr(credentials, "telegram_bot_token",
                        lambda: saved["token"])
    started, stopped = [], []

    async def fake_start(token, client):
        if token == "999:bad":
            raise InvalidToken("rejected")
        started.append(token)
        return SimpleNamespace(bot=SimpleNamespace(username=f"bot{token[0]}"))

    async def fake_stop(app):
        stopped.append(app.bot.username)

    monkeypatch.setattr(bot, "_start", fake_start)
    monkeypatch.setattr(bot, "_stop", fake_stop)

    async def scenario() -> list[tuple[str, str]]:
        stop = asyncio.Event()
        task = asyncio.create_task(bot.supervise(stop))
        seen = []
        for token in (None, "1:first", "2:second", "999:bad", None):
            saved["token"] = token
            await asyncio.sleep(0.1)
            status = bot_status.read()
            seen.append((status["state"], status["detail"]))
        stop.set()
        await task
        return seen

    seen = asyncio.run(scenario())
    assert seen == [
        ("waiting_for_token", ""),
        ("running", "@bot1"),
        ("running", "@bot2"),
        ("invalid_token", "Telegram rejected this bot token. Check it "
                          "below."),
        ("waiting_for_token", ""),
    ]
    assert started == ["1:first", "2:second"]
    assert stopped == ["bot1", "bot2"]
    assert bot_status.read()["state"] == "not_running"
