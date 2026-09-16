"""Telegram bot: Vervemint in a chat, for people who do not want to open
the web UI.

A thin client like the web UI: every action is a call to the API
(src/vervemint/api.py) through ui/api_client.py, so the bot holds no models
and no documents, and one backend serves both front ends.

What a user can do:
- send a question            -> cited answer (/ask), or the maintenance
                                agent (/agent), depending on /settings
- send a .log / .csv / .txt  -> diagnosis from the manuals (/analyze-log);
                                the caption, if any, is the question
- send a .pdf / .md          -> stored in "My documents" (/documents)
- /settings                  -> mode, LLM provider, what to search
- tap Approve / Reject       -> on a work order the agent drafted

It starts without any credentials and waits. The bot token and the
allowed users are set in the web UI's Settings page (credentials.py):
the process checks every few seconds, connects to Telegram as soon as a
token is saved, reconnects when it changes and disconnects when it is
removed. Its status (waiting, running as @name, token rejected, another
process polling the same token) goes to data/telegram_status.json, which
the Settings page shows live.

Production notes:
- Only allowed Telegram user ids may use the bot, checked on every
  message, so a change in Settings applies at once; anyone else is told
  their id. Private chats only: answers never land in a group.
- Long polling: no public URL or open port needed. Telegram allows one
  poller per token; a second one shows as "conflict" in Settings.
- Updates are handled concurrently, so one slow answer never blocks
  other users; each user has at most one request in flight.
- The provider is the one chosen on the web UI's Chat page, and
  /settings here changes that same choice. No provider key is sent from
  here: the API uses the keys saved in Settings.
- Per-chat settings and pending work orders survive restarts
  (data/telegram_bot.pickle). Bot tokens never appear in the logs.

Run from the project root:
    python -m ui.telegram_bot
"""

import asyncio
import functools
import html
import logging
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Conflict, InvalidToken, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PersistenceInput,
    PicklePersistence,
    filters,
)

# Allow `python ui/telegram_bot.py` as well as `python -m ui.telegram_bot`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vervemint import bot_status, credentials  # noqa: E402
from src.vervemint.config import settings  # noqa: E402
from src.vervemint.logging_config import setup_logging  # noqa: E402
from ui.api_client import ApiClient, ApiError  # noqa: E402
from ui.telegram_format import (  # noqa: E402
    footer,
    format_answer,
    format_work_order,
    plain_text,
    route_document,
    split_message,
)

logger = logging.getLogger(__name__)

PROVIDERS = {"ollama": "Ollama", "gemini": "Gemini", "claude": "Claude",
             "openai": "OpenAI"}
MODES = {"qa": "Ask the documents", "agent": "Maintenance agent"}
SCOPES = {"library": "Panasonic library", "documents": "My documents"}
LOG_QUESTION = ("Analyse the attached log: what is wrong, and how do I fix "
                "it according to the manuals?")
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # Telegram's limit for bot downloads
POLL_SECONDS = 3       # how often the saved token is checked
RETRY_SECONDS = 30     # after failing to reach Telegram
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

HELP_TEXT = (
    "<b>Vervemint</b>\n"
    "AI that mint-answers from your docs, with the sources it used.\n\n"
    "• Send a <b>question</b> as a message.\n"
    "• Send a <b>.log, .csv or .txt</b> machine log to get a diagnosis "
    "(write your question in the caption).\n"
    "• Send a <b>.pdf or .md</b> manual to add it to My documents.\n"
    "• /settings: mode (questions or maintenance agent), model (shared "
    "with the web app), and what to search.\n\n"
    "Sources are numbered [1], [2] and listed under each answer."
)

# Users with a request in flight. Handlers run as tasks on one event
# loop, so a plain set is safe. Not persisted on purpose: a crash must
# not lock anyone out.
_busy: set[int] = set()
# Every bot token seen by this process, so none can reach a log line.
_tokens: set[str] = set()


# --- Access control --------------------------------------------------------


def _allowed(update: Update) -> bool:
    user = update.effective_user
    return (user is not None
            and user.id in credentials.telegram_allowed_users())


async def _refuse(update: Update) -> None:
    user = update.effective_user
    logger.warning("Refused Telegram user %s (@%s)",
                   user.id if user else "?", user.username if user else "?")
    text = ("You are not allowed to use this bot. Your Telegram user id is "
            f"{user.id if user else 'unknown'}: send it to the "
            "administrator to get access.")
    if update.callback_query:
        await update.callback_query.answer(text, show_alert=True)
    elif update.effective_message:
        await update.effective_message.reply_text(text)


def restricted(handler: Callable) -> Callable:
    """Run the handler only for allowed users."""
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not _allowed(update):
            await _refuse(update)
            return
        await handler(update, context)
    return wrapper


# --- Helpers ---------------------------------------------------------------


def _chat_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    data = context.chat_data
    data.setdefault("mode", "qa")
    data.setdefault("scope", "library")
    return data


def _client(context: ContextTypes.DEFAULT_TYPE) -> ApiClient:
    return context.bot_data["client"]


def _llm(provider: str) -> dict:
    # model None: the API uses the model chosen for this provider in the
    # web UI. key None: the API uses the key saved in Settings.
    return {"provider": provider, "model": None, "key": None}


def _scope(client: ApiClient, name: str) -> dict:
    if name == "library":
        return {"scope": "library", "document_ids": []}
    ids = [d["doc_id"] for d in client.documents()]
    if not ids:
        raise ApiError("There are no saved documents yet. Send me a PDF or "
                       "MD file, or search the library (/settings).")
    return {"scope": "documents", "document_ids": ids}


async def _keep_typing(bot: Any, chat_id: int) -> None:
    """Telegram shows 'typing...' for about 5 s per call: repeat it until
    the answer is ready (cancelled by the caller)."""
    while True:
        try:
            await bot.send_chat_action(chat_id=chat_id,
                                       action=ChatAction.TYPING)
        except TelegramError:
            pass  # cosmetic only; never fail a request over it
        await asyncio.sleep(4)


async def _call_api(update: Update, context: ContextTypes.DEFAULT_TYPE,
                    fn: Callable, *args: Any) -> Any:
    """Run a blocking API call off the event loop, one per user, with
    'typing...' shown meanwhile. Returns None if it could not run; the
    user has then already been told why."""
    user_id = update.effective_user.id
    message = update.effective_message
    if user_id in _busy:
        await message.reply_text("I'm still working on your previous "
                                 "request. I'll answer that one first.")
        return None
    _busy.add(user_id)
    typing = asyncio.create_task(
        _keep_typing(context.bot, update.effective_chat.id))
    try:
        return await asyncio.to_thread(fn, *args)
    except ApiError as exc:
        await message.reply_text(str(exc))
        return None
    finally:
        typing.cancel()
        _busy.discard(user_id)


async def _reply(message: Message, text: str,
                 reply_markup: InlineKeyboardMarkup | None = None) -> None:
    """Send Telegram HTML, split to the message limit. If Telegram
    rejects the markup, the same text goes out as plain text."""
    parts = split_message(text)
    for i, part in enumerate(parts):
        markup = reply_markup if i == len(parts) - 1 else None
        try:
            await message.reply_text(part, parse_mode=ParseMode.HTML,
                                     reply_markup=markup,
                                     link_preview_options=_NO_PREVIEW)
        except BadRequest as exc:
            logger.warning("Telegram rejected the HTML (%s); sending plain "
                           "text", exc)
            await message.reply_text(plain_text(part), reply_markup=markup)


async def _send_result(message: Message, context: ContextTypes.DEFAULT_TYPE,
                       result: dict, provider: str, seconds: float) -> None:
    text = format_answer(result, footer(result, PROVIDERS[provider], seconds))
    draft = result.get("pending_work_order")
    if not draft:
        await _reply(message, text)
        return
    # Human in the loop: nothing is created until someone taps Approve.
    order_id = uuid.uuid4().hex[:8]
    context.chat_data.setdefault("work_orders", {})[order_id] = draft
    buttons = InlineKeyboardMarkup([[
        InlineKeyboardButton("Approve",
                             callback_data=f"wo:approve:{order_id}"),
        InlineKeyboardButton("Reject",
                             callback_data=f"wo:reject:{order_id}"),
    ]])
    await _reply(message, f"{text}\n\n{format_work_order(draft)}", buttons)


def _settings_text(chat: dict) -> str:
    provider = credentials.active_provider()
    model = html.escape(credentials.model_for(provider))
    return (
        "<b>Settings</b>\n"
        f"Mode: {MODES[chat['mode']]}\n"
        f"Model: {PROVIDERS[provider]} · {model}\n"
        f"Search in: {SCOPES[chat['scope']]}\n\n"
        "The model is shared with the web app. Tap to change:"
    )


def _settings_keyboard(chat: dict) -> InlineKeyboardMarkup:
    current = {**chat, "provider": credentials.active_provider()}

    def row(key: str, options: dict[str, str]) -> list[InlineKeyboardButton]:
        return [
            InlineKeyboardButton(
                ("✓ " if current[key] == value else "") + label,
                callback_data=f"set:{key}:{value}",
            )
            for value, label in options.items()
        ]
    return InlineKeyboardMarkup([row("mode", MODES),
                                 row("provider", PROVIDERS),
                                 row("scope", SCOPES)])


# --- Handlers --------------------------------------------------------------


@restricted
async def cmd_start(update: Update,
                    context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = _chat_settings(context)
    await _reply(update.effective_message,
                 f"{HELP_TEXT}\n\n{_settings_text(chat)}",
                 _settings_keyboard(chat))


@restricted
async def cmd_settings(update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = _chat_settings(context)
    await _reply(update.effective_message, _settings_text(chat),
                 _settings_keyboard(chat))


@restricted
async def on_settings_button(update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    _, key, value = query.data.split(":", 2)
    options = {"mode": MODES, "provider": PROVIDERS, "scope": SCOPES}
    if value not in options.get(key, {}):
        await query.answer()
        return
    chat = _chat_settings(context)
    if key == "provider":
        # One choice for the web app and the bot: saved on the server.
        try:
            await asyncio.to_thread(_client(context).update_settings,
                                    {"provider": value})
        except ApiError as exc:
            await query.answer(str(exc)[:200], show_alert=True)
            return
    else:
        chat[key] = value
    await query.answer(f"{options[key][value]} selected")
    try:
        await query.edit_message_text(_settings_text(chat),
                                      parse_mode=ParseMode.HTML,
                                      reply_markup=_settings_keyboard(chat))
    except BadRequest:
        pass  # tapped the option that was already selected


@restricted
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = _chat_settings(context)
    client = _client(context)
    mode, provider = chat["mode"], credentials.active_provider()

    def ask() -> dict:
        scope = _scope(client, chat["scope"])
        if mode == "agent":
            return client.agent(_llm(provider), update.message.text, scope)
        return client.ask(_llm(provider), update.message.text, scope)

    start = time.perf_counter()
    result = await _call_api(update, context, ask)
    if result is not None:
        await _send_result(update.message, context, result, provider,
                           time.perf_counter() - start)


@restricted
async def on_document(update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    document = message.document
    name = Path(document.file_name or "file").name
    kind = route_document(name)
    if kind is None:
        await message.reply_text(
            "Send a .log, .csv or .txt machine log to diagnose it, or a "
            ".pdf or .md manual to add it to My documents.")
        return
    if document.file_size and document.file_size > MAX_DOWNLOAD_BYTES:
        await message.reply_text("That file is larger than 20 MB, the most "
                                 "a Telegram bot can download.")
        return
    data = bytes(await (await document.get_file()).download_as_bytearray())
    chat = _chat_settings(context)
    client = _client(context)

    if kind == "document":
        report = await _call_api(update, context, client.upload,
                                 [(name, data)])
        if report is None:
            return
        lines = [f"{name}: already stored, reused" if d["cached"]
                 else f"{name}: added ({d['chunks']} passages)"
                 for d in report["documents"]] + report["errors"]
        if report["documents"]:
            chat["scope"] = "documents"
            lines.append("Now searching My documents. Change it in "
                         "/settings.")
        await message.reply_text("\n".join(lines))
        return

    question = (message.caption or "").strip() or LOG_QUESTION
    provider = credentials.active_provider()

    def analyze() -> dict:
        return client.analyze_log(_llm(provider), question,
                                  _scope(client, chat["scope"]),
                                  data.decode("utf-8", "replace"), name)

    start = time.perf_counter()
    result = await _call_api(update, context, analyze)
    if result is not None:
        await _send_result(message, context, result, provider,
                           time.perf_counter() - start)


@restricted
async def on_work_order_button(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    _, action, order_id = query.data.split(":", 2)
    # pop: a second tap (or a second device) cannot approve it twice.
    draft = context.chat_data.get("work_orders", {}).pop(order_id, None)
    if draft is None:
        await query.answer("This work order was already handled.",
                           show_alert=True)
        return
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    if action != "approve":
        await query.message.reply_text("Work order discarded. Nothing was "
                                       "created.")
        return
    user = update.effective_user
    approved_by = f"{user.full_name} (Telegram {user.id})"
    try:
        record = await asyncio.to_thread(_client(context).approve, draft,
                                         approved_by)
    except ApiError as exc:
        await query.message.reply_text(f"The work order was not created: "
                                       f"{exc}")
        return
    await query.message.reply_text(
        f"Work order {record['work_order_id']} created for "
        f"{record['asset_id']} ({record['priority']} priority)."
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error in the Telegram bot", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Something went wrong on my side. Please try again.")
        except TelegramError:
            pass


# --- Wiring ----------------------------------------------------------------


def build_application(token: str, client: ApiClient,
                      state_file: Path | None = None) -> Application:
    builder = (Application.builder().token(token)
               .concurrent_updates(True))  # a slow answer blocks nobody
    if state_file:
        builder = builder.persistence(PicklePersistence(
            filepath=state_file,
            # Only per-chat settings and drafts; bot_data holds the client.
            store_data=PersistenceInput(bot_data=False, user_data=False,
                                        callback_data=False),
        ))
    app = builder.build()
    app.bot_data["client"] = client
    # New messages in private chats only: no groups, no edited messages.
    private = filters.ChatType.PRIVATE & filters.UpdateType.MESSAGE
    app.add_handler(CommandHandler(["start", "help"], cmd_start,
                                   filters=private))
    app.add_handler(CommandHandler("settings", cmd_settings, filters=private))
    app.add_handler(CallbackQueryHandler(on_settings_button, pattern=r"^set:"))
    app.add_handler(CallbackQueryHandler(on_work_order_button,
                                         pattern=r"^wo:"))
    app.add_handler(MessageHandler(private & filters.Document.ALL,
                                   on_document))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND,
                                   on_text))
    app.add_error_handler(on_error)
    return app


def _on_polling_error(exc: TelegramError) -> None:
    """Errors while fetching updates; PTB retries by itself. Logged only:
    the status stays connected/disconnected."""
    if isinstance(exc, Conflict):
        logger.warning("Another process is polling this bot token; Telegram "
                       "allows one. The backend runs the bot already.")
    else:
        logger.warning("Telegram polling error: %s", exc)


async def _start(token: str, client: ApiClient) -> Application:
    """Connect with this token and start polling. Raises InvalidToken if
    Telegram rejects it."""
    app = build_application(token, client,
                            settings.data_dir / "telegram_bot.pickle")
    await app.initialize()  # asks Telegram who we are: checks the token
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "What this bot does"),
            BotCommand("settings", "Mode, model and what to search"),
            BotCommand("help", "How to use the bot"),
        ])
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES,
                                        error_callback=_on_polling_error)
    except BaseException:
        await _stop(app)
        raise
    return app


async def _stop(app: Application) -> None:
    try:
        if app.updater and app.updater.running:
            await app.updater.stop()
        if app.running:
            await app.stop()
        await app.shutdown()  # also saves the chat settings
    except Exception:
        logger.exception("Error while stopping the bot")


async def supervise(stop: asyncio.Event) -> None:
    """Keep the bot in step with the token saved in Settings, and write
    a status heartbeat, until `stop` is set."""
    client = ApiClient(settings.api_url, credentials.api_token,
                       settings.request_timeout_s)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app: Application | None = None
    token: str | None = None
    state = ("waiting_for_token", "")
    retry_at = 0.0
    try:
        while not stop.is_set():
            wanted = credentials.telegram_bot_token()
            retry = (app is None and state[0] == "error"
                     and time.monotonic() >= retry_at)
            if wanted != token or retry:
                if app is not None:
                    logger.info("Bot token changed: disconnecting")
                    await _stop(app)
                    app = None
                token = wanted
                if not token:
                    state = ("waiting_for_token", "")
                    logger.info("Waiting for a bot token (web UI: Settings)")
                else:
                    _tokens.add(token)
                    bot_status.write("starting")
                    try:
                        app = await _start(token, client)
                        state = ("running", f"@{app.bot.username}")
                        logger.info("Connected to Telegram as @%s",
                                    app.bot.username)
                    except InvalidToken:
                        state = ("invalid_token", "Telegram rejected this "
                                 "bot token. Check it below.")
                        logger.error("Telegram rejected the bot token")
                    except TelegramError as exc:
                        state = ("error", f"Cannot reach Telegram ({exc}). "
                                 f"Retrying in {RETRY_SECONDS} s.")
                        retry_at = time.monotonic() + RETRY_SECONDS
                        logger.warning("Cannot reach Telegram: %s", exc)
            bot_status.write(*state)
            try:
                await asyncio.wait_for(stop.wait(), POLL_SECONDS)
            except TimeoutError:
                pass
    finally:
        if app is not None:
            await _stop(app)
        bot_status.write("not_running", "The bot process stopped.")


class _RedactTokens(logging.Formatter):
    """Wraps a formatter and removes every bot token seen so far from each
    line, tracebacks included: Telegram API URLs contain the token."""

    def __init__(self, inner: logging.Formatter) -> None:
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        text = self._inner.format(record)
        for token in _tokens:
            text = text.replace(token, "[BOT-TOKEN]")
        return text


async def _run() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)  # Docker stop: clean exit
        except (NotImplementedError, RuntimeError):
            pass  # Windows: Ctrl+C raises KeyboardInterrupt instead
    await supervise(stop)


def main() -> None:
    setup_logging("telegram_bot.log")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(
            _RedactTokens(handler.formatter or logging.Formatter()))
    logger.info("Telegram bot starting | API %s", settings.api_url)
    # Python's default crash output goes straight to stderr, past the
    # redacting formatter, and PTB's errors quote the token: log instead.
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Telegram bot stopped")
        sys.exit(1)


if __name__ == "__main__":
    main()
