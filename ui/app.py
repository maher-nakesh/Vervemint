"""Vervemint web UI (Streamlit): AI that mint-answers from your docs.

A thin client: every action is a call to the FastAPI backend
(src/vervemint/api.py) through ui/api_client.py. The UI holds no models
and no documents, so it starts instantly and one backend serves the web
UI, the Telegram bot and other tools at the same time.

Two pages, chosen in the sidebar menu:
- Chat      ask, attach a machine log, or run the maintenance agent. The
            sidebar holds the provider and model (saved on the server, so
            the Telegram bot follows the choice), the documents and the mode
- Settings  provider API keys, the Ollama address, the Telegram bot, the
            web UI password and the API token (credentials.py)

Start the backend first, then the UI, from the project root:
    python -m uvicorn src.vervemint.api:app --host 127.0.0.1 --port 8000
    python -m streamlit run ui/app.py
"""

import hmac
import logging
import sys
import time
from pathlib import Path

import streamlit as st

# Streamlit runs this file directly, so make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vervemint import credentials  # noqa: E402
from src.vervemint.config import settings  # noqa: E402
from src.vervemint.logging_config import setup_logging  # noqa: E402
from ui.api_client import ApiClient, ApiError  # noqa: E402
from ui.sources import build_entries, render_entries  # noqa: E402

setup_logging("ui.log")
logger = logging.getLogger("ui")

APP_NAME = "Vervemint"
TAGLINE = "AI that mint-answers from your docs"

st.set_page_config(page_title=APP_NAME, page_icon="🌿", layout="wide")

PROVIDERS = {
    "ollama": "Ollama (local)",
    "gemini": "Gemini",
    "claude": "Claude",
    "openai": "OpenAI",
}
KB_LIBRARY = "Built-in Panasonic manual library"
KB_DOCUMENTS = "My documents"
MODE_QA = "Ask the documents"
MODE_AGENT = "Maintenance agent (tools)"
LOG_TYPES = ["txt", "log", "csv"]
LOG_QUESTION = ("Analyse the attached log: what is wrong, and how do I fix "
                "it according to the manuals?")
_pages: dict[str, st.Page] = {}


@st.cache_resource
def get_client() -> ApiClient:
    # credentials.api_token is read on every request: the token the API
    # generated on its first start, or a newer one after a rotation.
    return ApiClient(settings.api_url, credentials.api_token,
                     settings.request_timeout_s)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models() -> list[str]:
    return get_client().ollama_models()


@st.cache_data(ttl=300, show_spinner=False)
def simulated_assets() -> list[dict]:
    return get_client().assets()


def flash(kind: str, text: str) -> None:
    """A message that survives the rerun after a change."""
    st.session_state.flash.append((kind, text))


def show_flash() -> None:
    for kind, text in st.session_state.flash:
        getattr(st, kind)(text)
    st.session_state.flash = []


def save_settings(changes: dict) -> dict | None:
    try:
        return get_client().update_settings(changes)
    except ApiError as exc:
        flash("error", str(exc))
        return None


# --- Login ---------------------------------------------------------------


def login_gate(client: ApiClient) -> bool:
    """When the UI password is required (on in Docker), ask for it once
    per browser session. The password itself is in Settings."""
    if not settings.ui_require_password or st.session_state.authenticated:
        return True
    st.title(APP_NAME)
    st.caption(TAGLINE)
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    st.caption("First start: the generated password is in the API log "
               "(`docker compose logs api | grep password`). Change it in "
               "Settings.")
    if submitted:
        try:
            expected = client.get_settings()["ui_password"]
        except ApiError as exc:
            st.error(str(exc))
            return False
        if expected and hmac.compare_digest(password.encode(),
                                            expected.encode()):
            st.session_state.authenticated = True
            st.rerun()
        time.sleep(1)  # slows down guessing
        st.error("Wrong password.")
    return False


# --- Chat page: provider and model ----------------------------------------


def _sync(key: str, value: str) -> None:
    """Show the server's value in a widget, unless the user just changed
    it (then the server already has it). Keeps every browser session and
    the Telegram bot on the same choice."""
    st.session_state[key] = value


def _on_provider() -> None:
    save_settings({"provider": st.session_state.provider_choice})


def _on_model(provider: str) -> None:
    model = st.session_state[f"model_{provider}"].strip()
    if model:
        save_settings({"models": {provider: model}})


def model_picker(conf: dict) -> dict | None:
    """Provider, model and a small connection badge, in the sidebar. The
    choice is saved on the server, so the Telegram bot follows it.
    Returns {"provider", "model", "key"} once usable."""
    bar = st.sidebar
    bar.subheader("Model")
    _sync("provider_choice", conf["provider"])
    provider = bar.selectbox("Provider", list(PROVIDERS),
                             format_func=PROVIDERS.get, key="provider_choice",
                             on_change=_on_provider,
                             label_visibility="collapsed")
    model = conf["models"][provider]

    if provider == "ollama":
        try:
            models = ollama_models()
        except ApiError as exc:
            bar.badge("Not connected", icon=":material/error:", color="red")
            bar.caption(str(exc))
            return None
        if not models:
            bar.badge("No models", icon=":material/error:", color="red")
            bar.caption(f"{conf['ollama_host']} has no models. Pull one, "
                        "e.g. `ollama pull qwen2.5:7b`.")
            return None
        if model not in models:
            # Use a model that exists, and save it so the bot uses it too.
            model = models[0]
            save_settings({"models": {"ollama": model}})
        _sync("model_ollama", model)
        bar.selectbox("Model", models, key="model_ollama",
                      on_change=_on_model, args=("ollama",),
                      label_visibility="collapsed")
    else:
        if not conf["keys"][provider]["value"]:
            bar.badge("No API key", icon=":material/key_off:", color="orange")
            bar.page_link(_pages["settings"], label="Add it in Settings",
                          icon=":material/settings:")
            return None
        _sync(f"model_{provider}", model)
        bar.text_input("Model", key=f"model_{provider}", on_change=_on_model,
                       args=(provider,), label_visibility="collapsed",
                       placeholder="Model name", help="Press Enter to save.")

    # One free check per provider / model / key per browser session.
    key = conf["keys"].get(provider, {}).get("value", "")
    check_key = (provider, model, key[-6:])
    result = st.session_state.checks.get(check_key)
    if result is None:
        with bar, st.spinner("Checking..."):
            try:
                result = get_client().connect(provider, model)
            except ApiError as exc:
                result = {"ok": False, "message": str(exc)}
        st.session_state.checks[check_key] = result
    if result["ok"]:
        bar.badge("Connected", icon=":material/check:", color="green")
        return {"provider": provider, "model": model, "key": None}
    bar.badge("Not connected", icon=":material/error:", color="red")
    bar.caption(result["message"])
    if bar.button("Retry", icon=":material/refresh:", width="stretch"):
        st.session_state.checks.pop(check_key, None)
        st.rerun()
    return None


# --- Chat page: sidebar ---------------------------------------------------


def _upload(client: ApiClient, files) -> None:
    with st.sidebar, st.spinner("Uploading and indexing..."):
        try:
            report = client.upload([(f.name, f.getvalue()) for f in files])
        except ApiError as exc:
            st.sidebar.error(str(exc))
            return
    st.session_state.upload_report = report
    new_ids = [d["doc_id"] for d in report["documents"]]
    st.session_state.selected_docs = list(
        dict.fromkeys(st.session_state.get("selected_docs", []) + new_ids)
    )
    st.session_state.uploader_round += 1  # empties the file picker
    st.rerun()


def sidebar_documents(client: ApiClient, health: dict) -> dict | None:
    """Returns the search scope for /ask and /agent, or None."""
    st.sidebar.subheader("Documents")
    options = [KB_DOCUMENTS]
    if health["library"]["available"]:
        options.insert(0, KB_LIBRARY)
    choice = st.sidebar.radio("Search in", options)
    if choice == KB_LIBRARY:
        st.sidebar.caption(f"{health['library']['chunks']:,} passages from "
                           "the Panasonic manuals")
        return {"scope": "library", "document_ids": []}

    files = st.sidebar.file_uploader(
        "Add PDF, TXT or MD files", type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_round}",
    )
    if files and st.sidebar.button("Upload and index", type="primary"):
        _upload(client, files)

    report = st.session_state.upload_report
    if report:
        for doc in report["documents"]:
            if doc["cached"]:
                st.sidebar.info(f"{doc['filename']}: already stored, reused "
                                "without reprocessing")
            else:
                st.sidebar.success(f"{doc['filename']}: indexed "
                                   f"({doc['chunks']} passages)")
        for error in report["errors"]:
            st.sidebar.error(error)

    try:
        docs = client.documents()
    except ApiError as exc:
        st.sidebar.error(str(exc))
        return None
    if not docs:
        st.sidebar.info("No documents yet. Add files above.")
        return None

    labels = {d["doc_id"]: f"{d['filename']} ({d['chunks']} passages)"
              for d in docs}
    # Default to every document; drop ones that were deleted elsewhere.
    st.session_state.selected_docs = [
        d for d in st.session_state.get("selected_docs", list(labels))
        if d in labels
    ]
    selected = st.sidebar.multiselect(
        "Search these saved documents", list(labels),
        format_func=labels.get, key="selected_docs",
    )
    with st.sidebar.expander("Delete a saved document"):
        doomed = st.selectbox("Document", list(labels),
                              format_func=labels.get)
        if st.button("Delete", icon=":material/delete:"):
            try:
                client.delete_document(doomed)
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state.upload_report = None
                st.rerun()
    if not selected:
        st.sidebar.info("Select at least one document to search.")
        return None
    return {"scope": "documents", "document_ids": selected}


def sidebar_rest() -> str:
    st.sidebar.subheader("Mode")
    mode = st.sidebar.radio("Mode", [MODE_QA, MODE_AGENT],
                            label_visibility="collapsed")
    if mode == MODE_AGENT:
        assets = ", ".join(f"{a['asset_id']} ({a['status']})"
                           for a in simulated_assets())
        st.sidebar.caption(f"Simulated machine data for: {assets}")
    st.sidebar.divider()
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_work_order = None
    if settings.ui_require_password and st.sidebar.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()
    st.sidebar.caption(f"Backend: {get_client().base_url}")
    return mode


# --- Chat page: messages -------------------------------------------------


def to_message(result: dict, llm: dict, seconds: float) -> dict:
    """Plain data for the chat history, built from an API response."""
    text, entries = build_entries(result["answer"], result["sources"],
                                  result["pages"])
    meta = f"{llm['provider']} / {llm['model']} / {seconds:.1f}s"
    if "stopped_reason" in result:
        meta = "agent / " + meta
        if result["stopped_reason"] != "answered":
            meta += f" / {result['stopped_reason'].replace('_', ' ')}"
    elif result["blocked_reason"]:
        meta = "Blocked by input guardrail"
    elif result["abstained"] and "steps" in result:
        meta = "log analysis / " + meta + " / not enough evidence"
    elif "steps" in result:
        meta = "log analysis / " + meta
    elif result["abstained"]:
        meta += " / not enough evidence"
    return {"role": "assistant", "content": text, "entries": entries,
            "meta": meta}


def render_message(msg: dict, key: str) -> None:
    """`key` must be stable per message: it keys the source buttons."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("attachment"):
            st.caption(f"Attached: {msg['attachment']}")
        if msg.get("entries"):
            render_entries(msg["entries"], key)
        if msg.get("meta"):
            st.caption(msg["meta"])


def render_pending_work_order(client: ApiClient) -> None:
    """Human-in-the-loop: the agent drafted it, a person decides."""
    draft = st.session_state.pending_work_order
    if draft is None:
        return
    with st.container(border=True):
        st.markdown("**Work order awaiting your approval**")
        st.markdown(f"Asset **{draft['asset_id']}**, priority "
                    f"**{draft['priority']}**")
        st.markdown(draft["summary"])
        approve, reject, _ = st.columns([1, 1, 4])
        if approve.button("Approve", type="primary"):
            try:
                record = client.approve(draft, "technician (UI)")
            except ApiError as exc:
                st.error(str(exc))
                return
            note = (f"Work order **{record['work_order_id']}** created for "
                    f"{record['asset_id']} ({record['priority']} priority).")
        elif reject.button("Reject"):
            note = "Work order discarded. Nothing was created."
        else:
            return
    st.session_state.messages.append({"role": "assistant", "content": note})
    st.session_state.pending_work_order = None
    st.rerun()


def answer(client: ApiClient, llm: dict, scope: dict, mode: str,
           question: str, log: tuple[str, str] | None) -> dict | None:
    """Call /analyze-log (a log is attached), /agent or /ask."""
    start = time.perf_counter()
    try:
        if log:
            result = client.analyze_log(llm, question, scope, log[1], log[0])
        elif mode == MODE_AGENT:
            result = client.agent(llm, question, scope)
            st.session_state.pending_work_order = result["pending_work_order"]
        else:
            result = client.ask(llm, question, scope)
    except ApiError as exc:
        st.error(str(exc))
        return None
    return to_message(result, llm, time.perf_counter() - start)


def chat_page() -> None:
    client = get_client()
    st.title(APP_NAME)
    st.caption(f"{TAGLINE}. Ask about your technical documents, switch to "
               "the maintenance agent to check machines, or attach a "
               "machine log (.txt, .log, .csv) for a diagnosis.")
    show_flash()
    try:
        conf = client.get_settings()
        health = client.health()
    except ApiError as exc:
        st.error(str(exc), icon=":material/cloud_off:")
        return
    llm = model_picker(conf)
    scope = sidebar_documents(client, health)
    mode = sidebar_rest()

    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, f"m{i}")
    render_pending_work_order(client)

    if scope is None:
        st.info("Choose documents to search in the sidebar to start.")
    elif llm is None:
        st.info("Choose a working model in the sidebar to start.")

    submitted = st.chat_input(
        "Ask a question, or attach a log with the paperclip",
        accept_file=True, file_type=LOG_TYPES,
        disabled=scope is None or llm is None,
    )
    if not submitted:
        return

    question = (submitted.text or "").strip()
    log = None
    if submitted.files:
        attached = submitted.files[0]
        log = (attached.name, attached.getvalue().decode("utf-8", "replace"))
        question = question or LOG_QUESTION
    if not question:
        return

    user_msg = {"role": "user", "content": question}
    if log:
        user_msg["attachment"] = (f"{log[0]} "
                                  f"({len(log[1].splitlines())} lines)")
    st.session_state.messages.append(user_msg)
    render_message(user_msg, f"m{len(st.session_state.messages) - 1}")

    with st.spinner("Working on it..."):
        msg = answer(client, llm, scope, mode, question, log)
    if msg is None:
        return
    st.session_state.messages.append(msg)
    render_message(msg, f"m{len(st.session_state.messages) - 1}")
    if st.session_state.pending_work_order:
        st.rerun()  # show the approval card below the new message


# --- Settings page --------------------------------------------------------


def _setting_row(label: str, note: str):
    """One compact row: name and note on the left, the input on the
    right. Returns the right-hand column."""
    left, right = st.columns([1, 2.4], vertical_alignment="center")
    left.markdown(f"**{label}**")
    left.caption(note)
    return right


def _note(secret: dict, env_var: str, unset: str) -> str:
    if secret["locked"]:
        return f"Set by `{env_var}`"
    return "Saved" if secret["value"] else unset


def _kind(show: bool) -> str:
    return "default" if show else "password"


def providers_section(conf: dict, show: bool) -> None:
    st.subheader("Providers")
    st.caption("Saved keys stay on the server. Pick the provider and the "
               "model in the sidebar.")
    with st.form("providers", border=True):
        typed: dict[str, str] = {}
        for provider in credentials.KEY_PROVIDERS:
            secret = conf["keys"][provider]
            right = _setting_row(
                PROVIDERS[provider],
                _note(secret, credentials.ENV_VARS[provider], "No key yet"))
            typed[provider] = right.text_input(
                PROVIDERS[provider], value=secret["value"], type=_kind(show),
                placeholder="API key", label_visibility="collapsed",
                disabled=secret["locked"])
        locked_host = conf["ollama_host_locked"]
        right = _setting_row(
            "Ollama",
            f"Set by `{credentials.ENV_VARS['ollama_host']}`" if locked_host
            else "Server address, no key")
        host = right.text_input(
            "Ollama address", value=conf["ollama_host"],
            placeholder=credentials.DEFAULT_OLLAMA_HOST,
            label_visibility="collapsed", disabled=locked_host)
        saved = st.form_submit_button("Save", type="primary")
    if not saved:
        return

    changes: dict = {}
    keys = {p: value.strip() for p, value in typed.items()
            if not conf["keys"][p]["locked"]
            and value.strip() != conf["keys"][p]["value"]}
    if keys:
        changes["keys"] = keys
    if not locked_host and host.strip() != conf["ollama_host"]:
        changes["ollama_host"] = host.strip()
    if not changes:
        flash("info", "Nothing changed.")
        st.rerun()
    new = save_settings(changes)
    if new:
        st.session_state.checks.clear()  # the sidebar checks again
        flash("success", " · ".join(_saved_report(new, changes)))
    st.rerun()


def _saved_report(new: dict, changes: dict) -> list[str]:
    """What happened, with a free check of every key that was set."""
    lines = []
    for provider in changes.get("keys", {}):
        name = PROVIDERS[provider]
        if not new["keys"][provider]["value"]:
            lines.append(f"{name} key removed")
            continue
        result = get_client().connect(provider, new["models"][provider])
        lines.append(f"{name}: {result['message']}")
    if "ollama_host" in changes:
        lines.append(f"Ollama address saved: {new['ollama_host']}")
    return lines


@st.fragment(run_every="3s")
def bot_badge() -> None:
    """Refreshes by itself, so the bot shows as connected within seconds
    of a token being saved."""
    try:
        status = get_client().telegram_status()
    except ApiError:
        status = {"state": "unknown", "detail": ""}
    if status["state"] == "running":
        st.badge(f"Connected {status['detail']}", icon=":material/check:",
                 color="green")
        return
    st.badge("Disconnected", icon=":material/link_off:", color="gray")
    if status["state"] == "invalid_token":
        st.caption("Telegram rejected this token.")


def telegram_section(conf: dict, show: bool) -> None:
    st.subheader("Telegram bot")
    st.caption("Runs with the backend. Paste a token from @BotFather and it "
               "connects within seconds; it answers with the model chosen "
               "in the sidebar.")
    bot_badge()
    telegram = conf["telegram"]
    users_locked = telegram["allowed_users_locked"]
    with st.form("telegram", border=True):
        right = _setting_row(
            "Bot token",
            _note(telegram["token"],
                  credentials.ENV_VARS["telegram_bot_token"],
                  "From @BotFather"))
        token = right.text_input(
            "Bot token", value=telegram["token"]["value"], type=_kind(show),
            placeholder="123456789:AAE...", label_visibility="collapsed",
            disabled=telegram["token"]["locked"])
        right = _setting_row(
            "Allowed users",
            f"Set by `{credentials.ENV_VARS['telegram_allowed_users']}`"
            if users_locked else "Telegram user ids, comma separated")
        users = right.text_input(
            "Allowed users",
            value=", ".join(str(u) for u in telegram["allowed_users"]),
            placeholder="123456789, 987654321",
            label_visibility="collapsed", disabled=users_locked)
        saved = st.form_submit_button("Save", type="primary")
    if not saved:
        return

    changes: dict = {}
    if (not telegram["token"]["locked"]
            and token.strip() != telegram["token"]["value"]):
        changes["telegram_bot_token"] = token.strip()
    if not users_locked:
        try:
            ids = credentials.parse_user_ids(users)
        except ValueError:
            st.error("Telegram user ids are numbers, e.g. 123456789.")
            return
        if ids != telegram["allowed_users"]:
            changes["telegram_allowed_users"] = ids
    if not changes:
        flash("info", "Nothing changed.")
    elif save_settings(changes):
        flash("success", "Telegram settings saved.")
    st.rerun()


def access_section(conf: dict, show: bool) -> None:
    st.subheader("Access")
    st.caption("The web UI and the Telegram bot use the API token by "
               "themselves; other clients send it as a Bearer token.")
    with st.form("access", border=True):
        right = _setting_row(
            "Web UI password",
            f"Set by `{credentials.ENV_VARS['ui_password']}`"
            if conf["ui_password_locked"] else
            ("Asked when this UI opens" if settings.ui_require_password
             else "Off: set ui_require_password to ask for it"))
        password = right.text_input(
            "Web UI password", value=conf["ui_password"], type=_kind(show),
            label_visibility="collapsed",
            disabled=conf["ui_password_locked"])
        right = _setting_row(
            "API token",
            f"Set by `{credentials.ENV_VARS['api_token']}`"
            if conf["api_token_locked"] else "Generated on the first start")
        right.text_input("API token", value=conf["api_token"],
                         type=_kind(show), label_visibility="collapsed",
                         disabled=True)
        save_col, pw_col, token_col = st.columns(3)
        saved = save_col.form_submit_button("Save", type="primary",
                                            width="stretch")
        new_password = pw_col.form_submit_button(
            "New password", width="stretch",
            disabled=conf["ui_password_locked"])
        new_token = token_col.form_submit_button(
            "New API token", width="stretch",
            disabled=conf["api_token_locked"])

    if new_password and save_settings({"new_ui_password": True}):
        flash("success", "New password generated.")
        st.rerun()
    if new_token and save_settings({"new_api_token": True}):
        flash("success", "New API token. The web UI and the Telegram bot "
                         "use it already; update other clients.")
        st.rerun()
    if not saved:
        return
    if conf["ui_password_locked"] or password == conf["ui_password"]:
        flash("info", "Nothing changed.")
    elif len(password) < 8:
        st.error("Use at least 8 characters.")
        return
    elif save_settings({"ui_password": password}):
        flash("success", "Password saved.")
    st.rerun()


def settings_page() -> None:
    header, toggle = st.columns([3, 1], vertical_alignment="center")
    header.title("Settings")
    show = toggle.toggle("Show saved values", key="show_secrets")
    st.caption("Saved on the server: the API, the web UI and the Telegram "
               "bot pick up a change at once.")
    show_flash()
    try:
        conf = get_client().get_settings()
    except ApiError as exc:
        st.error(str(exc), icon=":material/cloud_off:")
        return
    providers_section(conf, show)
    telegram_section(conf, show)
    access_section(conf, show)


# --- App -----------------------------------------------------------------


def main() -> None:
    for name, default in [("messages", []), ("pending_work_order", None),
                          ("upload_report", None), ("uploader_round", 0),
                          ("authenticated", False), ("flash", []),
                          ("checks", {})]:
        st.session_state.setdefault(name, default)

    client = get_client()
    try:
        client.health()
    except ApiError as exc:
        st.title(APP_NAME)
        st.error(str(exc), icon=":material/cloud_off:")
        return
    if not login_gate(client):
        return

    _pages["chat"] = st.Page(chat_page, title="Chat", icon=":material/chat:",
                             default=True)
    _pages["settings"] = st.Page(settings_page, title="Settings",
                                 icon=":material/settings:")
    # In the sidebar menu, above the model and document controls.
    st.navigation(list(_pages.values()), position="sidebar").run()


# Streamlit runs this file as __main__; importing it (tests) runs nothing.
if __name__ == "__main__":
    main()
