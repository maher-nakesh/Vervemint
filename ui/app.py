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
LIBRARY_ID = "library"   # the built-in index, in the same list
# What the pen shows for one source: the two buttons, the name
# box, or the delete question.
EDIT_MENU, EDIT_RENAME, EDIT_DELETE = "menu", "rename", "delete"
MODE_QA = "Ask the documents"
MODE_AGENT = "Maintenance agent (tools)"
LOG_TYPES = ["txt", "log", "csv"]
LOG_QUESTION = ("Analyse the attached log: what is wrong, and how do I fix "
                "it according to the manuals?")
RECHECK_AFTER_S = 30  # how long a failed connection check is remembered
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


def _remembered_check(check_key: tuple) -> dict | None:
    """The connection check from earlier in this session. A failure older
    than RECHECK_AFTER_S is dropped: the server it could not reach (a
    stopped Ollama, a provider outage) may be back."""
    remembered = st.session_state.checks.get(check_key)
    if remembered is None:
        return None
    result, checked_at = remembered
    if not result["ok"] and time.time() - checked_at > RECHECK_AFTER_S:
        return None
    return result


def _check_again(bar) -> None:
    """Check now instead of waiting, after starting Ollama or saving a key."""
    if bar.button("Check again", icon=":material/refresh:", width="stretch"):
        ollama_models.clear()  # the model list is cached for 30 s
        st.session_state.checks.clear()
        st.rerun()


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
            _check_again(bar)  # e.g. after starting `ollama serve`
            return None
        if not models:
            bar.badge("No models", icon=":material/error:", color="red")
            bar.caption(f"{conf['ollama_host']} has no models. Pull one, "
                        "e.g. `ollama pull qwen2.5:7b`.")
            _check_again(bar)
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

    # One free check per provider / model / key, kept for this browser
    # session; a failed one is retried so the app recovers by itself once
    # the provider is back.
    key = conf["keys"].get(provider, {}).get("value", "")
    check_key = (provider, model, key[-6:])
    result = _remembered_check(check_key)
    if result is None:
        with bar, st.spinner("Checking..."):
            try:
                result = get_client().connect(provider, model)
            except ApiError as exc:
                result = {"ok": False, "message": str(exc)}
        st.session_state.checks[check_key] = (result, time.time())
    if result["ok"]:
        bar.badge("Connected", icon=":material/check:", color="green")
        return {"provider": provider, "model": model, "key": None}
    bar.badge("Not connected", icon=":material/error:", color="red")
    bar.caption(result["message"])
    _check_again(bar)
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
    # Search what was just uploaded; the library is searched on its own.
    kept = [s for s in st.session_state.get("selected_sources", [])
            if s != LIBRARY_ID]
    st.session_state.selected_sources = list(dict.fromkeys(kept + new_ids))
    st.session_state.uploader_round += 1  # empties the file picker
    st.rerun()


def _sources(client: ApiClient) -> list[dict]:
    """Everything that is already chunked and embedded: the built-in
    index (index.py) first, then the stored documents (doc_store.py),
    newest first. Raises ApiError if the backend cannot be read."""
    sources = []
    library = client.library()
    if library["available"]:
        sources.append({"id": LIBRARY_ID, "kind": "index",
                        "name": library["name"], "chunks": library["chunks"],
                        "added_at": library["built_at"]})
    sources += [{"id": d["doc_id"], "kind": "document",
                 "name": d["filename"], "chunks": d["chunks"],
                 "added_at": d["added_at"]} for d in client.documents()]
    return sources


def _on_pick(source_id: str) -> None:
    """One scope per question: the built-in index is searched on its own,
    so ticking it clears the documents, and ticking a document clears
    it."""
    chosen = list(st.session_state.selected_sources)
    if not st.session_state[f"pick_{source_id}"]:
        chosen = [s for s in chosen if s != source_id]
    elif source_id == LIBRARY_ID:
        chosen = [LIBRARY_ID]
    else:
        chosen = [s for s in chosen if s != LIBRARY_ID] + [source_id]
    st.session_state.selected_sources = chosen


def _open_editor(source_id: str | None, mode: str = EDIT_MENU) -> None:
    st.session_state.editing = source_id
    st.session_state.edit_mode = mode
    if mode == EDIT_RENAME:
        # A fresh name box, starting from the stored name again.
        st.session_state.edit_round += 1


def _apply_rename(client: ApiClient, source: dict, key: str) -> None:
    """Save the name typed in the row, then close the editor."""
    name = st.session_state.get(key, "").strip()
    _open_editor(None)
    if not name or name == source["name"]:
        return  # nothing typed, or the same name: just close
    try:
        if source["kind"] == "index":
            new = client.rename_library(name)["name"]
        else:
            new = client.rename_document(source["id"], name)["filename"]
    except ApiError as exc:
        flash("error", str(exc))
        return
    st.session_state.upload_report = None
    flash("success", f"Renamed to {new}.")


def _delete_source(client: ApiClient, source: dict) -> None:
    try:
        if source["kind"] == "index":
            client.delete_library()
        else:
            client.delete_document(source["id"])
    except ApiError as exc:
        flash("error", str(exc))
        return
    finally:
        _open_editor(None)
    st.session_state.upload_report = None
    flash("success", f"Deleted {source['name']}.")


def _icon_button(bar, icon: str, key: str, hint: str,
                 kind: str = "tertiary") -> bool:
    """A small borderless icon, sitting in the row it belongs to."""
    return bar.button("", icon=icon, type=kind, key=key, help=hint)


def _confirm_delete(client: ApiClient, source: dict) -> None:
    """Deleting is not undone by another click, so it is asked first."""
    source_id = source["id"]
    back = ("embedding its passages again "
            "(`python -m src.vervemint.index`)" if source["kind"] == "index"
            else "uploading the file again")
    with st.sidebar.container(border=True):
        st.caption(f"**Delete {source['name']}?** Getting it back means "
                   f"{back}.")
        row = st.container(horizontal=True)
        if row.button("Delete", type="primary", key=f"yes_{source_id}"):
            _delete_source(client, source)
            st.rerun()
        if row.button("Cancel", key=f"no_{source_id}"):
            _open_editor(None)
            st.rerun()


def _rename_row(client: ApiClient, source: dict) -> None:
    """The row becomes the name box: type over it and press Enter, or
    the tick. A form, so leaving the box saves nothing by itself and
    the cross really cancels."""
    source_id = source["id"]
    key = f"name_{source_id}_{st.session_state.edit_round}"
    with st.sidebar.form(f"rename_{source_id}", border=False):
        row = st.container(horizontal=True, vertical_alignment="center")
        row.text_input("Name", value=source["name"], key=key, width="stretch",
                       label_visibility="collapsed",
                       help="Press Enter to save")
        saved = row.form_submit_button("", icon=":material/check:",
                                       type="primary", key=f"save_{source_id}",
                                       help="Save")
        cancelled = row.form_submit_button("", icon=":material/close:",
                                           type="tertiary",
                                           key=f"cancel_{source_id}",
                                           help="Cancel")
    if saved:
        _apply_rename(client, source, key)
        st.rerun()
    if cancelled:
        _open_editor(None)
        st.rerun()


def _source_row(client: ApiClient, source: dict, selected: list[str]) -> None:
    """One line: a checkbox to search it, and a pen that turns into the
    two things you can do to it."""
    source_id = source["id"]
    mode = (st.session_state.edit_mode
            if st.session_state.editing == source_id else None)
    if mode == EDIT_RENAME:  # the name is being typed over, in place
        _rename_row(client, source)
        return

    # The server's list decides what is ticked, not a stale widget.
    st.session_state[f"pick_{source_id}"] = source_id in selected
    row = st.sidebar.container(horizontal=True, vertical_alignment="center")
    row.checkbox(f"{source['name']} :gray[· {source['chunks']:,}]",
                 key=f"pick_{source_id}", width="stretch",
                 on_change=_on_pick, args=(source_id,),
                 help=f"{source['chunks']:,} passages · added "
                      f"{source['added_at'][:10]}")
    if mode is None:
        if _icon_button(row, ":material/edit:", f"pen_{source_id}",
                        "Rename or delete"):
            _open_editor(source_id)
            st.rerun()
        return

    # The pen was pressed: it makes room for what it opens.
    if _icon_button(row, ":material/drive_file_rename_outline:",
                    f"rename_{source_id}", "Rename"):
        _open_editor(source_id, EDIT_RENAME)
        st.rerun()
    if _icon_button(row, ":material/delete:", f"delete_{source_id}", "Delete"):
        _open_editor(source_id, EDIT_DELETE)
        st.rerun()
    if _icon_button(row, ":material/close:", f"close_{source_id}", "Close"):
        _open_editor(None)
        st.rerun()
    if mode == EDIT_DELETE:
        _confirm_delete(client, source)


def sidebar_sources(client: ApiClient) -> dict | None:
    """Add files, then tick what to search. Everything listed is already
    chunked and embedded: the built-in index and the stored documents.
    Returns the search scope for /ask and /agent, or None."""
    st.sidebar.subheader("Sources")
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
        sources = _sources(client)
    except ApiError as exc:
        st.sidebar.error(str(exc))
        return None
    if not sources:
        st.sidebar.info("Nothing indexed yet. Add files above.")
        return None

    # Default to the documents, else the built-in index; drop anything
    # that was deleted meanwhile.
    ids = [s["id"] for s in sources]
    documents = [s["id"] for s in sources if s["kind"] == "document"]
    selected = [s for s in st.session_state.get("selected_sources",
                                                documents or [LIBRARY_ID])
                if s in ids]
    st.session_state.selected_sources = selected

    st.sidebar.caption("Search in")
    for source in sources:
        _source_row(client, source, selected)

    if not selected:
        st.sidebar.info("Tick at least one source to search.")
        return None
    if LIBRARY_ID in selected:  # _on_pick keeps it on its own
        return {"scope": "library", "document_ids": []}
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
    except ApiError as exc:
        st.error(str(exc), icon=":material/cloud_off:")
        return
    llm = model_picker(conf)
    scope = sidebar_sources(client)
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
                          ("checks", {}), ("editing", None),
                          ("edit_mode", EDIT_MENU), ("edit_round", 0)]:
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
