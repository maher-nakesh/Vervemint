"""REST API: the one backend for the web UI, the Telegram bot and any
other client (a ticketing tool, a plant dashboard, a script).

Run from the project root, with ONE worker: the models live on one GPU
and every extra worker would load its own copy.
    uvicorn src.vervemint.api:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs

Production concerns handled here:
- Auth: every endpoint except /health requires the header
  `Authorization: Bearer <token>`. The token is generated on the first
  start (credentials.py); the web UI and the bot read it from the shared
  store, and the Settings page shows it for other clients.
- Credentials: GET/PUT /settings is how the web UI's Settings page reads
  and changes provider keys, the Telegram bot and passwords. Values set
  in the environment are shown as locked and cannot be changed here.
- LLM keys: the ones saved in Settings, unless a request brings its own
  in the X-LLM-API-Key header (never in the JSON body, never logged).
- Every response carries an X-Request-ID and every request writes one
  access-log line. Unexpected errors are logged with a traceback and
  returned as a 500 that names the request id, never as a stack trace.
- Uploads are size- and type-checked; each file is processed once and
  stored (doc_store.py), then reused.
- Bad input gets a 4xx with a clear message; a failing model provider
  gets a 502 with the provider's reason.
- Endpoints are plain `def`: the work is blocking (GPU inference, LLM
  calls), so FastAPI runs them in its thread pool.
"""

import asyncio
import hmac
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.vervemint import bot_status, credentials, doc_store
from src.vervemint.agent import (
    PRIORITIES,
    SIMULATED_ASSETS,
    WorkOrderDraft,
    approve_work_order,
    run_agent,
)
from src.vervemint.config import settings
from src.vervemint.index import library_is_current
from src.vervemint.ingest import SUPPORTED_UPLOAD_TYPES
from src.vervemint.llm import (
    LLMConfig,
    LLMError,
    Provider,
    check_connection,
)
from src.vervemint.llm import ollama_models as llm_ollama_models
from src.vervemint.machine_logs import analyze_log
from src.vervemint.logging_config import setup_logging
from src.vervemint.pipeline import ask
from src.vervemint.retrieve import RetrievedChunk, Retriever, load_models
from src.vervemint.tracing import load_traces, set_request_id, summarize

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9-]{1,64}")
# "<bot id>:<secret>", as @BotFather hands it out.
_TELEGRAM_TOKEN_RE = re.compile(r"\d{5,}:[A-Za-z0-9_-]{30,}")
_MODEL_NAME_RE = re.compile(r"[\w.:/-]{1,200}")
_MAX_CACHED_RETRIEVERS = 8

# Loaded once at startup, shared by every request.
_state: dict[str, Any] = {}
# Retrievers over sets of stored documents, keyed by the sorted doc ids.
_doc_retrievers: dict[tuple[str, ...], Retriever] = {}
_cache_lock = threading.Lock()


async def _start_telegram_bot() -> tuple[Any, Any] | None:
    """Run the Telegram bot in this process, so starting the backend is
    all it takes. It waits for a token to be saved in Settings."""
    if not settings.telegram_bot:
        return None
    try:
        from ui.telegram_bot import supervise
    except ImportError:
        logger.exception("Telegram bot not started (python-telegram-bot "
                         "missing)")
        return None
    stop = asyncio.Event()
    return stop, asyncio.create_task(supervise(stop))


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    credentials.bootstrap()  # first start: API token + UI password
    bot = await _start_telegram_bot()
    _state["embedder"], _state["reranker"] = load_models()
    _state["library"] = None
    if (settings.index_dir / "dense.faiss").exists():
        _state["library"] = Retriever.from_disk(
            _state["embedder"], _state["reranker"]
        )
        # A deployed index usually ships without its 2 GB source dataset;
        # freshness can only be checked where the dataset is present.
        if settings.corpus_dir.exists() and not library_is_current():
            logger.warning("Library index is out of date. Rebuild it with: "
                           "python -m src.vervemint.index")
    else:
        logger.warning("No library index. Build it with: "
                       "python -m src.vervemint.index")
    yield
    if bot:
        stop, task = bot
        stop.set()
        await task
    _state.clear()
    _doc_retrievers.clear()


app = FastAPI(title="Vervemint API", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Request id + one access-log line per request + safe 500s."""
    incoming = request.headers.get("X-Request-ID", "")
    # Only accept a simple id from the client: anything else could forge
    # or break log lines.
    request_id = (incoming if _REQUEST_ID_RE.fullmatch(incoming)
                  else uuid.uuid4().hex[:12])
    set_request_id(request_id)  # tags this request's logs and trace
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error [%s] %s %s", request_id,
                         request.method, request.url.path)
        response = JSONResponse(status_code=500, content={
            "detail": f"Internal server error (request id {request_id}). "
                      "Details are in the server log."
        })
    response.headers["X-Request-ID"] = request_id
    logger.info("%s %s -> %d (%.0f ms) [%s]", request.method,
                request.url.path, response.status_code,
                (time.perf_counter() - start) * 1000, request_id)
    return response


def require_token(authorization: str | None = Header(default=None)) -> None:
    token = credentials.api_token()
    if not token:  # fail closed; bootstrap() makes one at startup
        raise HTTPException(503, "The API token is not initialised yet.")
    expected = f"Bearer {token}"
    # compare_digest: timing does not reveal how much of a guess matched.
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(401, "Missing or invalid API token.",
                            headers={"WWW-Authenticate": "Bearer"})


api = APIRouter(dependencies=[Depends(require_token)])


# --- Schemas ---------------------------------------------------------------


class LLMChoice(BaseModel):
    # None: the provider / model chosen on the web UI's Chat page.
    provider: Provider | None = None
    model: str | None = None


class QuestionIn(LLMChoice):
    question: str
    scope: Literal["library", "documents"] = "library"
    document_ids: list[str] = []


class LogIn(QuestionIn):
    # A machine log attached by the user (plain text). It is cleaned and
    # cut to an excerpt before the model reads it (machine_logs.py).
    question: str = ("Analyse the attached log: what is wrong, and how do "
                     "I fix it according to the manuals?")
    log_text: str = Field(min_length=1, max_length=2_000_000)
    log_name: str = "log.txt"


class SourceOut(BaseModel):
    chunk_id: str
    source: str
    page_no: int
    score: float


class PageChunkOut(BaseModel):
    chunk_id: str
    text: str


class AskOut(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    abstained: bool
    blocked_reason: str | None
    latency_ms: dict[str, float]
    sources: list[SourceOut]
    # Every chunk of each retrieved page, so a client can show a cited
    # passage inside its full page. Keyed by "source#pN".
    pages: dict[str, list[PageChunkOut]]


class LogOut(AskOut):
    # The manual searches the workflow made, one per error in the log.
    steps: list[dict[str, Any]]


class WorkOrderOut(BaseModel):
    asset_id: str
    priority: str
    summary: str


class AgentOut(BaseModel):
    answer: str
    steps: list[dict[str, Any]]
    pending_work_order: WorkOrderOut | None
    stopped_reason: str
    sources: list[SourceOut]
    pages: dict[str, list[PageChunkOut]]


class ApprovalIn(BaseModel):
    asset_id: str
    priority: str = Field(pattern="^(" + "|".join(PRIORITIES) + ")$")
    summary: str
    approved_by: str = Field(min_length=1)


class ConnectOut(BaseModel):
    ok: bool
    message: str


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    added_at: str
    cached: bool = False  # True: already stored, nothing was reprocessed


class UploadOut(BaseModel):
    documents: list[DocumentOut]
    errors: list[str]


class SecretOut(BaseModel):
    # The saved value, so Settings can show and edit it; "" when not set.
    # The caller already holds the API token, so nothing is hidden here.
    value: str
    locked: bool   # set in the environment: read-only in Settings


class TelegramOut(BaseModel):
    token: SecretOut
    allowed_users: list[int]
    allowed_users_locked: bool
    status: dict[str, Any]  # bot_status.read()


class SettingsOut(BaseModel):
    provider: Provider
    models: dict[str, str]         # the model used for each provider
    keys: dict[str, SecretOut]     # gemini / claude / openai
    telegram: TelegramOut
    # Not masked: the Settings page shows them, and the caller already
    # holds the API token.
    ui_password: str
    ui_password_locked: bool
    api_token: str
    api_token_locked: bool
    ollama_host: str
    ollama_host_locked: bool


class SettingsIn(BaseModel):
    """A partial update: omitted fields stay as they are. An empty string
    removes a provider key or the bot token."""
    provider: Provider | None = None
    models: dict[Provider, str] | None = None
    keys: dict[Literal["gemini", "claude", "openai"], str] | None = None
    telegram_bot_token: str | None = None
    telegram_allowed_users: list[int] | None = None
    ollama_host: str | None = None  # "" goes back to the default address
    ui_password: str | None = Field(default=None, min_length=8,
                                    max_length=200)
    new_ui_password: bool = False  # generate a new one
    new_api_token: bool = False    # the UI and the bot follow at once

    @field_validator("models")
    @classmethod
    def _model_names(cls, value: dict | None) -> dict | None:
        if value is None:
            return value
        names = {provider: name.strip() for provider, name in value.items()}
        for provider, name in names.items():
            if not _MODEL_NAME_RE.fullmatch(name):
                raise ValueError(f"'{name}' is not a model name "
                                 f"({provider}).")
        return names

    @field_validator("keys")
    @classmethod
    def _key_values(cls, value: dict | None) -> dict | None:
        if value is None:
            return value
        keys = {provider: key.strip() for provider, key in value.items()}
        if any(len(key) > 500 or " " in key for key in keys.values()):
            raise ValueError("That does not look like an API key.")
        return keys

    @field_validator("telegram_bot_token")
    @classmethod
    def _bot_token(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if value and not _TELEGRAM_TOKEN_RE.fullmatch(value):
            raise ValueError("That is not a Telegram bot token. It looks "
                             "like 123456789:AAE... and comes from "
                             "@BotFather.")
        return value

    @field_validator("ollama_host")
    @classmethod
    def _host(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip().rstrip("/")
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("The Ollama address starts with http:// or "
                             "https://, e.g. http://127.0.0.1:11434.")
        return value

    @field_validator("telegram_allowed_users")
    @classmethod
    def _user_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(user_id <= 0 for user_id in value):
            raise ValueError("Telegram user ids are positive numbers.")
        return sorted(set(value)) if value is not None else value


# --- Helpers ---------------------------------------------------------------


def _llm(req: LLMChoice, key_header: str | None) -> LLMConfig:
    """The request's provider and model, else the ones chosen on the Chat
    page; the key sent with the request, else the one saved in Settings
    (or its environment variable)."""
    provider = req.provider or credentials.active_provider()
    return LLMConfig(provider, req.model or credentials.model_for(provider),
                     key_header or credentials.provider_key(provider))


def _secret(field: str, value: str | None) -> SecretOut:
    return SecretOut(value=value or "", locked=credentials.locked(field))


def _settings_out() -> SettingsOut:
    return SettingsOut(
        provider=credentials.active_provider(),
        models={p: credentials.model_for(p) for p in credentials.PROVIDERS},
        keys={p: _secret(p, credentials.provider_key(p))
              for p in credentials.KEY_PROVIDERS},
        telegram=TelegramOut(
            token=_secret("telegram_bot_token",
                          credentials.telegram_bot_token()),
            allowed_users=credentials.telegram_allowed_users(),
            allowed_users_locked=credentials.locked("telegram_allowed_users"),
            status=bot_status.read(),
        ),
        ui_password=credentials.ui_password() or "",
        ui_password_locked=credentials.locked("ui_password"),
        api_token=credentials.api_token() or "",
        api_token_locked=credentials.locked("api_token"),
        ollama_host=credentials.ollama_host(),
        ollama_host_locked=credentials.locked("ollama_host"),
    )


def _retriever(scope: str, document_ids: list[str]) -> Retriever:
    if scope == "library":
        if _state.get("library") is None:
            raise HTTPException(503, "The library index is not built. "
                                     "Run: python -m src.vervemint.index")
        return _state["library"]

    if not document_ids:
        raise HTTPException(400, "Select at least one document to search.")
    key = tuple(sorted(set(document_ids)))
    with _cache_lock:
        cached = _doc_retrievers.get(key)
    if cached:
        return cached
    try:
        chunks, vectors = doc_store.load_documents(list(key),
                                                   _state["embedder"])
    except KeyError as exc:
        raise HTTPException(404, f"Unknown document: {exc.args[0]}") from exc
    retriever = Retriever.from_embeddings(
        chunks, vectors, _state["embedder"], _state["reranker"]
    )
    with _cache_lock:
        if len(_doc_retrievers) >= _MAX_CACHED_RETRIEVERS:
            _doc_retrievers.pop(next(iter(_doc_retrievers)))
        _doc_retrievers[key] = retriever
    return retriever


def _sources_and_pages(
    retriever: Retriever, sources: list[RetrievedChunk]
) -> tuple[list[SourceOut], dict[str, list[PageChunkOut]]]:
    out = [SourceOut(chunk_id=s.chunk_id, source=s.source,
                     page_no=s.page_no, score=s.score) for s in sources]
    pages: dict[str, list[PageChunkOut]] = {}
    for s in sources:
        label = f"{s.source}#p{s.page_no}"
        if label not in pages:
            pages[label] = [
                PageChunkOut(chunk_id=cid, text=text)
                for cid, text in retriever.page_chunks(s.source, s.page_no)
            ]
    return out, pages


# --- Endpoints -------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + what is loaded. Public: no token needed."""
    library = _state.get("library")
    return {
        "status": "ok",
        "library": {"available": library is not None,
                    "chunks": library.size if library else 0},
        "auth_required": True,
    }


@api.get("/assets")
def assets() -> list[dict[str, str]]:
    """The simulated machines the agent can check."""
    return [{"asset_id": a, "status": d["status"], "asset": d["asset"]}
            for a, d in SIMULATED_ASSETS.items()]


@api.get("/providers/ollama/models")
def ollama_models() -> list[str]:
    """What the Ollama server set in Settings has."""
    try:
        return llm_ollama_models()
    except ConnectionError as exc:
        raise HTTPException(
            502, f"Cannot reach Ollama at {credentials.ollama_host()}. Check "
                 "the address in Settings, or start it with `ollama serve`."
        ) from exc


@api.post("/connect", response_model=ConnectOut)
def connect(req: LLMChoice,
            x_llm_api_key: str | None = Header(default=None)) -> ConnectOut:
    """Check a provider key + model for free (no tokens generated)."""
    try:
        return ConnectOut(ok=True,
                          message=check_connection(_llm(req, x_llm_api_key)))
    except LLMError as exc:
        return ConnectOut(ok=False, message=str(exc))


@api.get("/settings", response_model=SettingsOut)
def get_settings() -> SettingsOut:
    """What the web UI's Settings page shows, and the Telegram bot's live
    status."""
    return _settings_out()


@api.get("/telegram/status")
def telegram_status() -> dict[str, Any]:
    """Just the bot's status, for the Settings page to poll."""
    return bot_status.read()


@api.put("/settings", response_model=SettingsOut)
def put_settings(req: SettingsIn) -> SettingsOut:
    """Change credentials or the provider / model choice. Takes effect at
    once in the API, the web UI and the Telegram bot."""
    changes: dict[str, Any] = {}
    locked: list[str] = []

    def change(name: str, value: Any, field: str | None = None) -> None:
        if credentials.locked(field or name):
            locked.append(credentials.ENV_VARS[field or name])
        changes[name] = value

    if req.provider is not None:
        changes["provider"] = req.provider
    if req.models:
        changes["models"] = req.models
    if req.keys:
        for provider in req.keys:
            if credentials.locked(provider):
                locked.append(credentials.ENV_VARS[provider])
        changes["keys"] = req.keys
    if req.telegram_bot_token is not None:
        change("telegram_bot_token", req.telegram_bot_token)
    if req.telegram_allowed_users is not None:
        change("telegram_allowed_users", req.telegram_allowed_users)
    if req.ollama_host is not None:
        change("ollama_host", req.ollama_host)
    if req.new_ui_password or req.ui_password is not None:
        change("ui_password", req.ui_password or credentials.new_password())
    if req.new_api_token:
        change("api_token", credentials.new_secret())
    if locked:
        raise HTTPException(409, "Set in the environment, so it cannot be "
                                 f"changed here: {', '.join(locked)}")
    changed = credentials.update(changes)
    if changed:  # names only, never the values
        logger.info("Settings changed: %s", ", ".join(changed))
    return _settings_out()


@api.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[dict]:
    return doc_store.list_documents()


@api.post("/documents", response_model=UploadOut)
def upload_documents(files: list[UploadFile]) -> UploadOut:
    """Store uploaded files. A file stored before (same bytes) is reused,
    not reprocessed. Bad files are reported without failing the rest."""
    limit = settings.max_upload_mb * 1024 * 1024
    stored, errors = [], []
    for upload in files:
        name = Path(upload.filename or "file").name  # drop client paths
        if Path(name).suffix.lower().lstrip(".") not in SUPPORTED_UPLOAD_TYPES:
            errors.append(f"{name}: unsupported file type (use "
                          f"{', '.join(SUPPORTED_UPLOAD_TYPES)})")
            continue
        data = upload.file.read(limit + 1)
        if len(data) > limit:
            errors.append(f"{name}: larger than {settings.max_upload_mb} MB")
            continue
        try:
            meta, cached = doc_store.add_document(name, data,
                                                  _state["embedder"])
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        stored.append(DocumentOut(**meta, cached=cached))
    return UploadOut(documents=stored, errors=errors)


@api.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    try:
        found = doc_store.delete_document(doc_id)
    except KeyError:
        found = False
    if not found:
        raise HTTPException(404, "Unknown document.")
    with _cache_lock:
        for key in [k for k in _doc_retrievers if doc_id in k]:
            del _doc_retrievers[key]


@api.post("/ask", response_model=AskOut)
def ask_endpoint(req: QuestionIn,
                 x_llm_api_key: str | None = Header(default=None)) -> AskOut:
    """Document Q&A with citations (the RAG pipeline)."""
    retriever = _retriever(req.scope, req.document_ids)
    try:
        result = ask(req.question, retriever, _llm(req, x_llm_api_key))
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    sources, pages = _sources_and_pages(retriever, result.sources)
    return AskOut(
        request_id=result.request_id,
        answer=result.answer,
        citations=result.citations,
        abstained=result.abstained,
        blocked_reason=result.blocked_reason,
        latency_ms=result.latency_ms,
        sources=sources,
        pages=pages,
    )


@api.post("/agent", response_model=AgentOut)
def agent_endpoint(req: QuestionIn,
                   x_llm_api_key: str | None = Header(default=None)
                   ) -> AgentOut:
    """Maintenance agent with tools. A drafted work order comes back as
    `pending_work_order`; nothing is created until POST /work-orders."""
    retriever = _retriever(req.scope, req.document_ids)
    try:
        result = run_agent(req.question, retriever,
                           _llm(req, x_llm_api_key))
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    sources, pages = _sources_and_pages(retriever, result.sources)
    draft = result.pending_work_order
    return AgentOut(
        answer=result.answer,
        steps=result.steps,
        pending_work_order=WorkOrderOut(**vars(draft)) if draft else None,
        stopped_reason=result.stopped_reason,
        sources=sources,
        pages=pages,
    )


@api.post("/analyze-log", response_model=LogOut)
def analyze_log_endpoint(req: LogIn,
                         x_llm_api_key: str | None = Header(default=None)
                         ) -> LogOut:
    """Diagnose an attached machine log from the manuals: one search per
    error in the log, then one cited answer."""
    retriever = _retriever(req.scope, req.document_ids)
    try:
        result, queries = analyze_log(req.question, req.log_text,
                                      req.log_name, retriever,
                                      _llm(req, x_llm_api_key))
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    sources, pages = _sources_and_pages(retriever, result.sources)
    return LogOut(
        request_id=result.request_id,
        answer=result.answer,
        citations=result.citations,
        abstained=result.abstained,
        blocked_reason=result.blocked_reason,
        latency_ms=result.latency_ms,
        sources=sources,
        pages=pages,
        steps=[{"tool": "search_manuals", "arguments": {"query": q}}
               for q in queries],
    )


@api.post("/work-orders", status_code=201)
def approve_endpoint(req: ApprovalIn) -> dict[str, Any]:
    """Create a work order. Call this only after a human approved it."""
    draft = WorkOrderDraft(req.asset_id, req.priority, req.summary)
    return approve_work_order(draft, req.approved_by)


@api.get("/stats")
def stats() -> dict[str, Any]:
    """Headline numbers from logs/traces.jsonl."""
    return summarize(load_traces())


app.include_router(api)
