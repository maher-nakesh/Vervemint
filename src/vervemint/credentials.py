"""Credentials and the model choice, set from the web UI and shared by
the API, the web UI and the Telegram bot.

Stored in data/credentials.json (owner-only file mode). In Docker that
is the shared data volume, so a change made in the UI reaches every
service at once, without a restart:

- api_token        secret the web UI and the bot send to the API;
                   generated on the API's first start
- ui_password      the web UI's login password (when it is required);
                   generated on the API's first start, shown in Settings
- keys             Gemini, Claude and OpenAI API keys (Settings page)
- telegram_bot_token, telegram_allowed_users          (Settings page)
- ollama_host      the Ollama server address (Settings page)
- provider         the provider chosen on the Chat page; the Telegram
                   bot follows it
- models           the model chosen for each provider

The file holds only what is actually set: nothing is written with a
default or an empty value, and clearing a field in Settings removes it,
so the file always reads as "what the user chose".

Only the API writes the file (the UI changes it through PUT /settings),
atomically and under a lock; readers re-read it whenever it changes.

An environment variable, when set, wins over the stored value and locks
that field in the Settings page: the way to inject a secret from a
secrets manager.
"""

import copy
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from src.vervemint.config import settings

logger = logging.getLogger(__name__)

PROVIDERS = ("ollama", "gemini", "claude", "openai")
KEY_PROVIDERS = ("gemini", "claude", "openai")  # the ones that need a key
# What the ollama package uses when nothing is configured.
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Environment variables that override (and lock) a stored value.
ENV_VARS = {
    "api_token": "VERVEMINT_API_TOKEN",
    "ui_password": "VERVEMINT_UI_PASSWORD",
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "telegram_bot_token": "VERVEMINT_TELEGRAM_BOT_TOKEN",
    "telegram_allowed_users": "VERVEMINT_TELEGRAM_ALLOWED_USERS",
    "ollama_host": "OLLAMA_HOST",
}

_lock = threading.Lock()
_cache: tuple[tuple, dict] | None = None


def path() -> Path:
    return settings.data_dir / "credentials.json"


def write_atomic(file: Path, text: str, mode: int = 0o600) -> None:
    """Readers see the old file or the new one, never half of it.

    The temp file name includes the pid and a random suffix, so two
    writers -- two threads, or two processes, e.g. an old and a new API
    instance during a restart -- never share one temp file. Without that,
    one writer's os.replace() can remove the temp file a second writer
    is about to rename, which raised FileNotFoundError here before.
    Whichever writer finishes last simply wins, as before.
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.{secrets.token_hex(4)}"
                         ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        for attempt in range(20):
            try:
                os.replace(tmp, file)
                return
            except PermissionError:  # Windows: a reader has the file open
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def load() -> dict[str, Any]:
    """The stored values; {} before the API's first start."""
    global _cache
    file = path()
    try:
        stat = file.stat()
    except FileNotFoundError:
        return {}
    key = (str(file), stat.st_mtime_ns, stat.st_size)
    if _cache is None or _cache[0] != key:
        _cache = (key, json.loads(file.read_text(encoding="utf-8")))
    return _cache[1]


def _save(data: dict[str, Any]) -> None:
    global _cache
    write_atomic(path(), json.dumps(data, indent=2))
    _cache = None


def new_secret() -> str:
    """For the API token."""
    return secrets.token_urlsafe(32)


def new_password() -> str:
    """16 characters, easy to copy: letters, digits, - and _."""
    return secrets.token_urlsafe(12)


def default_models() -> dict[str, str]:
    return {"ollama": settings.ollama_model, "gemini": settings.gemini_model,
            "claude": settings.claude_model, "openai": settings.openai_model}


def _prune(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only what the user really chose: no empty entries, and no
    value the app would use anyway. Also tidies files written by an
    older version."""
    defaults = {"ollama_host": DEFAULT_OLLAMA_HOST, "provider": "ollama"}
    data = {name: value for name, value in data.items()
            if value and defaults.get(name) != value}
    for section, unwanted in (("keys", {}), ("models", default_models())):
        values = {name: value for name, value in data.get(section, {}).items()
                  if value and unwanted.get(name) != value}
        if values:
            data[section] = values
        else:
            data.pop(section, None)
    return data


def bootstrap() -> dict[str, Any]:
    """Create the store on the API's first start. Only the two secrets
    the app cannot ask the user for are generated; everything else
    appears in the file when it is set in Settings."""
    with _lock:
        data = _prune(copy.deepcopy(load()))
        before = copy.deepcopy(load())
        if not data.get("api_token"):
            data["api_token"] = new_secret()
        if not data.get("ui_password"):
            data["ui_password"] = new_password()
            if settings.ui_require_password and not locked("ui_password"):
                # Only when the UI actually asks for it: otherwise the
                # password is simply there in Settings, unused.
                logger.warning("Generated the web UI password: %s (shown "
                               "in Settings, where you can change it)",
                               data["ui_password"])
        if data != before:
            _save(data)
        return data


def update(changes: dict[str, Any]) -> list[str]:
    """Save validated changes from PUT /settings. `keys` and `models` are
    merged per provider. An empty value (a removed key, no allowed users)
    drops the entry instead of storing a blank, so the file holds only
    what is set. Returns the names of the changed fields, for the log
    (never their values)."""
    with _lock:
        original = load()
        data = copy.deepcopy(original)
        changed = []
        for name, value in changes.items():
            if name in ("keys", "models"):
                section = dict(data.get(name, {}))
                for provider, item in value.items():
                    if section.get(provider, "") != item:
                        changed.append(f"{name}.{provider}")
                    section[provider] = item
                data[name] = section
            else:
                if data.get(name, "") != value:
                    changed.append(name)
                data[name] = value
        data = _prune(data)
        if data != original:
            _save(data)
        return changed


# --- Reading, environment first ------------------------------------------


def _env(field: str) -> str | None:
    value = os.environ.get(ENV_VARS[field], "").strip()
    return value or None


def locked(field: str) -> bool:
    """True if an environment variable sets this field."""
    return _env(field) is not None


def api_token() -> str | None:
    return _env("api_token") or load().get("api_token") or None


def ui_password() -> str | None:
    return _env("ui_password") or load().get("ui_password") or None


def provider_key(provider: str) -> str | None:
    if provider not in KEY_PROVIDERS:
        return None
    return _env(provider) or load().get("keys", {}).get(provider) or None


def telegram_bot_token() -> str | None:
    return (_env("telegram_bot_token") or load().get("telegram_bot_token")
            or None)


def parse_user_ids(text: str) -> list[int]:
    """'123, 456' -> [123, 456]. ValueError for anything but numbers."""
    return [int(part) for part in text.replace(",", " ").split()]


def telegram_allowed_users() -> list[int]:
    raw = _env("telegram_allowed_users")
    if raw is None:
        return [int(u) for u in load().get("telegram_allowed_users", [])]
    try:
        return parse_user_ids(raw)
    except ValueError:
        # Fail closed: a typo in the environment must not let anyone in.
        logger.error("%s is not a list of numbers: nobody is allowed",
                     ENV_VARS["telegram_allowed_users"])
        return []


def ollama_host() -> str:
    """The Ollama server the API talks to (Settings page)."""
    return (_env("ollama_host") or load().get("ollama_host")
            or DEFAULT_OLLAMA_HOST)


def active_provider() -> str:
    provider = load().get("provider")
    return provider if provider in PROVIDERS else "ollama"


def model_for(provider: str) -> str:
    return (load().get("models", {}).get(provider)
            or default_models()[provider])
