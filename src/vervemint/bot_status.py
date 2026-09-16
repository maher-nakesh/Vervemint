"""Live status of the Telegram bot, shown in the web UI's Settings page.

The bot process rewrites data/telegram_status.json every few seconds (a
heartbeat) and the API reads it. A status older than STALE_AFTER_S means
the bot process is not running.

States: waiting_for_token, starting, running, invalid_token, conflict,
error, not_running.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.vervemint.config import settings
from src.vervemint.credentials import write_atomic

logger = logging.getLogger(__name__)

STALE_AFTER_S = 20


def path() -> Path:
    return settings.data_dir / "telegram_status.json"


def write(state: str, detail: str = "") -> None:
    record = {"state": state, "detail": detail, "updated_at": time.time()}
    try:
        write_atomic(path(), json.dumps(record), mode=0o644)
    except OSError:
        logger.exception("Could not write the bot status")


def read() -> dict[str, Any]:
    try:
        record = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if record is None or time.time() - record["updated_at"] > STALE_AFTER_S:
        return {"state": "not_running",
                "detail": "The Telegram bot process is not running.",
                "updated_at": record["updated_at"] if record else None}
    return record
