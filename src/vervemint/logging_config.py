"""One place that configures logging for the whole project.

Every module gets its logger with `logging.getLogger(__name__)` and never
configures handlers itself. Entry points (the UI, the index build script)
call `setup_logging()` once at startup.

Output goes to three places:
- the console, for development
- logs/vervemint.log, rotated at 5 MB with 3 backups, so the log folder
  can never grow without bound
- logs/llm.log: the full text of every LLM prompt and reply (llm.py),
  kept out of the console and vervemint.log because it is long
"""

import logging
from logging.handlers import RotatingFileHandler

from src.vervemint.config import settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_HANDLER_NAME = "vervemint_file"
LLM_IO_LOGGER = "src.vervemint.llm.io"

# Third-party libraries that are chatty at INFO level.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "bm25s",
    "faiss",
    "anthropic",
    "openai",
    "google_genai",
)
# These warn on every start about things that are fine here: no Hugging
# Face token (the two models are public and already downloaded) and
# unused keys in a checkpoint. Real failures still come through at ERROR.
_STARTUP_CHATTER = ("huggingface_hub", "transformers")
# ...except their retry messages ("Retrying request ... in 2 seconds"),
# which explain why an answer is slow.
_RETRY_LOGGERS = (
    "google_genai._api_client",
    "anthropic._base_client",
    "openai._base_client",
)


def setup_logging(filename: str = "vervemint.log") -> None:
    """Attach console + rotating-file handlers to the root logger.

    Each process gets its own file (the API writes vervemint.log, the UI
    ui.log): two processes rotating one file fails on Windows.

    Safe to call many times: Streamlit re-runs the UI script on every
    click, and without this guard each re-run would add another pair of
    handlers and every log line would be written N times.
    """
    root = logging.getLogger()
    if any(h.get_name() == _FILE_HANDLER_NAME for h in root.handlers):
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        settings.log_dir / filename,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.set_name(_FILE_HANDLER_NAME)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(settings.log_level)

    llm_handler = RotatingFileHandler(
        settings.log_dir / "llm.log",
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
        delay=True,  # opened on the first LLM call; the UI never makes one
    )
    llm_handler.setFormatter(formatter)
    io_logger = logging.getLogger(LLM_IO_LOGGER)
    io_logger.addHandler(llm_handler)
    io_logger.setLevel(logging.INFO)
    io_logger.propagate = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _STARTUP_CHATTER:
        logging.getLogger(name).setLevel(logging.ERROR)
    for name in _RETRY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)


def quiet_model_loading() -> None:
    """Turn off the progress bars and the key-by-key "LOAD REPORT" that
    transformers prints while a model loads. Called where models are
    loaded, so the UI and the Telegram bot never import transformers
    just to stay quiet."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
