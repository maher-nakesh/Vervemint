"""Central configuration for Vervemint.

Where each value comes from, highest priority first:
  1. Real environment variables (Docker, CI, a shell `export`/`$env:`)
  2. .env in the project root, loaded into the process environment
     below. Optional: Docker Compose reads its own .env instead.
  3. config.yaml in the project root, or the file named by the
     VERVEMINT_CONFIG_FILE environment variable
  4. The defaults defined in this file

Credentials (provider API keys, the Telegram bot, passwords and the API
token) are not configured here: the user sets them in the web UI's
Settings page, and credentials.py stores them.

Every module imports `settings` from here, so config.yaml controls the
whole app. Values are read once at startup: restart the UI or API after
editing config.yaml. Invalid values (wrong type, out of range, or a
misspelled key) stop the app at startup with a clear error instead of
failing later in the middle of a request.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.yaml"
_DATASET_DIR = PROJECT_ROOT / "data" / "industrial-instruction-dataset"

# override=False: a real environment variable (Docker, CI, the shell)
# always wins over .env. No-op if the file does not exist.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def config_file() -> Path:
    """The YAML file to load. A missing file is fine: defaults apply."""
    return Path(os.environ.get("VERVEMINT_CONFIG_FILE", DEFAULT_CONFIG_FILE))


class Settings(BaseSettings):
    # extra="forbid": a misspelled key in config.yaml is an error, not a
    # silently ignored setting.
    model_config = SettingsConfigDict(env_prefix="VERVEMINT_", extra="forbid")

    # --- Paths (relative paths are resolved against the project root) --
    data_dir: Path = PROJECT_ROOT / "data"
    corpus_dir: Path = _DATASET_DIR / "panasonic_v0_0"
    qa_dir: Path = _DATASET_DIR / "panasonic_qa_v1"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    log_dir: Path = PROJECT_ROOT / "logs"

    # --- Logging ---------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Write every prompt sent to an LLM and every reply to logs/llm.log.
    # Prompts contain document text and questions: keep the file private.
    log_llm_messages: bool = True

    # --- Chunking --------------------------------------------------------
    max_chunk_chars: int = Field(1200, gt=0)
    chunk_overlap_chars: int = Field(150, ge=0)

    # --- Retrieval -------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    bm25_top_k: int = Field(20, gt=0)
    dense_top_k: int = Field(20, gt=0)
    rerank_top_k: int = Field(5, gt=0)
    rrf_k: int = Field(60, gt=0)  # standard Reciprocal Rank Fusion constant

    # --- LLM providers ----------------------------------------------------
    # Default model per provider, until the user picks one on the home
    # page (the choice is then stored by credentials.py).
    ollama_model: str = "qwen2.5-7b-gpu:latest"
    # 0 = same answer every time; small local models follow the
    # citation and "not in sources" rules more reliably at 0.
    ollama_temperature: float = Field(0.0, ge=0, le=2)
    # Context window Ollama reserves GPU memory for. Our prompts need at
    # most ~5k tokens; a 32k window can fill an 8 GB GPU, and Windows then
    # silently spills to system RAM, which is many times slower.
    ollama_num_ctx: int = Field(8192, ge=2048)
    claude_model: str = "claude-opus-5"
    openai_model: str = "gpt-5-mini"
    gemini_model: str = "gemini-3.8-flash"
    # Grounded Q&A needs little reasoning; "low" answers much faster.
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    # Per-attempt limit for a Gemini call. Without one, an overloaded
    # model can hold a request for minutes; a timed-out attempt is retried.
    gemini_timeout_s: float = Field(60, gt=0)

    # --- Generation / guardrails / agent ---------------------------------
    min_rerank_score: float = Field(0.15, ge=0, le=1)  # below: abstain
    max_query_chars: int = Field(1000, gt=0)
    max_agent_steps: int = Field(6, gt=0)
    max_log_chars: int = Field(12000, gt=0)  # log excerpt sent to the agent

    # --- API server and front ends ------------------------------------------
    api_url: str = "http://127.0.0.1:8000"  # where the UI and bot find the API
    max_upload_mb: int = Field(50, gt=0)
    request_timeout_s: float = Field(300, gt=0)  # UI wait for LLM answers
    # Ask for the web UI password (shown in Settings) before showing the
    # app. docker-compose.yml turns it on; off for local development.
    ui_require_password: bool = False
    # Run the Telegram bot inside the API process, so starting the
    # backend starts the bot too. Turn it off only to run the bot
    # separately (`python -m ui.telegram_bot`): Telegram allows one
    # poller per token, so never run both.
    telegram_bot: bool = True

    @field_validator(
        "data_dir", "corpus_dir", "qa_dir", "index_dir", "documents_dir",
        "log_dir",
    )
    @classmethod
    def _resolve_relative(cls, path: Path) -> Path:
        # "data/index" in config.yaml must mean the project's data folder,
        # not a folder relative to wherever the app was started from.
        return path if path.is_absolute() else PROJECT_ROOT / path

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than max_chunk_chars"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order = priority: environment (.env is loaded into it above)
        beats YAML beats defaults."""
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=config_file()),
        )


settings = Settings()
