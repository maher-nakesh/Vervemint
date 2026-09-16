"""Fixtures shared by every test module."""

import os

import pytest

# Names config.py reads directly (not VERVEMINT_-prefixed): provider keys
# and the Ollama host.
_PROVIDER_ENV_VARS = (
    "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_HOST",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Tests must give deterministic results regardless of a developer's
    local .env: src.vervemint.config loads it into os.environ at import
    time (for `python -m ui.telegram_bot` and friends), and real
    environment variables outrank config.yaml. Clear every VERVEMINT_
    variable and the provider keys before each test; monkeypatch
    restores them afterwards, and a test can still set exactly what it
    needs with its own monkeypatch.setenv."""
    for key in list(os.environ):
        if key.startswith("VERVEMINT_") or key in _PROVIDER_ENV_VARS:
            monkeypatch.delenv(key, raising=False)
