"""The credentials store: generated secrets, environment overrides and
partial updates -- files only, no server."""

import json
import os
import stat

import pytest

from src.vervemint import credentials
from src.vervemint.config import settings


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path / "credentials.json"


def test_first_start_generates_the_token_and_password(store):
    first = credentials.bootstrap()
    assert len(first["api_token"]) >= 40 and len(first["ui_password"]) >= 16
    assert credentials.bootstrap() == first  # the next start keeps them
    # Only what is really set: no empty keys, no default provider or models.
    assert set(json.loads(store.read_text())) == {"api_token", "ui_password"}
    if os.name == "posix":  # Windows has no owner-only file mode
        assert stat.S_IMODE(store.stat().st_mode) == 0o600


def test_only_what_is_set_is_stored(store):
    credentials.bootstrap()
    credentials.update({"keys": {"gemini": "g-key", "claude": ""},
                        "telegram_bot_token": "", "provider": "gemini"})
    saved = json.loads(store.read_text())
    assert saved["keys"] == {"gemini": "g-key"}      # no blank claude entry
    assert "telegram_bot_token" not in saved         # not set: not written
    credentials.update({"keys": {"gemini": ""}, "provider": ""})
    saved = json.loads(store.read_text())
    assert "keys" not in saved and "provider" not in saved


def test_nothing_is_stored_before_the_first_start():
    assert credentials.api_token() is None
    assert credentials.active_provider() == "ollama"
    assert credentials.model_for("gemini") == settings.gemini_model


def test_update_merges_and_removes_keys():
    credentials.bootstrap()
    changed = credentials.update({"keys": {"gemini": "g-key", "openai": "o"},
                                  "provider": "gemini"})
    assert changed == ["keys.gemini", "keys.openai", "provider"]
    credentials.update({"keys": {"openai": ""}})
    assert credentials.provider_key("gemini") == "g-key"
    assert credentials.provider_key("openai") is None
    assert credentials.provider_key("ollama") is None
    assert credentials.update({"provider": "gemini"}) == []  # no change


def test_values_equal_to_the_defaults_are_not_stored(store):
    """The file shows what the user chose, not what the app would do
    anyway -- including after an older version wrote the defaults."""
    credentials.bootstrap()
    credentials.update({"ollama_host": credentials.DEFAULT_OLLAMA_HOST,
                        "provider": "ollama",
                        "models": {"ollama": settings.ollama_model}})
    assert set(json.loads(store.read_text())) == {"api_token", "ui_password"}
    assert credentials.ollama_host() == credentials.DEFAULT_OLLAMA_HOST
    assert credentials.active_provider() == "ollama"
    assert credentials.model_for("ollama") == settings.ollama_model


def test_ollama_host_falls_back_to_the_default(monkeypatch):
    credentials.bootstrap()
    assert credentials.ollama_host() == credentials.DEFAULT_OLLAMA_HOST
    credentials.update({"ollama_host": "http://ollama:11434"})
    assert credentials.ollama_host() == "http://ollama:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
    assert credentials.ollama_host() == "http://from-env:11434"
    assert credentials.locked("ollama_host")


def test_environment_wins_and_locks(monkeypatch):
    credentials.bootstrap()
    credentials.update({"keys": {"claude": "stored"}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert credentials.provider_key("claude") == "from-env"
    assert credentials.locked("claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")  # blank = not set
    assert credentials.provider_key("claude") == "stored"
    assert not credentials.locked("claude")


def test_bad_allowed_users_in_environment_let_nobody_in(monkeypatch):
    monkeypatch.setenv("VERVEMINT_TELEGRAM_ALLOWED_USERS", "123, 456")
    assert credentials.telegram_allowed_users() == [123, 456]
    monkeypatch.setenv("VERVEMINT_TELEGRAM_ALLOWED_USERS", "123, admin")
    assert credentials.telegram_allowed_users() == []


def test_models_fall_back_to_the_configured_defaults():
    credentials.bootstrap()
    assert credentials.model_for("ollama") == settings.ollama_model
    credentials.update({"models": {"ollama": "qwen2.5:7b"}})
    assert credentials.model_for("ollama") == "qwen2.5:7b"
    assert credentials.model_for("claude") == settings.claude_model
