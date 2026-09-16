"""config.yaml loading: priority order and validation."""

import pytest
from pydantic import ValidationError

from src.vervemint.config import PROJECT_ROOT, Settings


def _settings_from(tmp_path, monkeypatch, yaml_text: str) -> Settings:
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("VERVEMINT_CONFIG_FILE", str(path))
    return Settings()


def test_yaml_value_overrides_default(tmp_path, monkeypatch):
    s = _settings_from(tmp_path, monkeypatch, "rerank_top_k: 8\n")
    assert s.rerank_top_k == 8
    assert s.bm25_top_k == 20  # key not in the file -> default


def test_env_var_overrides_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("VERVEMINT_RERANK_TOP_K", "3")
    s = _settings_from(tmp_path, monkeypatch, "rerank_top_k: 8\n")
    assert s.rerank_top_k == 3


def test_missing_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("VERVEMINT_CONFIG_FILE", str(tmp_path / "nope.yaml"))
    assert Settings().rerank_top_k == 5


def test_relative_path_resolves_to_project_root(tmp_path, monkeypatch):
    s = _settings_from(tmp_path, monkeypatch, "index_dir: data/other\n")
    assert s.index_dir == PROJECT_ROOT / "data" / "other"


@pytest.mark.parametrize("bad_yaml", [
    "rerank_topk: 8\n",                                  # misspelled key
    "rerank_top_k: five\n",                              # wrong type
    "min_rerank_score: 1.5\n",                           # out of range
    "max_chunk_chars: 100\nchunk_overlap_chars: 200\n",  # inconsistent
])
def test_invalid_config_is_rejected(tmp_path, monkeypatch, bad_yaml):
    with pytest.raises(ValidationError):
        _settings_from(tmp_path, monkeypatch, bad_yaml)


def test_shipped_config_yaml_is_valid(monkeypatch):
    monkeypatch.delenv("VERVEMINT_CONFIG_FILE", raising=False)
    assert Settings().max_agent_steps > 0
