"""Settings loading, validation and secret handling."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from eris.config import DEFAULT_MODEL, Settings, load_settings
from eris.errors import ConfigError


def test_defaults_match_documented_values() -> None:
    settings = load_settings()
    assert settings.model == DEFAULT_MODEL
    assert settings.search_backend == "duckduckgo"
    assert settings.max_results == 8
    assert settings.top_k == 6
    assert settings.cache_ttl_s == 3600
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200
    assert settings.cache_dir == Path("./data/cache")


def test_api_key_is_optional_at_import() -> None:
    """The package must be usable, and testable, with no credential present."""
    settings = load_settings()
    assert settings.anthropic_api_key is None
    assert settings.has_api_key is False


def test_env_vars_are_read_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_TOP_K", "11")
    monkeypatch.setenv("ERIS_MODEL", "claude-test-model")
    monkeypatch.setenv("ERIS_SEARCH_BACKEND", "fake")
    settings = load_settings()
    assert settings.top_k == 11
    assert settings.model == "claude-test-model"
    assert settings.search_backend == "fake"


def test_env_var_without_prefix_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "99")
    assert load_settings().top_k == 6


def test_overrides_beat_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_TOP_K", "3")
    assert load_settings(top_k=9).top_k == 9


def test_none_overrides_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags default to None and must not clobber configured values."""
    monkeypatch.setenv("ERIS_TOP_K", "7")
    assert load_settings(top_k=None, model=None).top_k == 7


def test_secret_is_not_exposed_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_ANTHROPIC_API_KEY", "sk-ant-super-secret")
    settings = load_settings()
    assert "sk-ant-super-secret" not in repr(settings)
    assert "sk-ant-super-secret" not in str(settings)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-super-secret"


def test_require_api_key_returns_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_ANTHROPIC_API_KEY", "sk-ant-abc")
    assert load_settings().require_api_key() == "sk-ant-abc"


def test_require_api_key_raises_when_absent() -> None:
    with pytest.raises(ConfigError, match="ERIS_ANTHROPIC_API_KEY"):
        load_settings().require_api_key()


def test_blank_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_ANTHROPIC_API_KEY", "   ")
    settings = load_settings()
    assert settings.has_api_key is False
    with pytest.raises(ConfigError):
        settings.require_api_key()


def test_unknown_search_backend_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_settings(search_backend="bing")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", -1),
        ("max_results", 0),
        ("fetch_timeout_s", 0.0),
        ("fetch_timeout_s", -3.0),
        ("cache_ttl_s", -1),
        ("bm25_weight", 1.5),
        ("bm25_weight", -0.1),
        ("mmr_lambda", 2.0),
        ("temperature", -0.5),
        ("max_tokens", 0),
        ("fetch_concurrency", 0),
        ("max_chunks_per_url", 0),
        ("chunk_size", 10),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ConfigError):
        load_settings(**{field: value})


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_settings(chunk_size=500, chunk_overlap=500)


def test_equal_boundary_overlap_is_allowed_just_below() -> None:
    assert load_settings(chunk_size=500, chunk_overlap=499).chunk_overlap == 499


def test_blank_model_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_settings(model="   ")


def test_text_fields_are_stripped() -> None:
    assert load_settings(model="  claude-x  ").model == "claude-x"


def test_log_level_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_LOG_LEVEL", "debug")
    assert load_settings().log_level == "DEBUG"


def test_cache_path_is_derived_from_cache_dir(tmp_path: Path) -> None:
    settings = load_settings(cache_dir=tmp_path / "c")
    assert settings.cache_path == tmp_path / "c" / "eris.sqlite3"


def test_cache_dir_expands_tilde() -> None:
    settings = load_settings(cache_dir="~/eris-cache")
    assert "~" not in str(settings.cache_dir)
    assert settings.cache_dir.is_absolute()


def test_tfidf_weight_complements_bm25_weight() -> None:
    settings = load_settings(bm25_weight=0.75)
    assert settings.tfidf_weight == pytest.approx(0.25)


def test_configure_logging_only_touches_eris_logger() -> None:
    root_level = logging.getLogger().level
    load_settings(log_level="ERROR").configure_logging()
    assert logging.getLogger("eris").level == logging.ERROR
    assert logging.getLogger().level == root_level


def test_extra_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERIS_TOTALLY_UNKNOWN_SETTING", "x")
    assert load_settings().top_k == 6


def test_settings_can_be_constructed_directly() -> None:
    assert Settings(top_k=2).top_k == 2
