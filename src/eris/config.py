"""Typed configuration loaded from the environment.

All settings carry the ``ERIS_`` prefix and are read from the process
environment or a local ``.env`` file. No secret is ever hard-coded: the API key
is a :class:`~pydantic.SecretStr` and is *optional at import time*, so the
package, its tests and the offline eval harness all work with no key present.
:meth:`Settings.require_api_key` is the single place that demands one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from eris.errors import ConfigError

SearchBackendName = Literal["duckduckgo", "fake"]
LogLevelName = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_USER_AGENT = "eris/0.1 (+https://github.com/arhammxo/eris)"


class Settings(BaseSettings):
    """Runtime configuration for every Eris stage."""

    model_config = SettingsConfigDict(
        env_prefix="ERIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key. Optional at import; required only for real LLM calls.",
    )
    model: str = Field(default=DEFAULT_MODEL, description="Claude model id.")
    max_tokens: int = Field(default=1024, ge=1, le=64_000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Search ------------------------------------------------------------
    search_backend: SearchBackendName = Field(default="duckduckgo")
    max_results: int = Field(default=8, ge=1, le=50)

    # --- Fetch -------------------------------------------------------------
    fetch_timeout_s: float = Field(default=10.0, gt=0.0, le=300.0)
    max_page_bytes: int = Field(default=2_000_000, ge=1_024)
    fetch_concurrency: int = Field(default=8, ge=1, le=64)
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)

    # --- Cache -------------------------------------------------------------
    cache_dir: Path = Field(default=Path("./data/cache"))
    cache_ttl_s: int = Field(default=3600, ge=0)

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=6, ge=1, le=100)
    chunk_size: int = Field(default=1200, ge=100, le=20_000)
    chunk_overlap: int = Field(default=200, ge=0)
    bm25_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    max_chunks_per_url: int = Field(default=2, ge=1, le=100)

    # --- Logging -----------------------------------------------------------
    log_level: LogLevelName = Field(default="WARNING")

    @field_validator("cache_dir", mode="after")
    @classmethod
    def _expand_cache_dir(cls, value: Path) -> Path:
        """Expand ``~`` so ``ERIS_CACHE_DIR=~/x`` behaves as users expect."""
        return value.expanduser()

    @field_validator("model", "user_agent", mode="after")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        """Overlap must be smaller than the window, else chunking cannot advance."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    # --- Derived helpers ---------------------------------------------------
    @property
    def has_api_key(self) -> bool:
        """True when a non-blank API key is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value().strip())

    @property
    def cache_path(self) -> Path:
        """Full path to the sqlite cache file."""
        return self.cache_dir / "eris.sqlite3"

    @property
    def tfidf_weight(self) -> float:
        """Weight of the TF-IDF cosine component in the hybrid score."""
        return 1.0 - self.bm25_weight

    def require_api_key(self) -> str:
        """Return the API key or raise :class:`ConfigError`.

        The only place in the codebase that insists on a credential.
        """
        if not self.has_api_key:
            raise ConfigError(
                "ERIS_ANTHROPIC_API_KEY is not set. Export it or add it to .env "
                "(see .env.example). Use --search-backend fake with a fake LLM for offline runs."
            )
        assert self.anthropic_api_key is not None  # narrowed by has_api_key
        return self.anthropic_api_key.get_secret_value()

    def configure_logging(self) -> None:
        """Apply ``log_level`` to the ``eris`` logger tree.

        Scoped to our own logger so importing Eris never reconfigures logging
        for a host application.
        """
        logging.getLogger("eris").setLevel(getattr(logging, self.log_level))


def load_settings(**overrides: object) -> Settings:
    """Build :class:`Settings` from the environment, applying ``overrides``.

    Keyword overrides win over the environment, which lets the CLI map flags
    onto settings without mutating ``os.environ``.

    Raises:
        ConfigError: if any value fails validation.
    """
    filtered = {k: v for k, v in overrides.items() if v is not None}
    try:
        return Settings(**filtered)  # type: ignore[arg-type]
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"Invalid Eris configuration: {exc}") from exc
