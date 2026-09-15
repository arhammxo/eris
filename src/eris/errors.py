"""Exception hierarchy for Eris.

Every failure raised by Eris derives from :class:`ErisError`, so callers can
catch one type at the boundary. Third-party exceptions (``httpx``, ``ddgs``,
``anthropic``) are wrapped rather than propagated, keeping optional
dependencies out of caller ``except`` clauses.
"""

from __future__ import annotations


class ErisError(Exception):
    """Base class for all Eris failures."""


class ConfigError(ErisError):
    """Configuration is missing or invalid."""


class SearchError(ErisError):
    """A search backend failed or is unavailable."""


class FetchError(ErisError):
    """A page could not be fetched or decoded."""


class CacheError(ErisError):
    """The on-disk cache could not be read or written."""


class RetrievalError(ErisError):
    """Chunking or ranking failed."""


class LLMError(ErisError):
    """The language model client failed."""


class DependencyMissingError(ErisError):
    """An optional dependency is required for the requested feature.

    Carries the extra name so the message can tell the user exactly what to
    install, e.g. ``pip install 'eris[search]'``.
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{package!r} is not installed. Install it with: pip install 'eris[{extra}]'"
        )
