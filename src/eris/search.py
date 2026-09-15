"""Search backends.

Eris talks to search through the :class:`SearchBackend` protocol, so the real
keyless DuckDuckGo backend and the deterministic :class:`FakeSearch` used by
tests and the eval harness are interchangeable. That substitution is what keeps
the whole test suite offline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from eris.errors import DependencyMissingError, SearchError
from eris.models import SearchResult

log = logging.getLogger(__name__)

# Result dict keys vary across ddgs releases; accept any of these.
_URL_KEYS = ("href", "url", "link")
_SNIPPET_KEYS = ("body", "snippet", "description", "excerpt")
_TITLE_KEYS = ("title", "name")


@runtime_checkable
class SearchBackend(Protocol):
    """Anything that turns a query into ranked web results."""

    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return up to ``max_results`` hits for ``query``.

        Implementations must not raise for an empty result set; they return an
        empty list. Transport and quota failures raise :class:`SearchError`.
        """
        ...


def _first_str(mapping: Mapping[str, object], keys: Iterable[str]) -> str:
    """First non-empty string value among ``keys``, else ``""``."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_results(raw: Sequence[Mapping[str, object]]) -> list[SearchResult]:
    """Convert backend dicts to :class:`SearchResult`, dropping unusable rows.

    Also de-duplicates by URL while preserving rank order, since search
    backends occasionally return the same page twice.
    """
    results: list[SearchResult] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        url = _first_str(row, _URL_KEYS)
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(
                title=_first_str(row, _TITLE_KEYS) or url,
                url=url,
                snippet=_first_str(row, _SNIPPET_KEYS),
            )
        )
    return results


class DuckDuckGoSearch:
    """Keyless web search via the ``ddgs`` package.

    Chosen because it needs no API key or account, which keeps Eris runnable
    straight after ``pip install``. The import is deferred to call time so the
    dependency stays optional.
    """

    name = "duckduckgo"

    def __init__(self, *, region: str = "wt-wt", safesearch: str = "moderate") -> None:
        self.region = region
        self.safesearch = safesearch

    @staticmethod
    def _client() -> object:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise DependencyMissingError("ddgs", "search") from exc
        return DDGS()

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        if max_results <= 0:
            return []

        client = self._client()
        try:
            raw = client.text(  # type: ignore[attr-defined]
                query,
                max_results=max_results,
                region=self.region,
                safesearch=self.safesearch,
            )
        except DependencyMissingError:
            raise
        except Exception as exc:
            # ddgs raises rate-limit/timeout types we deliberately do not import.
            raise SearchError(f"DuckDuckGo search failed for {query!r}: {exc}") from exc

        results = normalize_results(raw or [])
        log.debug("duckduckgo returned %d results for %r", len(results), query)
        return results[:max_results]


class FakeSearch:
    """Deterministic in-memory backend for tests and offline eval.

    ``fixtures`` maps a query to its results. Lookup is case-insensitive and
    whitespace-normalised, then falls back to substring matching so eval files
    can use natural questions without exact-string brittleness.
    """

    name = "fake"

    def __init__(
        self,
        fixtures: Mapping[str, Sequence[SearchResult]] | None = None,
        *,
        default: Sequence[SearchResult] | None = None,
    ) -> None:
        self._fixtures = {self._key(k): list(v) for k, v in (fixtures or {}).items()}
        self._default = list(default or [])
        self.calls: list[str] = []

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.lower().split())

    def add(self, query: str, results: Sequence[SearchResult]) -> None:
        """Register or replace the results for ``query``."""
        self._fixtures[self._key(query)] = list(results)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls.append(query)
        key = self._key(query)
        if key in self._fixtures:
            return self._fixtures[key][:max_results]
        for fixture_key, results in self._fixtures.items():
            if fixture_key and (fixture_key in key or key in fixture_key):
                return results[:max_results]
        return self._default[:max_results]


def build_search_backend(name: str, **kwargs: object) -> SearchBackend:
    """Instantiate the backend called ``name``.

    Raises:
        SearchError: if ``name`` is not a known backend.
    """
    key = name.strip().lower()
    if key == "duckduckgo":
        return DuckDuckGoSearch(**kwargs)  # type: ignore[arg-type]
    if key == "fake":
        return FakeSearch()
    raise SearchError(f"Unknown search backend {name!r}. Expected 'duckduckgo' or 'fake'.")
