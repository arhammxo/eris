"""On-disk sqlite cache for fetched pages and search results.

Part one of the "optimized local layer". Re-asking a question, or asking two
questions that share sources, should not re-crawl the web: fetching dominates
latency, so a TTL cache turns a multi-second query into a millisecond one.

Keys are normalised URLs, so trivially different spellings of the same page
(``HTTP://Example.com/a?b=1#frag`` vs ``http://example.com/a?b=1``) share one
entry. sqlite is used through the stdlib ``sqlite3`` module: no dependency, and
it gives atomic writes across processes for free.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from eris.errors import CacheError
from eris.models import Document, SearchResult

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url        TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL DEFAULT '',
    fetched_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pages_expires ON pages(expires_at);

CREATE TABLE IF NOT EXISTS searches (
    query      TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_searches_expires ON searches(expires_at);
"""

# Query parameters that identify a campaign, not a document.
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset(
    {"gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "igshid", "ref_src", "ref_url"}
)
_DEFAULT_PORTS = {"http": "80", "https": "443"}


def normalize_url(url: str) -> str:
    """Canonicalise ``url`` for use as a cache key.

    Lowercases scheme and host, drops the fragment, strips default ports and
    tracking parameters, sorts the remaining query, and removes a trailing
    slash on non-root paths. Returns the stripped input unchanged if it cannot
    be parsed, so a weird URL still caches consistently.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned
    if not parts.scheme or not parts.netloc:
        return cleaned

    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if not host:
        return cleaned

    netloc = host
    port = parts.port
    if port is not None and str(port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
        and not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_query(query: str, max_results: int) -> str:
    """Cache key for a search: case-folded, whitespace-collapsed, size-scoped.

    ``max_results`` is part of the key because a cached 3-result response cannot
    satisfy a later request for 8.
    """
    return f"{' '.join((query or '').lower().split())}::{max_results}"


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Snapshot of cache contents."""

    pages: int
    expired_pages: int
    searches: int
    expired_searches: int
    size_bytes: int
    path: str

    @property
    def fresh_pages(self) -> int:
        return max(self.pages - self.expired_pages, 0)

    @property
    def fresh_searches(self) -> int:
        return max(self.searches - self.expired_searches, 0)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "pages": self.pages,
            "fresh_pages": self.fresh_pages,
            "expired_pages": self.expired_pages,
            "searches": self.searches,
            "fresh_searches": self.fresh_searches,
            "expired_searches": self.expired_searches,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_mb, 3),
        }


class Cache:
    """TTL cache over sqlite for pages and search results.

    ``ttl_s = 0`` disables caching: reads always miss and writes are skipped,
    which is the honest interpretation of a zero lifetime.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        ttl_s: int = 3600,
        time_fn: object = time.time,
    ) -> None:
        if ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        self.path = Path(path)
        self.ttl_s = ttl_s
        self._time_fn = time_fn
        self._conn: sqlite3.Connection | None = None

    # --- plumbing ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    def _now(self) -> float:
        return float(self._time_fn())  # type: ignore[operator]

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        try:
            if self.path.parent and str(self.path.parent) not in ("", "."):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # WAL keeps concurrent readers from blocking the fetch writers.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise CacheError(f"Could not open cache at {self.path}: {exc}") from exc
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Cache:
        self._connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- pages -------------------------------------------------------------
    def get_page(self, url: str) -> Document | None:
        """Return the cached page for ``url``, or ``None`` if missing/expired."""
        if not self.enabled:
            return None
        key = normalize_url(url)
        if not key:
            return None
        try:
            row = self._connect().execute("SELECT * FROM pages WHERE url = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache read failed for {url}: {exc}") from exc
        if row is None:
            return None
        if float(row["expires_at"]) <= self._now():
            log.debug("cache entry expired for %s", key)
            return None
        return Document(
            url=url,
            title=row["title"],
            text=row["text"],
            fetched_at=datetime.fromtimestamp(float(row["fetched_at"]), tz=timezone.utc),
        )

    def put_page(self, document: Document) -> None:
        """Insert or replace the cache entry for ``document``."""
        if not self.enabled:
            return
        key = normalize_url(document.url)
        if not key:
            return
        now = self._now()
        try:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO pages (url, title, text, fetched_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, document.title, document.text, now, now + self.ttl_s),
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache write failed for {document.url}: {exc}") from exc

    def put_pages(self, documents: Sequence[Document]) -> int:
        """Cache many documents in one transaction. Returns the count written."""
        if not self.enabled or not documents:
            return 0
        now = self._now()
        rows = [
            (normalize_url(d.url), d.title, d.text, now, now + self.ttl_s)
            for d in documents
            if normalize_url(d.url)
        ]
        if not rows:
            return 0
        try:
            conn = self._connect()
            conn.executemany(
                "INSERT OR REPLACE INTO pages (url, title, text, fetched_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Bulk cache write failed: {exc}") from exc
        return len(rows)

    def get_pages(self, urls: Sequence[str]) -> tuple[list[Document], list[str]]:
        """Split ``urls`` into cached documents and misses, preserving order."""
        hits: list[Document] = []
        misses: list[str] = []
        for url in urls:
            document = self.get_page(url)
            if document is None:
                misses.append(url)
            else:
                hits.append(document)
        return hits, misses

    def all_pages(self, *, include_expired: bool = False) -> list[Document]:
        """Every cached page, newest first. Backs ``--no-web`` mode."""
        sql = "SELECT * FROM pages"
        params: tuple[object, ...] = ()
        if not include_expired:
            sql += " WHERE expires_at > ?"
            params = (self._now(),)
        sql += " ORDER BY fetched_at DESC"
        try:
            rows = self._connect().execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache scan failed: {exc}") from exc
        return [
            Document(
                url=row["url"],
                title=row["title"],
                text=row["text"],
                fetched_at=datetime.fromtimestamp(float(row["fetched_at"]), tz=timezone.utc),
            )
            for row in rows
        ]

    # --- searches ----------------------------------------------------------
    def get_search(self, query: str, max_results: int) -> list[SearchResult] | None:
        """Return cached results for ``query``, or ``None`` if missing/expired."""
        if not self.enabled:
            return None
        key = normalize_query(query, max_results)
        try:
            row = (
                self._connect()
                .execute("SELECT * FROM searches WHERE query = ?", (key,))
                .fetchone()
            )
        except sqlite3.Error as exc:
            raise CacheError(f"Cache read failed for query {query!r}: {exc}") from exc
        if row is None or float(row["expires_at"]) <= self._now():
            return None
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            log.warning("discarding corrupt cached search payload for %r", query)
            return None
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item["url"],
                snippet=item.get("snippet", ""),
            )
            for item in payload
            if isinstance(item, dict) and item.get("url")
        ]

    def put_search(self, query: str, max_results: int, results: Sequence[SearchResult]) -> None:
        """Cache the result list for ``query``."""
        if not self.enabled:
            return
        key = normalize_query(query, max_results)
        payload = json.dumps(
            [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
        )
        now = self._now()
        try:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO searches (query, payload, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (key, payload, now, now + self.ttl_s),
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache write failed for query {query!r}: {exc}") from exc

    # --- maintenance -------------------------------------------------------
    def purge_expired(self) -> int:
        """Delete expired rows. Returns the number removed."""
        now = self._now()
        try:
            conn = self._connect()
            removed = conn.execute("DELETE FROM pages WHERE expires_at <= ?", (now,)).rowcount
            removed += conn.execute(
                "DELETE FROM searches WHERE expires_at <= ?", (now,)
            ).rowcount
            conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache purge failed: {exc}") from exc
        return max(removed, 0)

    def clear(self) -> int:
        """Delete every row. Returns the number removed."""
        try:
            conn = self._connect()
            removed = conn.execute("DELETE FROM pages").rowcount
            removed += conn.execute("DELETE FROM searches").rowcount
            conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache clear failed: {exc}") from exc
        return max(removed, 0)

    def stats(self) -> CacheStats:
        """Summarise cache contents and on-disk size."""
        now = self._now()
        try:
            conn = self._connect()
            pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            expired_pages = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE expires_at <= ?", (now,)
            ).fetchone()[0]
            searches = conn.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
            expired_searches = conn.execute(
                "SELECT COUNT(*) FROM searches WHERE expires_at <= ?", (now,)
            ).fetchone()[0]
        except sqlite3.Error as exc:
            raise CacheError(f"Cache stats failed: {exc}") from exc
        size = self.path.stat().st_size if self.path.exists() else 0
        return CacheStats(
            pages=int(pages),
            expired_pages=int(expired_pages),
            searches=int(searches),
            expired_searches=int(expired_searches),
            size_bytes=size,
            path=str(self.path),
        )


class NullCache:
    """Cache that stores nothing, for ``--no-cache`` runs and unit tests."""

    enabled = False
    ttl_s = 0
    path = Path("<null>")

    def get_page(self, url: str) -> Document | None:
        return None

    def put_page(self, document: Document) -> None:
        return None

    def put_pages(self, documents: Sequence[Document]) -> int:
        return 0

    def get_pages(self, urls: Sequence[str]) -> tuple[list[Document], list[str]]:
        return [], list(urls)

    def all_pages(self, *, include_expired: bool = False) -> list[Document]:
        return []

    def get_search(self, query: str, max_results: int) -> list[SearchResult] | None:
        return None

    def put_search(self, query: str, max_results: int, results: Sequence[SearchResult]) -> None:
        return None

    def purge_expired(self) -> int:
        return 0

    def clear(self) -> int:
        return 0

    def stats(self) -> CacheStats:
        return CacheStats(0, 0, 0, 0, 0, str(self.path))

    def close(self) -> None:
        return None
