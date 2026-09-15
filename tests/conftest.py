"""Shared fixtures.

Every fixture here is offline. There is no fixture that opens a socket, and the
``no_network`` autouse fixture fails any test that tries, so a regression that
reintroduces a real HTTP or LLM call is caught rather than merely slow.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from eris.cache import Cache
from eris.config import Settings, load_settings
from eris.fetch import PageFetcher, StaticFetcher
from eris.llm import EchoCitationClient, FakeClient
from eris.models import Chunk, Document, SearchResult
from eris.retrieval import Retriever
from eris.search import FakeSearch

# Environment variables that would otherwise leak the developer's real config
# (or a real API key) into a test run.
_ERIS_ENV_VARS = (
    "ERIS_ANTHROPIC_API_KEY",
    "ERIS_MODEL",
    "ERIS_MAX_TOKENS",
    "ERIS_TEMPERATURE",
    "ERIS_SEARCH_BACKEND",
    "ERIS_MAX_RESULTS",
    "ERIS_FETCH_TIMEOUT_S",
    "ERIS_MAX_PAGE_BYTES",
    "ERIS_FETCH_CONCURRENCY",
    "ERIS_USER_AGENT",
    "ERIS_CACHE_DIR",
    "ERIS_CACHE_TTL_S",
    "ERIS_TOP_K",
    "ERIS_CHUNK_SIZE",
    "ERIS_CHUNK_OVERLAP",
    "ERIS_BM25_WEIGHT",
    "ERIS_MMR_LAMBDA",
    "ERIS_MAX_CHUNKS_PER_URL",
    "ERIS_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate settings from the host environment and any local .env file."""
    for name in _ERIS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Settings reads .env relative to the cwd; point cwd at an empty tmp dir.
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any real outbound socket connection an immediate test failure."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline-friendly settings pointed at a temp cache."""
    return load_settings(
        search_backend="fake",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=60,
        top_k=4,
        chunk_size=300,
        chunk_overlap=50,
    )


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[Cache]:
    """A real sqlite cache in a temp directory."""
    instance = Cache(tmp_path / "cache" / "eris.sqlite3", ttl_s=60)
    yield instance
    instance.close()


@pytest.fixture
def documents() -> list[Document]:
    """Three documents on distinct topics, long enough to produce many chunks."""
    return [
        Document(
            url="https://a.example/python",
            title="Python 3.13 release notes",
            text=(
                "Python 3.13 disables the global interpreter lock in an experimental "
                "free-threaded build selected with the disable-gil configure flag. "
                "Threads can then execute bytecode in parallel across cores. "
            )
            * 6,
        ),
        Document(
            url="https://b.example/rust",
            title="Rust 1.80 release notes",
            text=(
                "Rust 1.80 stabilised LazyCell and LazyLock in the standard library "
                "and extended the borrow checker diagnostics for closures. "
            )
            * 6,
        ),
        Document(
            url="https://c.example/gardening",
            title="Companion planting guide",
            text=(
                "Tomatoes grow well beside basil, and marigolds deter nematodes "
                "from root vegetables planted in the same bed. "
            )
            * 6,
        ),
    ]


@pytest.fixture
def search_results(documents: list[Document]) -> list[SearchResult]:
    return [SearchResult(title=doc.title, url=doc.url, snippet=doc.text[:60]) for doc in documents]


@pytest.fixture
def chunks() -> list[Chunk]:
    """Four chunks across two URLs, two of them near-duplicates."""
    return [
        Chunk(
            url="https://a.example/one",
            title="A one",
            text="the global interpreter lock is disabled in free threaded builds",
            index=0,
        ),
        Chunk(
            url="https://a.example/one",
            title="A one",
            text="the global interpreter lock is disabled in free threaded builds too",
            index=1,
        ),
        Chunk(
            url="https://b.example/two",
            title="B two",
            text="marigolds deter nematodes from root vegetables in the same bed",
            index=0,
        ),
        Chunk(
            url="https://c.example/three",
            title="C three",
            text="lazycell and lazylock were stabilised in the rust standard library",
            index=0,
        ),
    ]


@pytest.fixture
def fake_search(search_results: list[SearchResult]) -> FakeSearch:
    return FakeSearch({"test query": search_results}, default=search_results)


@pytest.fixture
def static_fetcher(documents: list[Document]) -> StaticFetcher:
    return StaticFetcher(documents)


@pytest.fixture
def fake_llm() -> FakeClient:
    return FakeClient(["The answer is grounded in the sources [1]."])


@pytest.fixture
def echo_llm() -> EchoCitationClient:
    return EchoCitationClient()


@pytest.fixture
def retriever() -> Retriever:
    return Retriever(chunk_size=300, chunk_overlap=50, top_k=4)


def mock_fetcher(
    handler: object,
    *,
    timeout_s: float = 5.0,
    max_bytes: int = 1_000_000,
    concurrency: int = 4,
) -> PageFetcher:
    """Build a :class:`PageFetcher` backed by an ``httpx.MockTransport``.

    This is how fetch behaviour is tested without a socket: real ``httpx``
    request/response plumbing, fabricated wire responses.
    """
    client = httpx.Client(
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        follow_redirects=True,
    )
    return PageFetcher(
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        concurrency=concurrency,
        client=client,
    )


@pytest.fixture
def html_page() -> str:
    """A page whose real content is surrounded by boilerplate."""
    return """<!DOCTYPE html>
<html><head>
  <title>  Widget   Reference  </title>
  <style>body { color: red; }</style>
  <script>window.tracking = true;</script>
</head>
<body>
  <nav><a href="/">Home</a><a href="/docs">Docs</a></nav>
  <header>Accept all cookies</header>
  <main>
    <article>
      <h1>Widget Reference</h1>
      <p>The widget accepts a payload of up to 10 MiB per request.</p>
      <p>Larger payloads must use the multipart endpoint.</p>
    </article>
  </main>
  <aside>Subscribe now</aside>
  <footer>&copy; 2026 Example Corp. All rights reserved.</footer>
  <noscript>Enable JavaScript</noscript>
</body></html>"""
