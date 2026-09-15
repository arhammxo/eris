"""The Eris pipeline: search, fetch, retrieve, answer.

:class:`Eris` owns stage sequencing, the cache read-through, and timing. Every
collaborator (search backend, fetcher, cache, retriever, LLM client) is injected
and defaulted from :class:`~eris.config.Settings`, which is what lets the tests
exercise the real pipeline with fakes and no network.

Stage timings are recorded for every call rather than behind a debug flag: when
a query feels slow, the answer is nearly always "fetch", and you want the number
without re-running the query.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import suppress

from eris.answer import AnswerBuilder
from eris.cache import Cache, NullCache
from eris.config import Settings, load_settings
from eris.errors import ErisError, SearchError
from eris.fetch import Fetcher, PageFetcher
from eris.llm import LLMClient
from eris.models import AskResult, Document, SearchResult, StageTiming
from eris.retrieval import Retriever
from eris.search import SearchBackend, build_search_backend

log = logging.getLogger(__name__)

CacheLike = Cache | NullCache


class _Stopwatch:
    """Accumulates per-stage wall-clock timings in call order."""

    def __init__(self) -> None:
        self._timings: list[StageTiming] = []

    def time(self, name: str) -> _StageTimer:
        return _StageTimer(self, name)

    def record(self, name: str, ms: float) -> None:
        self._timings.append(StageTiming(name=name, ms=ms))

    @property
    def timings(self) -> tuple[StageTiming, ...]:
        return tuple(self._timings)


class _StageTimer:
    """Context manager that records one stage's duration, even on failure."""

    def __init__(self, watch: _Stopwatch, name: str) -> None:
        self._watch = watch
        self._name = name
        self._start = 0.0

    def __enter__(self) -> _StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._watch.record(self._name, (time.perf_counter() - self._start) * 1000.0)


class Eris:
    """Answers questions from live web sources with local retrieval."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        search_backend: SearchBackend | None = None,
        fetcher: Fetcher | None = None,
        cache: CacheLike | None = None,
        retriever: Retriever | None = None,
        llm_client: LLMClient | None = None,
        answer_builder: AnswerBuilder | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.settings.configure_logging()

        self.cache: CacheLike = cache if cache is not None else self._default_cache(self.settings)
        self.search_backend = search_backend or build_search_backend(self.settings.search_backend)
        self.fetcher = fetcher if fetcher is not None else self._default_fetcher(self.settings)
        self.retriever = retriever or self._default_retriever(self.settings)

        if answer_builder is not None:
            self.answer_builder = answer_builder
        else:
            self.answer_builder = AnswerBuilder(llm_client or self._default_llm(self.settings))

    # --- defaults ----------------------------------------------------------
    @staticmethod
    def _default_cache(settings: Settings) -> CacheLike:
        if settings.cache_ttl_s <= 0:
            return NullCache()
        return Cache(settings.cache_path, ttl_s=settings.cache_ttl_s)

    @staticmethod
    def _default_fetcher(settings: Settings) -> Fetcher:
        return PageFetcher(
            timeout_s=settings.fetch_timeout_s,
            max_bytes=settings.max_page_bytes,
            concurrency=settings.fetch_concurrency,
            user_agent=settings.user_agent,
        )

    @staticmethod
    def _default_retriever(settings: Settings) -> Retriever:
        return Retriever(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
            bm25_weight=settings.bm25_weight,
            mmr_lambda=settings.mmr_lambda,
            max_chunks_per_url=settings.max_chunks_per_url,
        )

    @staticmethod
    def _default_llm(settings: Settings) -> LLMClient:
        # Import here so the Anthropic SDK stays optional at module import.
        from eris.llm import AnthropicClient

        return AnthropicClient(
            settings.require_api_key(),
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )

    # --- stages ------------------------------------------------------------
    def search(self, question: str, *, max_results: int | None = None) -> list[SearchResult]:
        """Search for ``question``, reading through the cache first."""
        limit = max_results or self.settings.max_results
        cached = self.cache.get_search(question, limit)
        if cached is not None:
            log.debug("search cache hit for %r", question)
            return cached
        results = self.search_backend.search(question, limit)
        if results:
            with suppress(ErisError):
                # A cache write must never fail a query.
                self.cache.put_search(question, limit, results)
        return results

    def fetch(self, results: Sequence[SearchResult]) -> list[Document]:
        """Fetch pages for ``results``, serving cached bodies where possible."""
        if not results:
            return []

        by_url = {r.url: r for r in results}
        hits, misses = self.cache.get_pages(list(by_url))
        if hits:
            log.debug("page cache: %d hit(s), %d miss(es)", len(hits), len(misses))

        fetched: list[Document] = []
        if misses:
            fetched = self.fetcher.fetch_many([by_url[url] for url in misses])
            if fetched:
                with suppress(ErisError):
                    self.cache.put_pages(fetched)

        # Restore the search ranking, which the cache split does not preserve.
        documents = {doc.url: doc for doc in (*hits, *fetched)}
        return [documents[r.url] for r in results if r.url in documents]

    def cached_documents(self) -> list[Document]:
        """Every unexpired cached page. Backs offline (``--no-web``) mode."""
        return self.cache.all_pages()

    # --- entry point -------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        max_results: int | None = None,
        use_web: bool = True,
        strip_hallucinated: bool = False,
    ) -> AskResult:
        """Answer ``question`` end to end.

        Args:
            question: The natural-language question.
            top_k: Override the number of context chunks.
            max_results: Override the number of search results.
            use_web: When ``False``, answer only from cached pages: no search,
                no fetch, no network.
            strip_hallucinated: Remove invalid ``[n]`` markers from the answer
                text. They are always reported either way.

        Raises:
            ValueError: if ``question`` is blank.
            ErisError: if a stage fails unrecoverably.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")

        watch = _Stopwatch()
        results: list[SearchResult] = []
        documents: list[Document] = []
        offline = not use_web

        if offline:
            with watch.time("cache"):
                documents = self.cached_documents()
            log.debug("offline mode: %d cached document(s)", len(documents))
        else:
            with watch.time("search"):
                try:
                    results = self.search(question, max_results=max_results)
                except SearchError as exc:
                    # Degrade to cached pages rather than failing the query.
                    log.warning("search failed (%s); falling back to cache", exc)
                    results = []
            with watch.time("fetch"):
                documents = self.fetch(results)
            if not documents:
                with watch.time("cache_fallback"):
                    documents = self.cached_documents()

        with watch.time("retrieve"):
            chunks = self.retriever.retrieve(question, documents, top_k=top_k)

        with watch.time("answer"):
            answer = self.answer_builder.build(
                question, chunks, strip_hallucinated=strip_hallucinated
            )

        return AskResult(
            question=question,
            answer=answer,
            chunks=tuple(chunks),
            timings=watch.timings,
            n_search_results=len(results),
            n_documents=len(documents),
            from_cache=offline,
        )

    # --- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Release the HTTP client and cache connection."""
        for target in (self.fetcher, self.cache):
            closer = getattr(target, "close", None)
            if callable(closer):
                with suppress(Exception):
                    closer()

    def __enter__(self) -> Eris:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
