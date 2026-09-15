"""End-to-end pipeline behaviour with injected fakes.

These exercise the real ``Eris`` object, real chunking, ranking and citation
validation. Only the two I/O boundaries - search and fetch - and the model are
substituted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eris.cache import Cache, NullCache
from eris.config import Settings, load_settings
from eris.errors import FetchError, SearchError
from eris.fetch import StaticFetcher
from eris.llm import EchoCitationClient, FakeClient
from eris.models import Document, SearchResult
from eris.pipeline import Eris
from eris.retrieval import Retriever
from eris.search import FakeSearch


def _build(
    settings: Settings,
    *,
    search: FakeSearch,
    fetcher: object,
    cache: object | None = None,
    client: object | None = None,
) -> Eris:
    return Eris(
        settings,
        search_backend=search,
        fetcher=fetcher,  # type: ignore[arg-type]
        cache=cache if cache is not None else NullCache(),  # type: ignore[arg-type]
        retriever=Retriever(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
        ),
        llm_client=client or EchoCitationClient(),  # type: ignore[arg-type]
    )


@pytest.fixture
def eris(settings: Settings, fake_search: FakeSearch, static_fetcher: StaticFetcher) -> Eris:
    return _build(settings, search=fake_search, fetcher=static_fetcher)


class TestAskHappyPath:
    def test_produces_an_answer(self, eris: Eris) -> None:
        assert eris.ask("global interpreter lock").answer.text

    def test_answer_cites_real_sources(self, eris: Eris) -> None:
        answer = eris.ask("global interpreter lock").answer
        assert answer.citations
        assert answer.invalid_citations == ()

    def test_cited_urls_come_from_the_fixtures(self, eris: Eris) -> None:
        result = eris.ask("global interpreter lock")
        known = {
            "https://a.example/python",
            "https://b.example/rust",
            "https://c.example/gardening",
        }
        assert set(result.answer.cited_urls) <= known

    def test_retrieves_the_on_topic_source(self, eris: Eris) -> None:
        result = eris.ask("global interpreter lock free threaded build")
        assert "https://a.example/python" in {c.url for c in result.chunks}

    def test_respects_top_k(self, eris: Eris) -> None:
        assert len(eris.ask("interpreter lock", top_k=2).chunks) == 2

    def test_reports_counts(self, eris: Eris) -> None:
        result = eris.ask("interpreter lock")
        assert result.n_search_results == 3
        assert result.n_documents == 3

    def test_echoes_the_question(self, eris: Eris) -> None:
        assert eris.ask("  interpreter lock  ").question == "interpreter lock"

    def test_max_results_limits_search(self, eris: Eris) -> None:
        assert eris.ask("interpreter lock", max_results=1).n_search_results == 1

    def test_result_serialises(self, eris: Eris) -> None:
        payload = eris.ask("interpreter lock").to_dict()
        assert {"question", "answer", "sources", "timings_ms"} <= set(payload)


class TestTimings:
    def test_every_stage_is_timed(self, eris: Eris) -> None:
        timings = eris.ask("interpreter lock").timing_map()
        assert {"search", "fetch", "retrieve", "answer"} <= set(timings)

    def test_durations_are_non_negative(self, eris: Eris) -> None:
        assert all(t.ms >= 0.0 for t in eris.ask("interpreter lock").timings)

    def test_total_is_the_sum_of_stages(self, eris: Eris) -> None:
        result = eris.ask("interpreter lock")
        assert result.total_ms == pytest.approx(sum(t.ms for t in result.timings))

    def test_stage_order_is_preserved(self, eris: Eris) -> None:
        names = [t.name for t in eris.ask("interpreter lock").timings]
        assert names[:2] == ["search", "fetch"]
        assert names[-1] == "answer"

    def test_timing_renders_readably(self, eris: Eris) -> None:
        assert "ms" in str(eris.ask("interpreter lock").timings[0])


class TestCacheIntegration:
    def test_second_fetch_is_served_from_cache(
        self,
        settings: Settings,
        fake_search: FakeSearch,
        static_fetcher: StaticFetcher,
        cache: Cache,
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=static_fetcher, cache=cache)
        instance.ask("interpreter lock")
        calls_after_first = len(static_fetcher.calls)
        instance.ask("interpreter lock")
        assert len(static_fetcher.calls) == calls_after_first

    def test_search_results_are_cached(
        self,
        settings: Settings,
        fake_search: FakeSearch,
        static_fetcher: StaticFetcher,
        cache: Cache,
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=static_fetcher, cache=cache)
        instance.ask("interpreter lock")
        instance.ask("interpreter lock")
        assert len(fake_search.calls) == 1

    def test_pages_are_written_to_the_cache(
        self,
        settings: Settings,
        fake_search: FakeSearch,
        static_fetcher: StaticFetcher,
        cache: Cache,
    ) -> None:
        _build(settings, search=fake_search, fetcher=static_fetcher, cache=cache).ask("lock")
        assert cache.stats().fresh_pages == 3

    def test_null_cache_refetches_every_time(
        self, settings: Settings, fake_search: FakeSearch, static_fetcher: StaticFetcher
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=static_fetcher)
        instance.ask("interpreter lock")
        instance.ask("interpreter lock")
        assert len(static_fetcher.calls) == 6

    def test_cache_write_failure_does_not_fail_the_query(
        self, settings: Settings, fake_search: FakeSearch, static_fetcher: StaticFetcher
    ) -> None:
        class _BrokenCache(NullCache):
            def put_pages(self, documents: object) -> int:
                from eris.errors import CacheError

                raise CacheError("disk full")

        instance = _build(
            settings, search=fake_search, fetcher=static_fetcher, cache=_BrokenCache()
        )
        assert instance.ask("interpreter lock").answer.text


class TestOfflineMode:
    def test_answers_from_the_cache_without_searching(
        self,
        settings: Settings,
        fake_search: FakeSearch,
        static_fetcher: StaticFetcher,
        cache: Cache,
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=static_fetcher, cache=cache)
        instance.ask("interpreter lock")
        search_calls = len(fake_search.calls)
        fetch_calls = len(static_fetcher.calls)

        result = instance.ask("interpreter lock", use_web=False)
        assert result.from_cache is True
        assert result.answer.text
        assert len(fake_search.calls) == search_calls
        assert len(static_fetcher.calls) == fetch_calls

    def test_offline_mode_skips_the_network_stages(
        self,
        settings: Settings,
        fake_search: FakeSearch,
        static_fetcher: StaticFetcher,
        cache: Cache,
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=static_fetcher, cache=cache)
        instance.ask("interpreter lock")
        timings = instance.ask("interpreter lock", use_web=False).timing_map()
        assert "search" not in timings
        assert "fetch" not in timings
        assert "cache" in timings

    def test_empty_cache_offline_yields_a_refusal(
        self, settings: Settings, fake_search: FakeSearch, static_fetcher: StaticFetcher
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=static_fetcher, cache=NullCache())
        result = instance.ask("interpreter lock", use_web=False)
        assert result.chunks == ()
        assert "cannot answer" in result.answer.text


class TestDegradedPaths:
    def test_search_failure_falls_back_to_the_cache(
        self,
        settings: Settings,
        static_fetcher: StaticFetcher,
        cache: Cache,
        documents: list[Document],
    ) -> None:
        cache.put_pages(documents)

        class _BrokenSearch(FakeSearch):
            def search(self, query: str, max_results: int) -> list[SearchResult]:
                raise SearchError("rate limited")

        instance = _build(settings, search=_BrokenSearch({}), fetcher=static_fetcher, cache=cache)
        result = instance.ask("interpreter lock")
        assert result.chunks
        assert result.answer.citations

    def test_no_search_results_yields_a_refusal(
        self, settings: Settings, static_fetcher: StaticFetcher
    ) -> None:
        instance = _build(settings, search=FakeSearch({}), fetcher=static_fetcher)
        result = instance.ask("nothing matches")
        assert result.n_search_results == 0
        assert result.chunks == ()

    def test_all_fetches_failing_yields_a_refusal(
        self, settings: Settings, fake_search: FakeSearch
    ) -> None:
        class _BrokenFetcher:
            def fetch_many(self, results: object) -> list[Document]:
                return []

        instance = _build(settings, search=fake_search, fetcher=_BrokenFetcher())
        result = instance.ask("interpreter lock")
        assert result.n_documents == 0
        assert "cannot answer" in result.answer.text

    def test_partial_fetch_failure_still_answers(
        self, settings: Settings, fake_search: FakeSearch, documents: list[Document]
    ) -> None:
        instance = _build(settings, search=fake_search, fetcher=StaticFetcher(documents[:1]))
        result = instance.ask("interpreter lock")
        assert result.n_documents == 1
        assert result.answer.citations

    def test_hallucinated_citations_are_reported(
        self, settings: Settings, fake_search: FakeSearch, static_fetcher: StaticFetcher
    ) -> None:
        instance = _build(
            settings,
            search=fake_search,
            fetcher=static_fetcher,
            client=FakeClient(["Claim [1] and [42]."]),
        )
        assert instance.ask("interpreter lock").answer.invalid_citations == (42,)

    def test_fetch_error_on_a_single_page_is_survivable(
        self, settings: Settings, fake_search: FakeSearch, documents: list[Document]
    ) -> None:
        class _Flaky(StaticFetcher):
            def fetch(self, url: str, *, title_hint: str = "") -> Document:
                if "rust" in url:
                    raise FetchError("boom")
                return super().fetch(url, title_hint=title_hint)

        instance = _build(settings, search=fake_search, fetcher=_Flaky(documents))
        assert instance.ask("interpreter lock").n_documents == 2


class TestValidation:
    @pytest.mark.parametrize("question", ["", "   ", "\n"])
    def test_blank_question_is_rejected(self, eris: Eris, question: str) -> None:
        with pytest.raises(ValueError, match="question"):
            eris.ask(question)

    def test_fetch_of_nothing_returns_nothing(self, eris: Eris) -> None:
        assert eris.fetch([]) == []

    def test_fetch_preserves_search_ranking(self, eris: Eris) -> None:
        results = [
            SearchResult(title="c", url="https://c.example/gardening"),
            SearchResult(title="a", url="https://a.example/python"),
        ]
        assert [d.url for d in eris.fetch(results)] == [r.url for r in results]


class TestConstruction:
    def test_defaults_are_derived_from_settings(self, settings: Settings) -> None:
        instance = Eris(settings, llm_client=FakeClient(["x"]))
        assert instance.retriever.top_k == settings.top_k
        assert instance.search_backend.name == "fake"
        instance.close()

    def test_zero_ttl_selects_the_null_cache(self, tmp_path: Path) -> None:
        settings = load_settings(search_backend="fake", cache_ttl_s=0, cache_dir=tmp_path)
        instance = Eris(settings, llm_client=FakeClient(["x"]))
        assert isinstance(instance.cache, NullCache)
        instance.close()

    def test_positive_ttl_selects_the_sqlite_cache(self, tmp_path: Path) -> None:
        settings = load_settings(search_backend="fake", cache_ttl_s=60, cache_dir=tmp_path)
        instance = Eris(settings, llm_client=FakeClient(["x"]))
        assert isinstance(instance.cache, Cache)
        instance.close()

    def test_missing_api_key_raises_only_when_a_client_is_needed(self, settings: Settings) -> None:
        from eris.errors import ConfigError

        with pytest.raises(ConfigError, match="ERIS_ANTHROPIC_API_KEY"):
            Eris(settings)

    def test_context_manager_closes_resources(
        self, settings: Settings, fake_search: FakeSearch, static_fetcher: StaticFetcher
    ) -> None:
        with _build(settings, search=fake_search, fetcher=static_fetcher) as instance:
            assert instance.ask("interpreter lock").answer.text

    def test_close_is_idempotent(self, eris: Eris) -> None:
        eris.close()
        eris.close()
