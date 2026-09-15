"""Search backends: result normalisation, fakes, and error wrapping."""

from __future__ import annotations

import pytest

from eris.errors import DependencyMissingError, SearchError
from eris.models import SearchResult
from eris.search import (
    DuckDuckGoSearch,
    FakeSearch,
    SearchBackend,
    build_search_backend,
    normalize_results,
)


class TestNormalizeResults:
    def test_maps_ddgs_keys(self) -> None:
        results = normalize_results([{"title": "T", "href": "https://e.com/a", "body": "S"}])
        assert results == [SearchResult(title="T", url="https://e.com/a", snippet="S")]

    @pytest.mark.parametrize("key", ["href", "url", "link"])
    def test_accepts_alternative_url_keys(self, key: str) -> None:
        """ddgs has renamed this field across releases."""
        results = normalize_results([{"title": "T", key: "https://e.com/a"}])
        assert results[0].url == "https://e.com/a"

    @pytest.mark.parametrize("key", ["body", "snippet", "description", "excerpt"])
    def test_accepts_alternative_snippet_keys(self, key: str) -> None:
        results = normalize_results([{"title": "T", "href": "https://e.com/a", key: "S"}])
        assert results[0].snippet == "S"

    def test_falls_back_to_url_when_title_missing(self) -> None:
        assert normalize_results([{"href": "https://e.com/a"}])[0].title == "https://e.com/a"

    def test_missing_snippet_becomes_empty_string(self) -> None:
        assert normalize_results([{"title": "T", "href": "https://e.com/a"}])[0].snippet == ""

    def test_rows_without_a_url_are_dropped(self) -> None:
        assert normalize_results([{"title": "T"}, {"title": "U", "href": ""}]) == []

    def test_deduplicates_by_url_preserving_rank(self) -> None:
        rows = [
            {"title": "First", "href": "https://e.com/a"},
            {"title": "Dup", "href": "https://e.com/a"},
            {"title": "Second", "href": "https://e.com/b"},
        ]
        results = normalize_results(rows)
        assert [r.title for r in results] == ["First", "Second"]

    def test_non_mapping_rows_are_ignored(self) -> None:
        rows = ["not a dict", None, {"title": "T", "href": "https://e.com/a"}]
        assert len(normalize_results(rows)) == 1  # type: ignore[arg-type]

    def test_values_are_stripped(self) -> None:
        results = normalize_results([{"title": "  T  ", "href": "  https://e.com/a  "}])
        assert results[0].title == "T"
        assert results[0].url == "https://e.com/a"

    def test_empty_input_yields_empty_output(self) -> None:
        assert normalize_results([]) == []


class TestFakeSearch:
    def test_returns_registered_results(self, search_results: list[SearchResult]) -> None:
        backend = FakeSearch({"my query": search_results})
        assert backend.search("my query", 10) == search_results

    def test_respects_max_results(self, search_results: list[SearchResult]) -> None:
        backend = FakeSearch({"q": search_results})
        assert len(backend.search("q", 2)) == 2

    def test_lookup_ignores_case_and_spacing(self, search_results: list[SearchResult]) -> None:
        backend = FakeSearch({"My  Query": search_results})
        assert backend.search("my query", 10) == search_results

    def test_falls_back_to_substring_match(self, search_results: list[SearchResult]) -> None:
        """Lets eval files use natural questions without exact-string brittleness."""
        backend = FakeSearch({"python gil": search_results})
        assert backend.search("python gil changes in 3.13", 10) == search_results

    def test_falls_back_to_default(self, search_results: list[SearchResult]) -> None:
        backend = FakeSearch({}, default=search_results)
        assert backend.search("anything at all", 10) == search_results

    def test_unmatched_query_without_default_is_empty(self) -> None:
        assert FakeSearch({}).search("nothing", 10) == []

    def test_records_calls(self) -> None:
        backend = FakeSearch({})
        backend.search("first", 5)
        backend.search("second", 5)
        assert backend.calls == ["first", "second"]

    def test_results_can_be_registered_later(self, search_results: list[SearchResult]) -> None:
        backend = FakeSearch({})
        backend.add("later", search_results)
        assert backend.search("later", 10) == search_results

    def test_satisfies_the_backend_protocol(self) -> None:
        assert isinstance(FakeSearch({}), SearchBackend)

    def test_exposes_its_name(self) -> None:
        assert FakeSearch({}).name == "fake"


class TestDuckDuckGoSearch:
    def test_blank_query_returns_empty_without_calling_out(self) -> None:
        assert DuckDuckGoSearch().search("   ", 5) == []

    def test_non_positive_max_results_returns_empty(self) -> None:
        assert DuckDuckGoSearch().search("python", 0) == []

    def test_backend_errors_are_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Boom:
            def text(self, *args: object, **kwargs: object) -> list[dict[str, str]]:
                raise RuntimeError("rate limited")

        monkeypatch.setattr(DuckDuckGoSearch, "_client", staticmethod(lambda: _Boom()))
        with pytest.raises(SearchError, match="DuckDuckGo search failed"):
            DuckDuckGoSearch().search("python", 5)

    def test_missing_dependency_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise() -> object:
            raise DependencyMissingError("ddgs", "search")

        monkeypatch.setattr(DuckDuckGoSearch, "_client", staticmethod(_raise))
        with pytest.raises(DependencyMissingError, match="eris\\[search\\]"):
            DuckDuckGoSearch().search("python", 5)

    def test_results_are_normalised_and_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Stub:
            def text(self, *args: object, **kwargs: object) -> list[dict[str, str]]:
                return [
                    {"title": f"T{i}", "href": f"https://e.com/{i}", "body": "s"} for i in range(10)
                ]

        monkeypatch.setattr(DuckDuckGoSearch, "_client", staticmethod(lambda: _Stub()))
        results = DuckDuckGoSearch().search("python", 3)
        assert len(results) == 3
        assert results[0].url == "https://e.com/0"

    def test_none_response_becomes_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Stub:
            def text(self, *args: object, **kwargs: object) -> None:
                return None

        monkeypatch.setattr(DuckDuckGoSearch, "_client", staticmethod(lambda: _Stub()))
        assert DuckDuckGoSearch().search("python", 5) == []

    def test_query_and_limit_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        class _Stub:
            def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
                seen["query"] = query
                seen.update(kwargs)
                return []

        monkeypatch.setattr(DuckDuckGoSearch, "_client", staticmethod(lambda: _Stub()))
        DuckDuckGoSearch(region="uk-en").search("  python gil  ", 4)
        assert seen["query"] == "python gil"
        assert seen["max_results"] == 4
        assert seen["region"] == "uk-en"


class TestBuildSearchBackend:
    def test_builds_duckduckgo(self) -> None:
        assert build_search_backend("duckduckgo").name == "duckduckgo"

    def test_builds_fake(self) -> None:
        assert build_search_backend("fake").name == "fake"

    @pytest.mark.parametrize("name", ["DuckDuckGo", "  fake  ", "FAKE"])
    def test_name_matching_is_lenient(self, name: str) -> None:
        assert build_search_backend(name) is not None

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(SearchError, match="Unknown search backend"):
            build_search_backend("bing")
