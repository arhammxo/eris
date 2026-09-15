"""Domain model invariants and serialisation."""

from __future__ import annotations

import pytest

import eris
from eris.models import (
    Answer,
    AskResult,
    Chunk,
    Citation,
    Document,
    SearchResult,
    StageTiming,
    utcnow,
)


class TestSearchResult:
    def test_holds_its_fields(self) -> None:
        result = SearchResult(title="T", url="https://e.com/a", snippet="S")
        assert (result.title, result.url, result.snippet) == ("T", "https://e.com/a", "S")

    def test_snippet_defaults_to_empty(self) -> None:
        assert SearchResult(title="T", url="https://e.com/a").snippet == ""

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="url must not be empty"):
            SearchResult(title="T", url="")

    def test_is_immutable(self) -> None:
        result = SearchResult(title="T", url="https://e.com/a")
        with pytest.raises(AttributeError):
            result.title = "other"  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        assert len({SearchResult(title="T", url="https://e.com/a")}) == 1


class TestDocument:
    def test_defaults_fetched_at_to_now(self) -> None:
        assert Document(url="https://e.com", title="T", text="body").fetched_at.tzinfo is not None

    def test_reports_its_length(self) -> None:
        assert Document(url="https://e.com", title="T", text="12345").n_chars == 5

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="url must not be empty"):
            Document(url="", title="T", text="body")

    def test_is_immutable(self) -> None:
        document = Document(url="https://e.com", title="T", text="body")
        with pytest.raises(AttributeError):
            document.text = "other"  # type: ignore[misc]


class TestChunk:
    def test_source_id_combines_url_and_index(self) -> None:
        assert Chunk(url="https://e.com/a", title="T", text="x", index=3).source_id == (
            "https://e.com/a#3"
        )

    def test_source_id_is_unique_per_index(self) -> None:
        first = Chunk(url="https://e.com/a", title="T", text="x", index=0)
        second = Chunk(url="https://e.com/a", title="T", text="y", index=1)
        assert first.source_id != second.source_id

    def test_score_defaults_to_zero(self) -> None:
        assert Chunk(url="https://e.com", title="T", text="x").score == 0.0

    def test_with_score_returns_a_copy(self) -> None:
        original = Chunk(url="https://e.com", title="T", text="x", index=2)
        scored = original.with_score(0.75)
        assert scored.score == 0.75
        assert original.score == 0.0
        assert scored.source_id == original.source_id

    def test_with_score_preserves_every_other_field(self) -> None:
        original = Chunk(url="https://e.com", title="Title", text="body", index=4)
        scored = original.with_score(0.5)
        assert (scored.url, scored.title, scored.text, scored.index) == (
            original.url,
            original.title,
            original.text,
            original.index,
        )


class TestCitation:
    def test_serialises(self) -> None:
        assert Citation(n=1, url="https://e.com", title="T").to_dict() == {
            "n": 1,
            "url": "https://e.com",
            "title": "T",
        }


class TestAnswer:
    def test_defaults_are_empty(self) -> None:
        answer = Answer(text="hello")
        assert answer.citations == ()
        assert answer.invalid_citations == ()
        assert answer.cited_urls == ()

    def test_cited_urls_follow_citation_order(self) -> None:
        answer = Answer(
            text="x",
            citations=(
                Citation(n=2, url="https://b.com", title="B"),
                Citation(n=1, url="https://a.com", title="A"),
            ),
        )
        assert answer.cited_urls == ("https://b.com", "https://a.com")

    def test_serialises_with_invalid_citations(self) -> None:
        payload = Answer(text="x", invalid_citations=(9, 12)).to_dict()
        assert payload["invalid_citations"] == [9, 12]
        assert payload["text"] == "x"


class TestStageTiming:
    def test_renders_with_units(self) -> None:
        assert str(StageTiming(name="fetch", ms=12.345)) == "fetch=12.3ms"


class TestAskResult:
    def _result(self) -> AskResult:
        return AskResult(
            question="q",
            answer=Answer(text="a", citations=(Citation(n=1, url="https://e.com", title="T"),)),
            chunks=(Chunk(url="https://e.com", title="T", text="body", score=0.5),),
            timings=(StageTiming("search", 1.0), StageTiming("fetch", 2.0)),
            n_search_results=3,
            n_documents=2,
        )

    def test_total_sums_stage_durations(self) -> None:
        assert self._result().total_ms == pytest.approx(3.0)

    def test_timing_map_is_keyed_by_stage(self) -> None:
        assert self._result().timing_map() == {"search": 1.0, "fetch": 2.0}

    def test_total_of_no_stages_is_zero(self) -> None:
        assert AskResult(question="q", answer=Answer(text="a")).total_ms == 0.0

    def test_defaults_to_online(self) -> None:
        assert AskResult(question="q", answer=Answer(text="a")).from_cache is False

    def test_serialises_sources_and_timings(self) -> None:
        payload = self._result().to_dict()
        assert payload["question"] == "q"
        assert payload["sources"][0]["url"] == "https://e.com"
        assert payload["timings_ms"] == {"search": 1.0, "fetch": 2.0}
        assert payload["n_documents"] == 2

    def test_serialised_scores_are_rounded(self) -> None:
        sources = self._result().to_dict()["sources"]
        assert sources[0]["score"] == 0.5  # type: ignore[index]


class TestUtcNow:
    def test_is_timezone_aware(self) -> None:
        assert utcnow().tzinfo is not None

    def test_is_utc(self) -> None:
        assert utcnow().utcoffset() is not None
        assert utcnow().utcoffset().total_seconds() == 0.0  # type: ignore[union-attr]


class TestPublicApi:
    def test_version_is_exposed(self) -> None:
        assert eris.__version__ == "0.1.0"

    @pytest.mark.parametrize(
        "name",
        [
            "Eris",
            "Settings",
            "load_settings",
            "Document",
            "Chunk",
            "Answer",
            "SearchResult",
            "BM25Scorer",
            "Retriever",
            "mmr_select",
            "Cache",
            "NullCache",
            "PageFetcher",
            "StaticFetcher",
            "FakeSearch",
            "FakeClient",
            "AnswerBuilder",
            "ErisError",
        ],
    )
    def test_public_names_are_importable(self, name: str) -> None:
        assert hasattr(eris, name)

    def test_all_is_sorted(self) -> None:
        assert list(eris.__all__) == sorted(eris.__all__)

    def test_every_exported_name_resolves(self) -> None:
        missing = [name for name in eris.__all__ if not hasattr(eris, name)]
        assert missing == []

    def test_import_does_not_require_optional_dependencies(self) -> None:
        """``import eris`` must not pull in anthropic, ddgs or sklearn."""
        import subprocess
        import sys

        code = (
            "import sys, eris;"
            "leaked=[m for m in ('anthropic','ddgs','sklearn','trafilatura') "
            "if m in sys.modules];"
            "print(','.join(leaked))"
        )
        output = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert output.stdout.strip() == ""


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        "name",
        [
            "ConfigError",
            "SearchError",
            "FetchError",
            "CacheError",
            "RetrievalError",
            "LLMError",
            "DependencyMissingError",
        ],
    )
    def test_every_error_derives_from_the_base(self, name: str) -> None:
        assert issubclass(getattr(eris, name, None) or _lookup(name), eris.ErisError)

    def test_dependency_error_names_the_install_extra(self) -> None:
        error = eris.DependencyMissingError("ddgs", "search")
        assert error.package == "ddgs"
        assert error.extra == "search"
        assert "pip install 'eris[search]'" in str(error)


def _lookup(name: str) -> type[BaseException]:
    from eris import errors

    return getattr(errors, name)
