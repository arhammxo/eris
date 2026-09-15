"""Cache TTL semantics, URL normalisation and maintenance operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from eris.cache import Cache, NullCache, normalize_query, normalize_url
from eris.errors import CacheError
from eris.models import Document, SearchResult


class _Clock:
    """Controllable time source, so TTL tests never sleep."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def timed_cache(tmp_path: Path, clock: _Clock) -> Cache:
    return Cache(tmp_path / "c" / "eris.sqlite3", ttl_s=100, time_fn=clock)


def _doc(url: str = "https://example.com/a", text: str = "body text") -> Document:
    return Document(url=url, title="Title", text=text)


class TestNormalizeUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HTTP://Example.COM/Path", "http://example.com/Path"),
            ("https://example.com/a#frag", "https://example.com/a"),
            ("https://example.com", "https://example.com/"),
            ("https://example.com/a/", "https://example.com/a"),
            ("https://example.com:443/a", "https://example.com/a"),
            ("http://example.com:80/a", "http://example.com/a"),
            ("https://example.com:8443/a", "https://example.com:8443/a"),
            ("  https://example.com/a  ", "https://example.com/a"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert normalize_url(raw) == expected

    def test_drops_utm_parameters(self) -> None:
        assert normalize_url("https://e.com/a?utm_source=x&id=7") == "https://e.com/a?id=7"

    def test_drops_click_ids(self) -> None:
        assert normalize_url("https://e.com/a?gclid=abc&fbclid=d") == "https://e.com/a"

    def test_sorts_query_parameters(self) -> None:
        assert normalize_url("https://e.com/a?b=2&a=1") == normalize_url("https://e.com/a?a=1&b=2")

    def test_preserves_meaningful_parameters(self) -> None:
        assert "page=3" in normalize_url("https://e.com/a?page=3")

    def test_case_of_path_is_preserved(self) -> None:
        """Paths are case-sensitive on most servers, so they must not be folded."""
        assert normalize_url("https://e.com/CaseSensitive") == "https://e.com/CaseSensitive"

    def test_empty_input_maps_to_empty(self) -> None:
        assert normalize_url("") == ""
        assert normalize_url("   ") == ""

    def test_non_url_input_is_returned_unchanged(self) -> None:
        assert normalize_url("not a url") == "not a url"

    def test_is_idempotent(self) -> None:
        once = normalize_url("HTTPS://Example.com:443/a/?utm_source=x#f")
        assert normalize_url(once) == once


class TestNormalizeQuery:
    def test_folds_case_and_whitespace(self) -> None:
        assert normalize_query("  Hello   World ", 8) == normalize_query("hello world", 8)

    def test_max_results_is_part_of_the_key(self) -> None:
        """A cached 3-result response cannot satisfy a request for 8."""
        assert normalize_query("q", 3) != normalize_query("q", 8)


class TestCachePages:
    def test_roundtrip(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc(text="hello world"))
        cached = timed_cache.get_page("https://example.com/a")
        assert cached is not None
        assert cached.text == "hello world"
        assert cached.title == "Title"

    def test_miss_returns_none(self, timed_cache: Cache) -> None:
        assert timed_cache.get_page("https://example.com/missing") is None

    def test_entry_survives_until_ttl(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc())
        clock.advance(99)
        assert timed_cache.get_page("https://example.com/a") is not None

    def test_entry_expires_at_ttl(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc())
        clock.advance(100)
        assert timed_cache.get_page("https://example.com/a") is None

    def test_entry_expires_after_ttl(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc())
        clock.advance(1_000)
        assert timed_cache.get_page("https://example.com/a") is None

    def test_rewrite_refreshes_the_ttl(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc(text="old"))
        clock.advance(90)
        timed_cache.put_page(_doc(text="new"))
        clock.advance(90)
        cached = timed_cache.get_page("https://example.com/a")
        assert cached is not None
        assert cached.text == "new"

    def test_equivalent_urls_share_one_entry(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc(url="https://Example.com/a?utm_source=x", text="shared"))
        cached = timed_cache.get_page("https://example.com/a")
        assert cached is not None
        assert cached.text == "shared"

    def test_lookup_returns_the_requested_url_spelling(self, timed_cache: Cache) -> None:
        """The key is normalised, but the caller's URL is echoed back."""
        timed_cache.put_page(_doc(url="https://example.com/a"))
        cached = timed_cache.get_page("https://EXAMPLE.com/a#frag")
        assert cached is not None
        assert cached.url == "https://EXAMPLE.com/a#frag"

    def test_empty_url_is_ignored(self, timed_cache: Cache) -> None:
        assert timed_cache.get_page("") is None

    def test_fetched_at_is_timezone_aware(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc())
        cached = timed_cache.get_page("https://example.com/a")
        assert cached is not None
        assert cached.fetched_at.tzinfo is not None

    def test_bulk_write_returns_count(self, timed_cache: Cache) -> None:
        docs = [_doc(url=f"https://e.com/{i}") for i in range(4)]
        assert timed_cache.put_pages(docs) == 4

    def test_bulk_write_of_nothing_is_zero(self, timed_cache: Cache) -> None:
        assert timed_cache.put_pages([]) == 0

    def test_get_pages_splits_hits_and_misses(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc(url="https://e.com/1"))
        hits, misses = timed_cache.get_pages(["https://e.com/1", "https://e.com/2"])
        assert [d.url for d in hits] == ["https://e.com/1"]
        assert misses == ["https://e.com/2"]

    def test_all_pages_excludes_expired(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc(url="https://e.com/old"))
        clock.advance(150)
        timed_cache.put_page(_doc(url="https://e.com/new"))
        urls = {d.url for d in timed_cache.all_pages()}
        assert urls == {"https://e.com/new"}

    def test_all_pages_can_include_expired(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc(url="https://e.com/old"))
        clock.advance(150)
        assert len(timed_cache.all_pages(include_expired=True)) == 1


class TestCacheSearches:
    def test_roundtrip(self, timed_cache: Cache) -> None:
        results = [SearchResult(title="T", url="https://e.com/a", snippet="s")]
        timed_cache.put_search("my query", 8, results)
        cached = timed_cache.get_search("my query", 8)
        assert cached is not None
        assert cached[0].url == "https://e.com/a"
        assert cached[0].snippet == "s"

    def test_miss_returns_none(self, timed_cache: Cache) -> None:
        assert timed_cache.get_search("never asked", 8) is None

    def test_expires_after_ttl(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_search("q", 8, [SearchResult(title="T", url="https://e.com/a")])
        clock.advance(101)
        assert timed_cache.get_search("q", 8) is None

    def test_different_max_results_is_a_different_key(self, timed_cache: Cache) -> None:
        timed_cache.put_search("q", 3, [SearchResult(title="T", url="https://e.com/a")])
        assert timed_cache.get_search("q", 8) is None

    def test_query_matching_ignores_case_and_spacing(self, timed_cache: Cache) -> None:
        timed_cache.put_search("My  Query", 8, [SearchResult(title="T", url="https://e.com/a")])
        assert timed_cache.get_search("my query", 8) is not None

    def test_empty_result_list_roundtrips(self, timed_cache: Cache) -> None:
        timed_cache.put_search("q", 8, [])
        assert timed_cache.get_search("q", 8) == []

    def test_corrupt_payload_is_discarded(self, timed_cache: Cache) -> None:
        timed_cache.put_search("q", 8, [SearchResult(title="T", url="https://e.com/a")])
        conn = timed_cache._connect()
        conn.execute("UPDATE searches SET payload = ?", ("{not json",))
        conn.commit()
        assert timed_cache.get_search("q", 8) is None


class TestCacheMaintenance:
    def test_purge_removes_only_expired(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc(url="https://e.com/old"))
        clock.advance(150)
        timed_cache.put_page(_doc(url="https://e.com/new"))
        assert timed_cache.purge_expired() == 1
        assert timed_cache.get_page("https://e.com/new") is not None

    def test_clear_removes_everything(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc(url="https://e.com/1"))
        timed_cache.put_search("q", 8, [SearchResult(title="T", url="https://e.com/1")])
        assert timed_cache.clear() == 2
        assert timed_cache.stats().pages == 0

    def test_stats_counts_fresh_and_expired(self, timed_cache: Cache, clock: _Clock) -> None:
        timed_cache.put_page(_doc(url="https://e.com/old"))
        clock.advance(150)
        timed_cache.put_page(_doc(url="https://e.com/new"))
        stats = timed_cache.stats()
        assert stats.pages == 2
        assert stats.expired_pages == 1
        assert stats.fresh_pages == 1

    def test_stats_reports_size(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc(text="x" * 5_000))
        assert timed_cache.stats().size_bytes > 0

    def test_stats_serialises(self, timed_cache: Cache) -> None:
        payload = timed_cache.stats().to_dict()
        assert {"pages", "searches", "size_mb", "path"} <= set(payload)


class TestCacheLifecycle:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path / "deep" / "nested" / "eris.sqlite3", ttl_s=60)
        cache.put_page(_doc())
        assert (tmp_path / "deep" / "nested" / "eris.sqlite3").exists()
        cache.close()

    def test_data_survives_reopening(self, tmp_path: Path) -> None:
        path = tmp_path / "eris.sqlite3"
        with Cache(path, ttl_s=600) as first:
            first.put_page(_doc(text="persisted"))
        with Cache(path, ttl_s=600) as second:
            cached = second.get_page("https://example.com/a")
        assert cached is not None
        assert cached.text == "persisted"

    def test_zero_ttl_disables_caching(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path / "eris.sqlite3", ttl_s=0)
        assert cache.enabled is False
        cache.put_page(_doc())
        assert cache.get_page("https://example.com/a") is None
        cache.close()

    def test_negative_ttl_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ttl_s"):
            Cache(tmp_path / "x.sqlite3", ttl_s=-1)

    def test_unopenable_path_raises_cache_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        with pytest.raises(CacheError, match="Could not open cache"):
            Cache(blocker / "sub" / "eris.sqlite3", ttl_s=60).put_page(_doc())

    def test_close_is_idempotent(self, timed_cache: Cache) -> None:
        timed_cache.put_page(_doc())
        timed_cache.close()
        timed_cache.close()


class TestNullCache:
    def test_reports_itself_disabled(self) -> None:
        assert NullCache().enabled is False

    def test_writes_are_discarded(self) -> None:
        cache = NullCache()
        cache.put_page(_doc())
        assert cache.get_page("https://example.com/a") is None

    def test_everything_is_a_miss(self) -> None:
        cache = NullCache()
        hits, misses = cache.get_pages(["https://e.com/1", "https://e.com/2"])
        assert hits == []
        assert len(misses) == 2

    def test_search_reads_miss(self) -> None:
        cache = NullCache()
        cache.put_search("q", 8, [SearchResult(title="T", url="https://e.com/a")])
        assert cache.get_search("q", 8) is None

    def test_maintenance_is_a_no_op(self) -> None:
        cache = NullCache()
        assert cache.clear() == 0
        assert cache.purge_expired() == 0
        assert cache.all_pages() == []
        assert cache.stats().pages == 0
        cache.close()
