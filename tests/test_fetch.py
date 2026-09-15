"""HTML extraction and HTTP fetching.

No test here opens a socket: ``httpx.MockTransport`` supplies fabricated wire
responses while the real request/response plumbing still runs.
"""

from __future__ import annotations

import httpx
import pytest

from conftest import mock_fetcher
from eris.errors import FetchError
from eris.fetch import (
    PageFetcher,
    StaticFetcher,
    clean_text,
    extract_title,
    html_to_text,
)
from eris.models import Document, SearchResult


def _ok(body: str, content_type: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": content_type})


class TestCleanText:
    def test_collapses_runs_of_spaces(self) -> None:
        assert clean_text("a     b\t\tc") == "a b c"

    def test_collapses_blank_line_runs(self) -> None:
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_normalises_non_breaking_spaces(self) -> None:
        assert clean_text("a\u00a0b") == "a b"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert clean_text("\n\n  hello  \n\n") == "hello"

    @pytest.mark.parametrize(
        "line",
        [
            "Accept all cookies",
            "accept cookies",
            "Cookie Policy",
            "Skip to main content",
            "Subscribe now",
            "Advertisement",
            "All rights reserved",
            "© 2026 Example Corp",
        ],
    )
    def test_drops_boilerplate_lines(self, line: str) -> None:
        assert clean_text(f"Real content here.\n{line}\nMore real content.") == (
            "Real content here.\nMore real content."
        )

    def test_keeps_content_lines(self) -> None:
        assert "payload of up to 10 MiB" in clean_text("The payload of up to 10 MiB is allowed.")

    @pytest.mark.parametrize("noise", ["|", "•", "---", ">>", "  ·  "])
    def test_drops_lines_with_no_alphanumeric_content(self, noise: str) -> None:
        assert clean_text(f"Real line.\n{noise}\nAnother line.") == "Real line.\nAnother line."

    @pytest.mark.parametrize("short", ["5", "a", "£9"])
    def test_keeps_short_but_meaningful_lines(self, short: str) -> None:
        """Filtering on content, not length, preserves lines like a bare number."""
        assert short.strip() in clean_text(f"Total:\n{short}\nend")

    def test_empty_input_is_empty(self) -> None:
        assert clean_text("") == ""


class TestExtractTitle:
    def test_extracts_and_normalises(self, html_page: str) -> None:
        assert extract_title(html_page) == "Widget Reference"

    def test_missing_title_is_empty(self) -> None:
        assert extract_title("<html><body>no title</body></html>") == ""

    def test_unescapes_entities(self) -> None:
        assert extract_title("<title>A &amp; B</title>") == "A & B"


class TestHtmlToText:
    def test_keeps_article_content(self, html_page: str) -> None:
        text = html_to_text(html_page)
        assert "payload of up to 10 MiB per request" in text
        assert "multipart endpoint" in text

    @pytest.mark.parametrize(
        "noise",
        ["window.tracking", "color: red", "Enable JavaScript", "Accept all cookies"],
    )
    def test_strips_scripts_styles_and_boilerplate(self, html_page: str, noise: str) -> None:
        assert noise not in html_to_text(html_page)

    def test_drops_navigation_and_footer(self, html_page: str) -> None:
        text = html_to_text(html_page)
        assert "All rights reserved" not in text
        assert "Subscribe now" not in text

    def test_produces_no_angle_brackets(self, html_page: str) -> None:
        assert "<" not in html_to_text(html_page)

    def test_unescapes_entities(self) -> None:
        text = html_to_text("<html><body><p>Ampersand &amp; more &lt;tag&gt;</p></body></html>")
        assert "Ampersand & more" in text

    def test_block_elements_become_line_breaks(self) -> None:
        text = html_to_text("<html><body><p>First para</p><p>Second para</p></body></html>")
        assert "First para" in text
        assert "Second para" in text

    def test_empty_input_is_empty(self) -> None:
        assert html_to_text("") == ""
        assert html_to_text("   ") == ""

    def test_malformed_html_still_yields_text(self) -> None:
        assert "content" in html_to_text("<html><body><p>content<div><span>")

    def test_plain_text_passes_through(self) -> None:
        assert "just words here" in html_to_text("just words here")


class TestFetchSuccess:
    def test_returns_clean_document(self, html_page: str) -> None:
        fetcher = mock_fetcher(lambda request: _ok(html_page))
        document = fetcher.fetch("https://example.com/doc")
        assert document.url == "https://example.com/doc"
        assert document.title == "Widget Reference"
        assert "10 MiB per request" in document.text
        assert "window.tracking" not in document.text

    def test_sets_a_identifying_user_agent(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return _ok("<html><body><p>hello there</p></body></html>")

        mock_fetcher(handler).fetch("https://example.com/a")
        assert "eris" in seen["user-agent"].lower()
        assert "github.com/arhammxo/eris" in seen["user-agent"]

    def test_requests_text_content(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return _ok("<html><body><p>hello there</p></body></html>")

        mock_fetcher(handler).fetch("https://example.com/a")
        assert "text/html" in seen["accept"]

    def test_uses_title_hint_when_page_has_none(self) -> None:
        fetcher = mock_fetcher(lambda r: _ok("<html><body><p>body text here</p></body></html>"))
        document = fetcher.fetch("https://example.com/a", title_hint="From search")
        assert document.title == "From search"

    def test_falls_back_to_url_without_title_or_hint(self) -> None:
        fetcher = mock_fetcher(lambda r: _ok("<html><body><p>body text here</p></body></html>"))
        assert fetcher.fetch("https://example.com/a").title == "https://example.com/a"

    def test_fetched_at_is_timezone_aware(self) -> None:
        fetcher = mock_fetcher(lambda r: _ok("<html><body><p>body text here</p></body></html>"))
        assert fetcher.fetch("https://example.com/a").fetched_at.tzinfo is not None

    def test_follows_redirects(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/old":
                return httpx.Response(301, headers={"location": "https://example.com/new"})
            return _ok("<html><body><p>redirected content here</p></body></html>")

        assert "redirected content" in mock_fetcher(handler).fetch("https://example.com/old").text

    def test_plain_text_response_is_accepted(self) -> None:
        fetcher = mock_fetcher(lambda r: _ok("Plain body content.", "text/plain"))
        assert "Plain body content." in fetcher.fetch("https://example.com/a.txt").text

    def test_oversized_body_is_truncated_not_rejected(self) -> None:
        body = "<html><body><p>" + ("filler content " * 5_000) + "</p></body></html>"
        fetcher = mock_fetcher(lambda r: _ok(body), max_bytes=2_000)
        assert len(fetcher.fetch("https://example.com/big").text) < 5_000

    def test_missing_content_type_is_tolerated(self) -> None:
        fetcher = mock_fetcher(
            lambda r: httpx.Response(200, text="<html><body><p>content here</p></body></html>")
        )
        assert "content here" in fetcher.fetch("https://example.com/a").text


class TestFetchFailure:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 500, 502, 503])
    def test_error_statuses_raise(self, status: int) -> None:
        fetcher = mock_fetcher(lambda r: httpx.Response(status, text="nope"))
        with pytest.raises(FetchError, match=f"HTTP {status}"):
            fetcher.fetch("https://example.com/a")

    def test_transport_error_raises_fetch_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with pytest.raises(FetchError, match="Request failed"):
            mock_fetcher(handler).fetch("https://example.com/a")

    def test_timeout_raises_fetch_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow")

        with pytest.raises(FetchError, match="Request failed"):
            mock_fetcher(handler).fetch("https://example.com/a")

    @pytest.mark.parametrize("content_type", ["image/png", "application/pdf", "video/mp4"])
    def test_binary_content_is_skipped(self, content_type: str) -> None:
        fetcher = mock_fetcher(lambda r: _ok("binary-ish", content_type))
        with pytest.raises(FetchError, match="non-text content"):
            fetcher.fetch("https://example.com/file")

    def test_empty_body_raises(self) -> None:
        fetcher = mock_fetcher(lambda r: _ok(""))
        with pytest.raises(FetchError, match="No extractable text"):
            fetcher.fetch("https://example.com/empty")

    def test_content_free_html_raises(self) -> None:
        fetcher = mock_fetcher(lambda r: _ok("<html><head></head><body></body></html>"))
        with pytest.raises(FetchError, match="No extractable text"):
            fetcher.fetch("https://example.com/blank")

    @pytest.mark.parametrize("url", ["", "   "])
    def test_blank_url_raises(self, url: str) -> None:
        fetcher = mock_fetcher(lambda r: _ok("x"))
        with pytest.raises(FetchError, match="empty URL"):
            fetcher.fetch(url)

    @pytest.mark.parametrize("url", ["ftp://example.com/a", "file:///etc/passwd", "gopher://x"])
    def test_non_http_schemes_are_refused(self, url: str) -> None:
        """Guards against a search result steering the fetcher at a local file."""
        fetcher = mock_fetcher(lambda r: _ok("x"))
        with pytest.raises(FetchError, match="Unsupported URL scheme"):
            fetcher.fetch(url)

    def test_undecodable_bytes_do_not_crash(self) -> None:
        fetcher = mock_fetcher(
            lambda r: httpx.Response(
                200,
                content=b"<html><body><p>caf\xe9 content here</p></body></html>",
                headers={"content-type": "text/html"},
            )
        )
        assert fetcher.fetch("https://example.com/a").text


class TestFetchMany:
    def test_fetches_every_result(self, search_results: list[SearchResult]) -> None:
        fetcher = mock_fetcher(lambda r: _ok(f"<html><body><p>content of {r.url.path}</p></body>"))
        assert len(fetcher.fetch_many(search_results)) == len(search_results)

    def test_preserves_search_ranking(self, search_results: list[SearchResult]) -> None:
        fetcher = mock_fetcher(lambda r: _ok(f"<html><body><p>content of {r.url.path}</p></body>"))
        fetched = fetcher.fetch_many(search_results)
        assert [d.url for d in fetched] == [r.url for r in search_results]

    def test_one_failure_does_not_sink_the_batch(self, search_results: list[SearchResult]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "rust" in str(request.url):
                return httpx.Response(500, text="boom")
            return _ok("<html><body><p>content that is long enough</p></body></html>")

        fetched = mock_fetcher(handler).fetch_many(search_results)
        assert len(fetched) == len(search_results) - 1
        assert all("rust" not in d.url for d in fetched)

    def test_all_failures_yield_empty_list(self, search_results: list[SearchResult]) -> None:
        fetcher = mock_fetcher(lambda r: httpx.Response(503, text="down"))
        assert fetcher.fetch_many(search_results) == []

    def test_empty_input_yields_empty_list(self) -> None:
        assert mock_fetcher(lambda r: _ok("x")).fetch_many([]) == []

    def test_concurrency_is_capped_at_the_batch_size(self) -> None:
        fetcher = mock_fetcher(
            lambda r: _ok("<html><body><p>content long enough here</p></body></html>"),
            concurrency=32,
        )
        results = [SearchResult(title="t", url="https://example.com/only")]
        assert len(fetcher.fetch_many(results)) == 1


class TestFetcherConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"timeout_s": 0}, "timeout_s"),
            ({"timeout_s": -1.0}, "timeout_s"),
            ({"max_bytes": 0}, "max_bytes"),
            ({"concurrency": 0}, "concurrency"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PageFetcher(**kwargs)  # type: ignore[arg-type]

    def test_caller_owned_client_is_not_closed(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda r: _ok("<p>hi there</p>")))
        fetcher = PageFetcher(client=client)
        fetcher.close()
        assert client.is_closed is False
        client.close()

    def test_context_manager_closes_its_own_client(self) -> None:
        with PageFetcher(timeout_s=1.0) as fetcher:
            assert fetcher._client is not None
        assert fetcher._client is None


class TestStaticFetcher:
    def test_returns_registered_documents(self, documents: list[Document]) -> None:
        assert StaticFetcher(documents).fetch(documents[0].url).text == documents[0].text

    def test_unknown_url_raises(self) -> None:
        with pytest.raises(FetchError, match="No fixture"):
            StaticFetcher([]).fetch("https://unknown.example")

    def test_fetch_many_skips_unknown_urls(self, documents: list[Document]) -> None:
        fetcher = StaticFetcher(documents[:1])
        results = [SearchResult(title=d.title, url=d.url) for d in documents]
        assert len(fetcher.fetch_many(results)) == 1

    def test_records_requested_urls(self, documents: list[Document]) -> None:
        fetcher = StaticFetcher(documents)
        fetcher.fetch_many([SearchResult(title=d.title, url=d.url) for d in documents])
        assert fetcher.calls == [d.url for d in documents]

    def test_documents_can_be_added_later(self) -> None:
        fetcher = StaticFetcher()
        fetcher.add(Document(url="https://x.example", title="X", text="body"))
        assert fetcher.fetch("https://x.example").title == "X"
