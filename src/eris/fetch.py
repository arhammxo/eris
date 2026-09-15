"""Page fetching and HTML-to-text extraction.

Raw HTML is mostly navigation, cookie banners and footers. Feeding it to a
model wastes context and dilutes the ranker's signal, so every page is reduced
to article text here, before it reaches retrieval.

Extraction tries ``trafilatura`` first (best boilerplate removal), then
BeautifulSoup, then a regex stripper. The chain means Eris degrades in quality
rather than breaking when optional dependencies are absent.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import httpx

from eris.errors import FetchError
from eris.models import Document, SearchResult, utcnow

log = logging.getLogger(__name__)

# Elements whose contents are never article text.
_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "iframe",
)

_DROP_BLOCK_RE = re.compile(
    r"<(?P<tag>" + "|".join(_DROP_TAGS) + r")\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SELF_CLOSING_DROP_RE = re.compile(
    r"<(?:" + "|".join(_DROP_TAGS) + r")\b[^>]*/?>",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLOCK_END_RE = re.compile(
    r"</(?:p|div|section|article|li|ul|ol|tr|table|h[1-6]|blockquote|pre)\s*>"
    r"|<br\s*/?>|<hr\s*/?>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RUN_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Boilerplate lines that survive tag stripping.
_BOILERPLATE_RE = re.compile(
    r"^(?:accept(?:\s+all)?\s+cookies?|cookie\s+(?:policy|settings|preferences)|"
    r"skip\s+to\s+(?:main\s+)?content|subscribe\s+now|sign\s+up|log\s*in|share\s+this|"
    r"advertisement|sponsored|all\s+rights\s+reserved|©.*)$",
    re.IGNORECASE,
)

# A line with no letters or digits is an artefact of stripping markup (stray
# bullets, pipes, separators), never article text. Filtering on content rather
# than on length keeps short but meaningful lines such as "5" or "a".
_NO_ALNUM_RE = re.compile(r"^[^0-9a-zA-Z]+$")


@runtime_checkable
class Fetcher(Protocol):
    """Anything that turns search results into text documents.

    Injecting this protocol is how the pipeline tests avoid the network.
    """

    def fetch_many(self, results: Sequence[SearchResult]) -> list[Document]:
        """Fetch every result, skipping failures."""
        ...


def clean_text(raw: str) -> str:
    """Collapse whitespace and drop boilerplate lines from extracted text."""
    text = _WS_RUN_RE.sub(" ", raw.replace("\u00a0", " "))
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _NO_ALNUM_RE.match(stripped) or _BOILERPLATE_RE.match(stripped):
            continue
        kept.append(stripped)
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(kept)).strip()


def extract_title(html_text: str) -> str:
    """Best-effort ``<title>`` extraction."""
    match = _TITLE_RE.search(html_text)
    if not match:
        return ""
    return clean_text(html.unescape(_TAG_RE.sub(" ", match.group(1)))).strip()


def _extract_regex(html_text: str) -> str:
    """Dependency-free extractor: drop non-content tags, keep block breaks."""
    body = _COMMENT_RE.sub(" ", html_text)
    body = _DROP_BLOCK_RE.sub(" ", body)
    body = _SELF_CLOSING_DROP_RE.sub(" ", body)
    body = _BLOCK_END_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    return clean_text(html.unescape(body))


def _extract_bs4(html_text: str) -> str | None:
    """BeautifulSoup extractor. Returns ``None`` when bs4 is unavailable."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup.find_all(_DROP_TAGS):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        return clean_text(root.get_text(separator="\n"))
    except Exception as exc:  # malformed markup should not kill the fetch
        log.debug("bs4 extraction failed: %s", exc)
        return None


def _extract_trafilatura(html_text: str) -> str | None:
    """Trafilatura extractor. Returns ``None`` when unavailable or empty."""
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        extracted = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
    except Exception as exc:
        log.debug("trafilatura extraction failed: %s", exc)
        return None
    return clean_text(extracted) if extracted else None


def html_to_text(html_text: str) -> str:
    """Reduce HTML to article text using the best available extractor.

    Tries trafilatura, then BeautifulSoup, then regex; a later extractor is
    used when an earlier one yields nothing useful.
    """
    if not html_text or not html_text.strip():
        return ""
    for extractor in (_extract_trafilatura, _extract_bs4):
        text = extractor(html_text)
        if text:
            return text
    return _extract_regex(html_text)


class PageFetcher:
    """Fetches pages over HTTP and returns clean-text :class:`Document` objects.

    A caller-supplied ``client`` is honoured, which is how tests inject an
    ``httpx.MockTransport`` and stay offline.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        max_bytes: int = 2_000_000,
        concurrency: int = 8,
        user_agent: str = "eris/0.1 (+https://github.com/arhammxo/eris)",
        client: httpx.Client | None = None,
        max_redirects: int = 5,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.concurrency = concurrency
        self.user_agent = user_agent
        self.max_redirects = max_redirects
        self._external_client = client
        self._client: httpx.Client | None = client

    # --- lifecycle ---------------------------------------------------------
    @property
    def headers(self) -> dict[str, str]:
        """Identify the crawler and prefer text so operators can filter us."""
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout_s,
                follow_redirects=True,
                max_redirects=self.max_redirects,
                headers=self.headers,
            )
        return self._client

    def close(self) -> None:
        """Close the client, unless the caller owns it."""
        if self._client is not None and self._external_client is None:
            self._client.close()
            self._client = None

    def __enter__(self) -> PageFetcher:
        self._ensure_client()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- single fetch ------------------------------------------------------
    def fetch(self, url: str, *, title_hint: str = "") -> Document:
        """Fetch one URL and return its clean text.

        Raises:
            FetchError: on transport failure, non-2xx status, non-text content,
                or an oversized body.
        """
        if not url or not url.strip():
            raise FetchError("Refusing to fetch an empty URL")
        url = url.strip()
        if not url.lower().startswith(("http://", "https://")):
            raise FetchError(f"Unsupported URL scheme for {url!r}; expected http(s)")

        client = self._ensure_client()
        try:
            response = client.get(url, headers=self.headers)
        except httpx.HTTPError as exc:
            raise FetchError(f"Request failed for {url}: {exc}") from exc

        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code} for {url}")

        content_type = response.headers.get("content-type", "")
        if content_type and not self._is_texty(content_type):
            raise FetchError(f"Skipping non-text content ({content_type}) at {url}")

        body = response.content or b""
        if len(body) > self.max_bytes:
            log.debug("truncating %s from %d to %d bytes", url, len(body), self.max_bytes)
            body = body[: self.max_bytes]

        raw = self._decode(body, response)
        text = html_to_text(raw) if self._looks_like_html(raw, content_type) else clean_text(raw)
        if not text:
            raise FetchError(f"No extractable text at {url}")

        title = extract_title(raw) or title_hint.strip() or url
        return Document(url=url, title=title, text=text, fetched_at=utcnow())

    @staticmethod
    def _is_texty(content_type: str) -> bool:
        ct = content_type.lower()
        return ct.startswith("text/") or any(
            token in ct for token in ("html", "xml", "json", "javascript")
        )

    @staticmethod
    def _looks_like_html(raw: str, content_type: str) -> bool:
        if "html" in content_type.lower() or "xml" in content_type.lower():
            return True
        head = raw[:2048].lower()
        return "<html" in head or "<body" in head or "<!doctype html" in head or "<p" in head

    @staticmethod
    def _decode(body: bytes, response: httpx.Response) -> str:
        """Decode bytes using the response encoding, falling back to UTF-8."""
        if not body:
            return ""
        for encoding in (response.encoding, "utf-8"):
            if not encoding:
                continue
            try:
                return body.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return body.decode("utf-8", errors="replace")

    # --- batch fetch -------------------------------------------------------
    def fetch_many(self, results: Sequence[SearchResult]) -> list[Document]:
        """Fetch all ``results`` in parallel, preserving input order.

        Individual failures are logged and skipped: one dead link must not sink
        the query.
        """
        if not results:
            return []

        self._ensure_client()
        workers = min(self.concurrency, len(results))
        documents: list[Document | None] = [None] * len(results)

        def _task(pair: tuple[int, SearchResult]) -> None:
            position, result = pair
            try:
                documents[position] = self.fetch(result.url, title_hint=result.title)
            except FetchError as exc:
                log.debug("skipping %s: %s", result.url, exc)
            except Exception as exc:  # never let one page kill the batch
                log.warning("unexpected error fetching %s: %s", result.url, exc)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eris-fetch") as pool:
            list(pool.map(_task, enumerate(results)))

        fetched = [doc for doc in documents if doc is not None]
        log.debug("fetched %d/%d pages", len(fetched), len(results))
        return fetched


class StaticFetcher:
    """Fetcher backed by a URL-to-document mapping.

    Used by tests and the offline eval harness in place of the network.
    """

    def __init__(self, documents: Sequence[Document] | None = None) -> None:
        self._by_url = {doc.url: doc for doc in (documents or [])}
        self.calls: list[str] = []

    def add(self, document: Document) -> None:
        self._by_url[document.url] = document

    def fetch(self, url: str, *, title_hint: str = "") -> Document:
        self.calls.append(url)
        try:
            return self._by_url[url]
        except KeyError as exc:
            raise FetchError(f"No fixture registered for {url}") from exc

    def fetch_many(self, results: Sequence[SearchResult]) -> list[Document]:
        documents: list[Document] = []
        for result in results:
            try:
                documents.append(self.fetch(result.url, title_hint=result.title))
            except FetchError:
                continue
        return documents
