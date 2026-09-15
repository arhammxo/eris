"""Offline evaluation harness.

Retrieval quality is easy to fool yourself about, so Eris ships a deterministic
harness that scores the pipeline without touching the network. A ``.jsonl`` file
supplies the questions, the fake search results, and the page bodies; the eval
runs the real chunker, ranker, MMR selector and citation validator against them.

Two metrics:

* **hit-rate@k** - did ``expected_url`` reach the context window? This measures
  the retrieval layer alone; if it drops, ranking regressed.
* **citation validity** - were all ``[n]`` markers resolvable? This measures the
  answer layer's guardrail; it stays at 100% unless validation breaks.

Both are reported per case, so a regression names the question that broke.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eris.cache import NullCache
from eris.config import Settings, load_settings
from eris.errors import ErisError
from eris.fetch import StaticFetcher
from eris.llm import EchoCitationClient, LLMClient
from eris.models import Document, SearchResult
from eris.pipeline import Eris
from eris.retrieval import Retriever
from eris.search import FakeSearch

log = logging.getLogger(__name__)


class EvalError(ErisError):
    """The eval file is missing, malformed, or empty."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation question with its offline fixtures."""

    question: str
    expected_url: str
    documents: tuple[Document, ...]
    results: tuple[SearchResult, ...]
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, object], *, line_no: int) -> EvalCase:
        """Parse one JSONL record.

        Schema::

            {
              "question": "...",
              "expected_url": "https://...",
              "sources": [
                {"url": "https://...", "title": "...",
                 "snippet": "...", "text": "page body"}
              ],
              "note": "optional"
            }

        ``sources[].text`` is the page body the fake fetcher will return; when
        absent the snippet is used.
        """
        question = str(payload.get("question", "")).strip()
        if not question:
            raise EvalError(f"line {line_no}: 'question' is required")

        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise EvalError(f"line {line_no}: 'sources' must be a non-empty list")

        documents: list[Document] = []
        results: list[SearchResult] = []
        for position, source in enumerate(raw_sources):
            if not isinstance(source, dict):
                raise EvalError(f"line {line_no}: sources[{position}] must be an object")
            url = str(source.get("url", "")).strip()
            if not url:
                raise EvalError(f"line {line_no}: sources[{position}].url is required")
            title = str(source.get("title", "")).strip() or url
            snippet = str(source.get("snippet", "")).strip()
            text = str(source.get("text", "")).strip() or snippet
            if not text:
                raise EvalError(f"line {line_no}: sources[{position}] needs 'text' or 'snippet'")
            documents.append(Document(url=url, title=title, text=text))
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        expected_url = str(payload.get("expected_url", "")).strip() or documents[0].url
        known = {d.url for d in documents}
        if expected_url not in known:
            raise EvalError(
                f"line {line_no}: expected_url {expected_url!r} is not among the sources"
            )
        return cls(
            question=question,
            expected_url=expected_url,
            documents=tuple(documents),
            results=tuple(results),
            note=str(payload.get("note", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Outcome of evaluating one case."""

    question: str
    expected_url: str
    hit: bool
    citations_valid: bool
    n_chunks: int
    n_sources: int
    n_invalid_citations: int
    retrieved_urls: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "expected_url": self.expected_url,
            "hit": self.hit,
            "citations_valid": self.citations_valid,
            "n_chunks": self.n_chunks,
            "n_sources": self.n_sources,
            "n_invalid_citations": self.n_invalid_citations,
            "retrieved_urls": list(self.retrieved_urls),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate metrics across every case."""

    results: tuple[CaseResult, ...] = field(default=())
    top_k: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def valid_citations(self) -> int:
        return sum(1 for r in self.results if r.citations_valid)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def citation_validity(self) -> float:
        return self.valid_citations / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "total": self.total,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "valid_citations": self.valid_citations,
            "citation_validity": round(self.citation_validity, 4),
            "errors": self.errors,
            "cases": [r.to_dict() for r in self.results],
        }

    def to_table(self) -> str:
        """Render a fixed-width report suitable for a terminal or README."""
        if not self.results:
            return "No eval cases were run."

        width = min(max((len(r.question) for r in self.results), default=8), 58)
        header = f"{'#':>3}  {'question':<{width}}  {'hit@k':^6}  {'cites':^6}  {'chunks':>6}"
        rule = "-" * len(header)
        lines = [header, rule]
        for position, result in enumerate(self.results, start=1):
            question = result.question
            if len(question) > width:
                question = question[: width - 1] + "…"
            hit = "PASS" if result.hit else "FAIL"
            cites = "ok" if result.citations_valid else f"bad:{result.n_invalid_citations}"
            lines.append(
                f"{position:>3}  {question:<{width}}  {hit:^6}  {cites:^6}  {result.n_chunks:>6}"
            )
        lines.append(rule)
        lines.append(
            f"cases={self.total}  top_k={self.top_k}  "
            f"hit_rate@{self.top_k}={self.hit_rate:.1%}  "
            f"citation_validity={self.citation_validity:.1%}"
        )
        if self.errors:
            lines.append(f"errors={self.errors}")
        return "\n".join(lines)


def load_cases(path: Path | str) -> list[EvalCase]:
    """Read eval cases from a JSONL file, skipping blanks and ``#`` comments."""
    file_path = Path(path)
    if not file_path.exists():
        raise EvalError(f"Eval file not found: {file_path}")
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"Could not read {file_path}: {exc}") from exc

    cases: list[EvalCase] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalError(f"line {line_no}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise EvalError(f"line {line_no}: each line must be a JSON object")
        cases.append(EvalCase.from_dict(payload, line_no=line_no))

    if not cases:
        raise EvalError(f"No eval cases found in {file_path}")
    return cases


def evaluate_case(
    case: EvalCase,
    *,
    settings: Settings,
    llm_client: LLMClient | None = None,
    top_k: int | None = None,
) -> CaseResult:
    """Run one case through the real pipeline with fake I/O."""
    k = top_k or settings.top_k
    search = FakeSearch({case.question: case.results}, default=case.results)
    fetcher = StaticFetcher(case.documents)
    retriever = Retriever(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=k,
        bm25_weight=settings.bm25_weight,
        mmr_lambda=settings.mmr_lambda,
        max_chunks_per_url=settings.max_chunks_per_url,
    )
    eris = Eris(
        settings,
        search_backend=search,
        fetcher=fetcher,
        cache=NullCache(),
        retriever=retriever,
        llm_client=llm_client or EchoCitationClient(),
    )

    try:
        result = eris.ask(case.question, top_k=k)
    except (ErisError, ValueError) as exc:
        return CaseResult(
            question=case.question,
            expected_url=case.expected_url,
            hit=False,
            citations_valid=False,
            n_chunks=0,
            n_sources=0,
            n_invalid_citations=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    retrieved = tuple(dict.fromkeys(chunk.url for chunk in result.chunks))
    return CaseResult(
        question=case.question,
        expected_url=case.expected_url,
        hit=case.expected_url in retrieved,
        citations_valid=not result.answer.invalid_citations,
        n_chunks=len(result.chunks),
        n_sources=len(retrieved),
        n_invalid_citations=len(result.answer.invalid_citations),
        retrieved_urls=retrieved,
    )


def run_eval(
    cases: Iterable[EvalCase] | Sequence[EvalCase],
    *,
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    top_k: int | None = None,
) -> EvalReport:
    """Evaluate every case and aggregate the metrics."""
    active = settings or load_settings(search_backend="fake", cache_ttl_s=0)
    k = top_k or active.top_k
    results = [
        evaluate_case(case, settings=active, llm_client=llm_client, top_k=k) for case in cases
    ]
    return EvalReport(results=tuple(results), top_k=k)


def run_eval_file(
    path: Path | str,
    *,
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    top_k: int | None = None,
) -> EvalReport:
    """Load ``path`` and evaluate every case in it."""
    return run_eval(load_cases(path), settings=settings, llm_client=llm_client, top_k=top_k)
