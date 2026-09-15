"""Data structures passed between Eris stages.

These are frozen dataclasses rather than pydantic models: they are internal
values on a hot path (thousands of chunks per query), never parsed from
untrusted input, and benefit from ``slots`` and cheap hashing. Pydantic is
reserved for configuration, where validation and coercion earn their cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware current UTC time.

    Centralised so tests can monkeypatch a single symbol.
    """
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One hit from a search backend."""

    title: str
    url: str
    snippet: str = ""

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("SearchResult.url must not be empty")


@dataclass(frozen=True, slots=True)
class Document:
    """A fetched page reduced to clean text."""

    url: str
    title: str
    text: str
    fetched_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("Document.url must not be empty")

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A window of document text, the unit the ranker scores.

    ``index`` is the chunk's position within its source document, which makes
    ``(url, index)`` a stable identity across runs.
    """

    url: str
    title: str
    text: str
    index: int = 0
    score: float = 0.0

    @property
    def source_id(self) -> str:
        return f"{self.url}#{self.index}"

    def with_score(self, score: float) -> Chunk:
        """Return a copy carrying ``score`` (the dataclass is frozen)."""
        return Chunk(
            url=self.url,
            title=self.title,
            text=self.text,
            index=self.index,
            score=score,
        )


@dataclass(frozen=True, slots=True)
class Citation:
    """A numbered source the model was allowed to cite."""

    n: int
    url: str
    title: str

    def to_dict(self) -> dict[str, object]:
        return {"n": self.n, "url": self.url, "title": self.title}


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer plus the sources actually cited in its text."""

    text: str
    citations: tuple[Citation, ...] = ()
    invalid_citations: tuple[int, ...] = ()

    @property
    def cited_urls(self) -> tuple[str, ...]:
        return tuple(c.url for c in self.citations)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "invalid_citations": list(self.invalid_citations),
        }


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Wall-clock duration of one pipeline stage, in milliseconds."""

    name: str
    ms: float

    def __str__(self) -> str:
        return f"{self.name}={self.ms:.1f}ms"


@dataclass(frozen=True, slots=True)
class AskResult:
    """Everything one :meth:`eris.pipeline.Eris.ask` call produced.

    Bundling the answer with the retrieved chunks and per-stage timings keeps
    the pipeline observable without reaching into internals or re-running work.
    """

    question: str
    answer: Answer
    chunks: tuple[Chunk, ...] = ()
    timings: tuple[StageTiming, ...] = ()
    n_search_results: int = 0
    n_documents: int = 0
    from_cache: bool = False

    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)

    def timing_map(self) -> dict[str, float]:
        return {t.name: t.ms for t in self.timings}

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "sources": [
                {"url": c.url, "title": c.title, "score": round(c.score, 6)} for c in self.chunks
            ],
            "timings_ms": {t.name: round(t.ms, 3) for t in self.timings},
            "n_search_results": self.n_search_results,
            "n_documents": self.n_documents,
            "from_cache": self.from_cache,
        }
