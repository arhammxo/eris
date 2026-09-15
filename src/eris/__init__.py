"""Eris - an LLM augmented with real-time web search and a local retrieval layer.

Typical use::

    from eris import Eris

    with Eris() as eris:
        result = eris.ask("What changed in the latest Python release?")
        print(result.answer.text)
        for citation in result.answer.citations:
            print(citation.n, citation.url)

Only lightweight names are imported eagerly. Anything that would pull in an
optional dependency (``anthropic``, ``ddgs``, ``scikit-learn``, ``trafilatura``)
is imported inside the module that needs it, so ``import eris`` stays cheap and
works with the base install alone.
"""

from __future__ import annotations

from eris.answer import AnswerBuilder, build_prompt, validate_citations
from eris.cache import Cache, CacheStats, NullCache, normalize_url
from eris.config import Settings, load_settings
from eris.errors import (
    CacheError,
    ConfigError,
    DependencyMissingError,
    ErisError,
    FetchError,
    LLMError,
    RetrievalError,
    SearchError,
)
from eris.fetch import PageFetcher, StaticFetcher, html_to_text
from eris.llm import AnthropicClient, FakeClient, LLMClient
from eris.models import (
    Answer,
    AskResult,
    Chunk,
    Citation,
    Document,
    SearchResult,
    StageTiming,
)
from eris.pipeline import Eris
from eris.retrieval import BM25Scorer, HybridScorer, Retriever, TfidfCosineScorer, mmr_select
from eris.search import DuckDuckGoSearch, FakeSearch, SearchBackend

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "AnswerBuilder",
    "AnthropicClient",
    "AskResult",
    "BM25Scorer",
    "Cache",
    "CacheError",
    "CacheStats",
    "Chunk",
    "Citation",
    "ConfigError",
    "DependencyMissingError",
    "Document",
    "DuckDuckGoSearch",
    "Eris",
    "ErisError",
    "FakeClient",
    "FakeSearch",
    "FetchError",
    "HybridScorer",
    "LLMClient",
    "LLMError",
    "NullCache",
    "PageFetcher",
    "RetrievalError",
    "Retriever",
    "SearchBackend",
    "SearchError",
    "SearchResult",
    "Settings",
    "StageTiming",
    "StaticFetcher",
    "TfidfCosineScorer",
    "__version__",
    "build_prompt",
    "html_to_text",
    "load_settings",
    "mmr_select",
    "normalize_url",
    "validate_citations",
]
