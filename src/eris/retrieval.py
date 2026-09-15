"""The local retrieval layer: chunk, rank, diversify.

Why this exists. Handing a model a few raw pages wastes most of the context on
navigation and unrelated sections, and buries the passage that actually answers
the question. Ranking locally is cheap (milliseconds, no API call) and lets Eris
spend the context budget on the ~6 passages most likely to matter.

Three ideas, in order:

1. **Okapi BM25** (implemented here in numpy, no heavy dependency) for lexical
   relevance. It beats plain TF-IDF on short queries because term saturation
   stops one repeated keyword from dominating, and length normalisation stops
   long chunks from winning on volume alone.
2. **TF-IDF cosine** as an optional second opinion via scikit-learn, combined
   with BM25 as a weighted hybrid. Each is normalised to ``[0, 1]`` first, so
   the weight means what it says.
3. **MMR** (Maximal Marginal Relevance) to pick the final top-k. Pure relevance
   tends to return five near-identical paragraphs from the same page; MMR
   trades a little relevance for coverage, and a per-URL cap guarantees the
   model sees more than one source.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from eris.errors import RetrievalError
from eris.models import Chunk, Document

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Removing these lifts precision on natural-language questions: they appear in
# nearly every chunk, so they carry almost no discriminative signal.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "aren",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "don",
        "done",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "it",
        "its",
        "may",
        "me",
        "might",
        "must",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "s",
        "she",
        "should",
        "t",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "us",
        "was",
        "wasn",
        "were",
        "weren",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, split on word characters, and optionally drop stopwords."""
    tokens = _TOKEN_RE.findall((text or "").lower())
    if remove_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of about ``chunk_size`` chars.

    Windows prefer to end at a sentence or newline boundary near the target
    size, so passages stay readable and a fact is less likely to be split in
    half. ``overlap`` carries context across the seam.
    """
    if chunk_size <= 0:
        raise RetrievalError("chunk_size must be positive")
    if overlap < 0:
        raise RetrievalError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise RetrievalError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    # Look for a boundary in the last 25% of the window.
    search_span = max(1, chunk_size // 4)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = cleaned[start:end]
            cut = -1
            for match in _SENTENCE_END_RE.finditer(window):
                if match.start() >= chunk_size - search_span:
                    cut = match.start() + 1
                    break
            if cut == -1:
                newline = cleaned.rfind("\n", end - search_span, end)
                if newline > start:
                    cut = newline - start
            if cut == -1:
                space = cleaned.rfind(" ", end - search_span, end)
                if space > start:
                    cut = space - start
            if cut > 0:
                end = start + cut

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        # Always advance, even if the boundary landed inside the overlap.
        start = max(end - overlap, start + 1)

    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    overlap: int,
    min_chars: int = 40,
) -> list[Chunk]:
    """Turn documents into :class:`Chunk` objects with stable source ids.

    Chunks shorter than ``min_chars`` are dropped: a trailing fragment of a few
    words is noise in the ranker and useless as context.
    """
    chunks: list[Chunk] = []
    for document in documents:
        pieces = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        index = 0
        for piece in pieces:
            if len(piece) < min_chars:
                continue
            chunks.append(Chunk(url=document.url, title=document.title, text=piece, index=index))
            index += 1
    log.debug("chunked %d documents into %d chunks", len(list(documents or [])), len(chunks))
    return chunks


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max scale to ``[0, 1]``; all-equal input maps to zeros.

    Needed before blending BM25 with cosine: the two live on different scales,
    so a weighted sum of raw values would be meaningless.
    """
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


@runtime_checkable
class Scorer(Protocol):
    """Scores a corpus of texts against a query.

    Returns one non-negative float per document, higher meaning more relevant.
    """

    name: str

    def score(self, query: str, corpus: Sequence[str]) -> np.ndarray: ...


class BM25Scorer:
    """Okapi BM25, implemented directly over numpy.

    ``score(q, d) = Σ_t IDF(t) · (f(t,d)·(k1+1)) / (f(t,d) + k1·(1-b + b·|d|/avgdl))``

    ``k1`` controls term-frequency saturation (the 4th occurrence of a word adds
    much less than the 2nd); ``b`` controls length normalisation. The defaults
    (1.5, 0.75) are the standard values from the TREC literature.

    IDF uses the Robertson-Sparck-Jones form with the ``+1`` smoothing that
    keeps weights non-negative for terms appearing in more than half the corpus.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, *, remove_stopwords: bool = True) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

    def score(self, query: str, corpus: Sequence[str]) -> np.ndarray:
        if not corpus:
            return np.zeros(0, dtype=float)

        query_terms = tokenize(query, remove_stopwords=self.remove_stopwords)
        n_docs = len(corpus)
        scores = np.zeros(n_docs, dtype=float)
        if not query_terms:
            return scores

        doc_tokens = [tokenize(text, remove_stopwords=self.remove_stopwords) for text in corpus]
        doc_lengths = np.array([len(tokens) for tokens in doc_tokens], dtype=float)
        avg_length = float(doc_lengths.mean()) if doc_lengths.size else 0.0
        if avg_length <= 0.0:
            return scores

        term_freqs = [Counter(tokens) for tokens in doc_tokens]
        # Length-normalisation denominator term, shared by every query term.
        norm = self.k1 * (1.0 - self.b + self.b * (doc_lengths / avg_length))

        for term in set(query_terms):
            freqs = np.array([tf.get(term, 0) for tf in term_freqs], dtype=float)
            doc_freq = float(np.count_nonzero(freqs))
            if doc_freq == 0.0:
                continue
            idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            scores += idf * (freqs * (self.k1 + 1.0)) / (freqs + norm)

        # Repeated query terms should count once each, which set() already does;
        # clamp tiny negatives from floating point.
        return np.maximum(scores, 0.0)


class TfidfCosineScorer:
    """TF-IDF cosine similarity via scikit-learn.

    Complements BM25: sublinear TF plus L2 normalisation reacts differently to
    term overlap, so the hybrid is more robust than either alone. Optional -
    :meth:`available` reports whether scikit-learn is importable, and the
    scorer degrades to zeros rather than failing the query.
    """

    name = "tfidf"

    def __init__(self, *, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df

    @staticmethod
    def available() -> bool:
        try:
            import sklearn.feature_extraction.text  # noqa: F401
        except ImportError:
            return False
        return True

    def score(self, query: str, corpus: Sequence[str]) -> np.ndarray:
        if not corpus:
            return np.zeros(0, dtype=float)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            log.debug("scikit-learn unavailable; tfidf scorer returns zeros")
            return np.zeros(len(corpus), dtype=float)

        if not (query or "").strip():
            return np.zeros(len(corpus), dtype=float)

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform([*corpus, query])
        except ValueError as exc:
            # Raised when the vocabulary is empty (e.g. all stopwords).
            log.debug("tfidf vectorisation failed: %s", exc)
            return np.zeros(len(corpus), dtype=float)

        docs = matrix[:-1]
        query_vec = matrix[-1]
        # Rows are L2-normalised, so the dot product is the cosine.
        similarities = (docs @ query_vec.T).toarray().ravel()
        return np.maximum(np.nan_to_num(similarities, nan=0.0), 0.0)


class HybridScorer:
    """Weighted blend of two scorers after independent min-max normalisation.

    ``weight`` is the share given to ``primary``. At 1.0 the secondary scorer is
    skipped entirely, so the default BM25-only path costs nothing extra.
    """

    name = "hybrid"

    def __init__(
        self,
        primary: Scorer,
        secondary: Scorer | None = None,
        *,
        weight: float = 0.65,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.primary = primary
        self.secondary = secondary
        self.weight = weight

    def score(self, query: str, corpus: Sequence[str]) -> np.ndarray:
        if not corpus:
            return np.zeros(0, dtype=float)
        primary = _normalize_scores(np.asarray(self.primary.score(query, corpus), dtype=float))
        if self.secondary is None or self.weight >= 1.0:
            return primary
        secondary = _normalize_scores(np.asarray(self.secondary.score(query, corpus), dtype=float))
        if secondary.shape != primary.shape:
            raise RetrievalError("scorer outputs have mismatched shapes")
        if self.weight <= 0.0:
            return secondary
        return self.weight * primary + (1.0 - self.weight) * secondary


def _term_sets(texts: Sequence[str]) -> list[set[str]]:
    return [set(tokenize(text)) for text in texts]


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when both are empty.

    Used as the redundancy measure in MMR because it is symmetric, bounded and
    needs no vector model.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / len(a | b)


def mmr_select(
    chunks: Sequence[Chunk],
    scores: Sequence[float],
    *,
    top_k: int,
    lambda_: float = 0.7,
    max_per_url: int | None = 2,
) -> list[Chunk]:
    """Greedily select ``top_k`` chunks balancing relevance against redundancy.

    Each step picks ``argmax(lambda * relevance - (1 - lambda) * max_similarity_to_selected)``.
    A per-URL cap is applied on top: it is a hard guarantee of source diversity,
    where MMR alone only makes redundancy expensive. If the cap leaves fewer
    than ``top_k`` candidates, it is relaxed so the context is still filled.
    """
    if len(chunks) != len(scores):
        raise RetrievalError("chunks and scores must be the same length")
    if top_k <= 0:
        return []
    if not chunks:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    relevance = _normalize_scores(np.asarray(scores, dtype=float))
    terms = _term_sets([c.text for c in chunks])

    remaining = set(range(len(chunks)))
    selected: list[int] = []
    per_url: Counter[str] = Counter()

    while remaining and len(selected) < top_k:
        capped = remaining
        if max_per_url is not None:
            allowed = {i for i in remaining if per_url[chunks[i].url] < max_per_url}
            # Relax the cap rather than return a short list.
            capped = allowed or remaining

        best_index = -1
        best_value = -math.inf
        for index in sorted(capped):
            redundancy = max(
                (jaccard(terms[index], terms[chosen]) for chosen in selected),
                default=0.0,
            )
            value = lambda_ * float(relevance[index]) - (1.0 - lambda_) * redundancy
            if value > best_value + 1e-12:
                best_value = value
                best_index = index

        if best_index < 0:
            break
        selected.append(best_index)
        remaining.discard(best_index)
        per_url[chunks[best_index].url] += 1

    return [chunks[i].with_score(float(scores[i])) for i in selected]


class Retriever:
    """Chunks documents and returns the top-k most useful, diverse passages."""

    def __init__(
        self,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        top_k: int = 6,
        bm25_weight: float = 0.65,
        mmr_lambda: float = 0.7,
        max_chunks_per_url: int | None = 2,
        scorer: Scorer | None = None,
        use_tfidf: bool | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.max_chunks_per_url = max_chunks_per_url

        if scorer is not None:
            self.scorer: Scorer = scorer
        else:
            # Only pay for TF-IDF when it is both wanted and installed.
            wants_tfidf = bm25_weight < 1.0 if use_tfidf is None else use_tfidf
            secondary: Scorer | None = None
            if wants_tfidf and TfidfCosineScorer.available():
                secondary = TfidfCosineScorer()
            elif wants_tfidf:
                log.debug("scikit-learn missing; falling back to BM25-only ranking")
            self.scorer = HybridScorer(
                BM25Scorer(),
                secondary,
                weight=1.0 if secondary is None else bm25_weight,
            )

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        return chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Score ``chunks`` and return them sorted by relevance, no diversity."""
        if not chunks:
            return []
        scores = self.scorer.score(query, [c.text for c in chunks])
        scored = [
            chunk.with_score(float(score)) for chunk, score in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda c: (-c.score, c.url, c.index))
        return scored

    def retrieve(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Full local retrieval: chunk, hybrid score, MMR select."""
        k = self.top_k if top_k is None else top_k
        if k <= 0 or not documents:
            return []

        chunks = self.chunk(documents)
        if not chunks:
            return []

        scores = self.scorer.score(query, [c.text for c in chunks])
        if len(scores) != len(chunks):
            raise RetrievalError("scorer returned the wrong number of scores")

        selected = mmr_select(
            chunks,
            [float(s) for s in scores],
            top_k=k,
            lambda_=self.mmr_lambda,
            max_per_url=self.max_chunks_per_url,
        )
        log.debug(
            "retrieved %d/%d chunks from %d urls",
            len(selected),
            len(chunks),
            len({c.url for c in selected}),
        )
        return selected
