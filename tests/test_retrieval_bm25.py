"""BM25 correctness, chunking behaviour and the hybrid scorer."""

from __future__ import annotations

import numpy as np
import pytest

from eris.errors import RetrievalError
from eris.models import Document
from eris.retrieval import (
    BM25Scorer,
    HybridScorer,
    TfidfCosineScorer,
    chunk_documents,
    chunk_text,
    tokenize,
)

CORPUS = [
    "the cat sat on the rug in the warm sun",  # 0: 'cat' only
    "cat cat cat cat cat cat cat cat",  # 1: 'cat', keyword-stuffed
    "dogs bark loudly at the postman every morning",  # 2: neither term
    "a cat and a dog shared the mat quietly",  # 3: 'cat' + 'mat'
]


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self) -> None:
        assert tokenize("Hello, WORLD! 42x", remove_stopwords=False) == ["hello", "world", "42x"]

    def test_removes_stopwords_by_default(self) -> None:
        assert tokenize("the cat and the dog") == ["cat", "dog"]

    def test_keeps_stopwords_when_asked(self) -> None:
        assert "the" in tokenize("the cat", remove_stopwords=False)

    def test_keeps_internal_apostrophes(self) -> None:
        assert tokenize("it's Ada's model", remove_stopwords=False) == [
            "it's",
            "ada's",
            "model",
        ]

    def test_empty_input_yields_no_tokens(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestBM25:
    def test_ranks_relevant_document_first(self) -> None:
        scores = BM25Scorer().score("cat mat", CORPUS)
        assert int(np.argmax(scores)) == 3, "the doc containing both terms should win"

    def test_document_without_query_terms_scores_zero(self) -> None:
        scores = BM25Scorer().score("cat", CORPUS)
        assert scores[2] == pytest.approx(0.0)

    def test_all_scores_are_non_negative(self) -> None:
        assert np.all(BM25Scorer().score("cat mat sun", CORPUS) >= 0.0)

    def test_term_frequency_saturates(self) -> None:
        """Eight 'cat's must not score eight times one 'cat'.

        This is the property that distinguishes BM25 from raw TF weighting.
        """
        scorer = BM25Scorer(k1=1.5, b=0.0)
        single = scorer.score("cat", ["cat filler filler filler filler filler filler filler"])[0]
        many = scorer.score("cat", ["cat cat cat cat cat cat cat cat"])[0]
        assert many > single
        assert many < 8 * single

    def test_k1_zero_removes_frequency_sensitivity(self) -> None:
        """With k1=0 the TF term collapses to 1, so repeats stop mattering."""
        scorer = BM25Scorer(k1=0.0, b=0.0)
        once = scorer.score("cat", ["cat dog", "cat cat cat dog"])
        assert once[0] == pytest.approx(once[1])

    def test_length_normalisation_favours_shorter_documents(self) -> None:
        scorer = BM25Scorer(b=1.0)
        corpus = ["cat", "cat " + "filler " * 200]
        scores = scorer.score("cat", corpus)
        assert scores[0] > scores[1]

    def test_b_zero_disables_length_normalisation(self) -> None:
        scorer = BM25Scorer(b=0.0)
        scores = scorer.score("cat", ["cat", "cat " + "filler " * 200])
        assert scores[0] == pytest.approx(scores[1])

    def test_rare_term_outweighs_common_term(self) -> None:
        """IDF must make a term appearing once worth more than one in every doc."""
        corpus = ["common rare", "common x", "common y", "common z"]
        scorer = BM25Scorer(b=0.0)
        rare = scorer.score("rare", corpus)
        common = scorer.score("common", corpus)
        assert rare[0] > common[0]

    def test_repeated_query_terms_do_not_double_count(self) -> None:
        scorer = BM25Scorer()
        once = scorer.score("cat", CORPUS)
        twice = scorer.score("cat cat", CORPUS)
        assert np.allclose(once, twice)

    def test_stopword_only_query_scores_zero(self) -> None:
        assert np.all(BM25Scorer().score("the and of", CORPUS) == 0.0)

    def test_empty_query_returns_zeros_of_right_length(self) -> None:
        scores = BM25Scorer().score("", CORPUS)
        assert scores.shape == (len(CORPUS),)
        assert np.all(scores == 0.0)

    def test_empty_corpus_returns_empty_array(self) -> None:
        assert BM25Scorer().score("cat", []).shape == (0,)

    def test_corpus_of_empty_strings_is_safe(self) -> None:
        scores = BM25Scorer().score("cat", ["", "   "])
        assert scores.shape == (2,)
        assert np.all(scores == 0.0)

    def test_score_is_deterministic(self) -> None:
        scorer = BM25Scorer()
        assert np.allclose(scorer.score("cat mat", CORPUS), scorer.score("cat mat", CORPUS))

    def test_case_is_ignored(self) -> None:
        scorer = BM25Scorer()
        assert np.allclose(scorer.score("CAT", CORPUS), scorer.score("cat", CORPUS))

    @pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.5), (1.5, -0.1), (1.5, 1.1)])
    def test_invalid_parameters_are_rejected(self, k1: float, b: float) -> None:
        with pytest.raises(ValueError, match=r"k1 must|b must"):
            BM25Scorer(k1=k1, b=b)


class TestTfidfScorer:
    def test_reports_availability(self) -> None:
        assert isinstance(TfidfCosineScorer.available(), bool)

    @pytest.mark.skipif(not TfidfCosineScorer.available(), reason="scikit-learn not installed")
    def test_ranks_matching_documents_above_non_matching(self) -> None:
        scores = TfidfCosineScorer().score("cat mat", CORPUS)
        assert scores[3] > scores[2], "a doc with both terms must beat one with neither"
        assert scores[2] == pytest.approx(0.0)

    @pytest.mark.skipif(not TfidfCosineScorer.available(), reason="scikit-learn not installed")
    def test_cosine_rewards_keyword_stuffing_more_than_bm25(self) -> None:
        """Documents why BM25 carries the larger share of the hybrid score.

        Docs 0 and 1 both match 'cat', but doc 1 is nothing except repetitions
        of it. Measured against doc 0, that stuffing lifts the cosine score
        proportionally more than the BM25 score, because BM25 saturates repeated
        terms while cosine simply points the document vector at the query.
        """
        cosine = TfidfCosineScorer().score("cat mat", CORPUS)
        bm25 = BM25Scorer().score("cat mat", CORPUS)
        cosine_gain = float(cosine[1] / cosine[0])
        bm25_gain = float(bm25[1] / bm25[0])
        assert cosine_gain > bm25_gain

    @pytest.mark.skipif(not TfidfCosineScorer.available(), reason="scikit-learn not installed")
    def test_scores_are_bounded_and_non_negative(self) -> None:
        scores = TfidfCosineScorer().score("cat", CORPUS)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-9)

    def test_empty_corpus_returns_empty_array(self) -> None:
        assert TfidfCosineScorer().score("cat", []).shape == (0,)

    def test_blank_query_returns_zeros(self) -> None:
        assert np.all(TfidfCosineScorer().score("  ", CORPUS) == 0.0)

    @pytest.mark.skipif(not TfidfCosineScorer.available(), reason="scikit-learn not installed")
    def test_stopword_only_corpus_does_not_raise(self) -> None:
        """An empty vocabulary must degrade to zeros, not a ValueError."""
        scores = TfidfCosineScorer().score("the and", ["the and", "of the"])
        assert scores.shape == (2,)


class TestHybridScorer:
    def test_weight_one_ignores_secondary(self) -> None:
        primary = BM25Scorer()
        hybrid = HybridScorer(primary, _ConstantScorer(5.0), weight=1.0)
        assert np.allclose(hybrid.score("cat", CORPUS), _normalised(primary, "cat"))

    def test_weight_zero_uses_secondary_only(self) -> None:
        hybrid = HybridScorer(BM25Scorer(), _RankReversingScorer(), weight=0.0)
        scores = hybrid.score("cat", CORPUS)
        assert int(np.argmax(scores)) == len(CORPUS) - 1

    def test_output_stays_within_unit_range(self) -> None:
        hybrid = HybridScorer(BM25Scorer(), _RankReversingScorer(), weight=0.5)
        scores = hybrid.score("cat mat", CORPUS)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-9)

    def test_missing_secondary_is_allowed(self) -> None:
        scores = HybridScorer(BM25Scorer(), None, weight=0.4).score("cat", CORPUS)
        assert scores.shape == (len(CORPUS),)

    def test_blend_sits_between_its_components(self) -> None:
        primary, secondary = BM25Scorer(), _RankReversingScorer()
        blended = HybridScorer(primary, secondary, weight=0.5).score("cat", CORPUS)
        lo = np.minimum(_normalised(primary, "cat"), _normalised(secondary, "cat"))
        hi = np.maximum(_normalised(primary, "cat"), _normalised(secondary, "cat"))
        assert np.all(blended >= lo - 1e-9)
        assert np.all(blended <= hi + 1e-9)

    def test_empty_corpus_returns_empty_array(self) -> None:
        assert HybridScorer(BM25Scorer()).score("cat", []).shape == (0,)

    def test_mismatched_scorer_shape_is_rejected(self) -> None:
        with pytest.raises(RetrievalError, match="mismatched"):
            HybridScorer(BM25Scorer(), _WrongLengthScorer(), weight=0.5).score("cat", CORPUS)

    @pytest.mark.parametrize("weight", [-0.1, 1.1])
    def test_invalid_weight_is_rejected(self, weight: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            HybridScorer(BM25Scorer(), weight=weight)


class TestChunkText:
    def test_short_text_is_a_single_chunk(self) -> None:
        assert chunk_text("short text", chunk_size=100, overlap=10) == ["short text"]

    def test_long_text_is_split(self) -> None:
        chunks = chunk_text("word " * 500, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_chunks_respect_the_size_budget(self) -> None:
        for chunk in chunk_text("sentence here. " * 200, chunk_size=150, overlap=30):
            assert len(chunk) <= 150

    def test_chunks_cover_the_whole_text(self) -> None:
        text = " ".join(f"token{i}" for i in range(400))
        joined = " ".join(chunk_text(text, chunk_size=200, overlap=0))
        assert "token0" in joined
        assert "token399" in joined

    def test_overlap_repeats_content_across_the_seam(self) -> None:
        text = " ".join(f"w{i}" for i in range(200))
        with_overlap = chunk_text(text, chunk_size=200, overlap=100)
        without = chunk_text(text, chunk_size=200, overlap=0)
        assert len(with_overlap) > len(without)

    def test_prefers_sentence_boundaries(self) -> None:
        text = "First sentence is here. " * 30
        assert any(
            chunk.rstrip().endswith(".") for chunk in chunk_text(text, chunk_size=120, overlap=20)
        )

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("", chunk_size=100, overlap=10) == []
        assert chunk_text("    ", chunk_size=100, overlap=10) == []

    def test_terminates_on_unbroken_text(self) -> None:
        """No whitespace means no boundary; the loop must still advance."""
        chunks = chunk_text("x" * 1000, chunk_size=100, overlap=50)
        assert len(chunks) >= 10

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 200)],
    )
    def test_invalid_parameters_are_rejected(self, size: int, overlap: int) -> None:
        with pytest.raises(RetrievalError):
            chunk_text("some text here", chunk_size=size, overlap=overlap)


class TestChunkDocuments:
    def test_indexes_restart_per_document(self, documents: list[Document]) -> None:
        chunks = chunk_documents(documents, chunk_size=200, overlap=20)
        for document in documents:
            indexes = [c.index for c in chunks if c.url == document.url]
            assert indexes == list(range(len(indexes)))

    def test_source_id_is_unique(self, documents: list[Document]) -> None:
        chunks = chunk_documents(documents, chunk_size=200, overlap=20)
        ids = [c.source_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_is_carried_through(self, documents: list[Document]) -> None:
        chunks = chunk_documents(documents, chunk_size=200, overlap=20)
        titles = {c.url: c.title for c in chunks}
        assert titles["https://a.example/python"] == "Python 3.13 release notes"

    def test_tiny_chunks_are_dropped(self) -> None:
        document = Document(url="https://x.example", title="x", text="tiny")
        assert chunk_documents([document], chunk_size=500, overlap=0, min_chars=40) == []

    def test_no_documents_yields_no_chunks(self) -> None:
        assert chunk_documents([], chunk_size=200, overlap=20) == []


class _ConstantScorer:
    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, query: str, corpus: list[str]) -> np.ndarray:
        return np.full(len(corpus), self.value, dtype=float)


class _RankReversingScorer:
    """Scores strictly increasing with index, to invert BM25's ordering."""

    name = "reversed"

    def score(self, query: str, corpus: list[str]) -> np.ndarray:
        return np.arange(len(corpus), dtype=float)


class _WrongLengthScorer:
    name = "wrong-length"

    def score(self, query: str, corpus: list[str]) -> np.ndarray:
        return np.zeros(len(corpus) + 1, dtype=float)


def _normalised(scorer: object, query: str) -> np.ndarray:
    raw = np.asarray(scorer.score(query, CORPUS), dtype=float)  # type: ignore[attr-defined]
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)
