"""MMR selection, source diversity and the end-to-end Retriever."""

from __future__ import annotations

import pytest

from eris.errors import RetrievalError
from eris.models import Chunk, Document
from eris.retrieval import Retriever, jaccard, mmr_select


def _chunk(url: str, text: str, index: int = 0) -> Chunk:
    return Chunk(url=url, title=url, text=text, index=index)


class TestJaccard:
    def test_identical_sets_are_one(self) -> None:
        assert jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_disjoint_sets_are_zero(self) -> None:
        assert jaccard({"a"}, {"b"}) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_empty_sets_are_zero_not_undefined(self) -> None:
        assert jaccard(set(), set()) == 0.0
        assert jaccard({"a"}, set()) == 0.0

    def test_is_symmetric(self) -> None:
        a, b = {"a", "b", "c"}, {"b", "c", "d"}
        assert jaccard(a, b) == jaccard(b, a)


class TestMmrSelect:
    def test_returns_at_most_top_k(self, chunks: list[Chunk]) -> None:
        selected = mmr_select(chunks, [1.0, 0.9, 0.8, 0.7], top_k=2, max_per_url=None)
        assert len(selected) == 2

    def test_returns_everything_when_k_exceeds_supply(self, chunks: list[Chunk]) -> None:
        selected = mmr_select(chunks, [1.0, 0.9, 0.8, 0.7], top_k=99, max_per_url=None)
        assert len(selected) == len(chunks)

    def test_highest_scoring_chunk_is_selected_first(self, chunks: list[Chunk]) -> None:
        selected = mmr_select(chunks, [0.1, 0.2, 0.9, 0.3], top_k=1, lambda_=1.0)
        assert selected[0].url == "https://b.example/two"

    def test_lambda_one_is_pure_relevance_ordering(self) -> None:
        items = [_chunk(f"https://s{i}.example", f"unique text {i}") for i in range(4)]
        selected = mmr_select(items, [0.1, 0.4, 0.9, 0.6], top_k=3, lambda_=1.0, max_per_url=None)
        assert [c.url for c in selected] == [
            "https://s2.example",
            "https://s3.example",
            "https://s1.example",
        ]

    def test_lambda_zero_penalises_redundancy(self) -> None:
        """With relevance ignored, the near-duplicate must be avoided."""
        items = [
            _chunk("https://a.example", "alpha beta gamma delta"),
            _chunk("https://b.example", "alpha beta gamma delta epsilon"),
            _chunk("https://c.example", "completely different vocabulary entirely"),
        ]
        selected = mmr_select(items, [1.0, 0.99, 0.1], top_k=2, lambda_=0.0, max_per_url=None)
        assert selected[1].url == "https://c.example"

    def test_diversifies_across_urls_at_moderate_lambda(self) -> None:
        """MMR alone, with no per-URL cap, must still reach a second source.

        Three near-identical chunks from one page score highest; a distinct
        chunk from another page scores lower. A filler chunk anchors the low end
        of the score range so the distinct chunk does not normalise to zero
        relevance, which would make the comparison vacuous.
        """
        items = [
            _chunk("https://a.example", "shared token text about widgets", 0),
            _chunk("https://a.example", "shared token text about widgets again", 1),
            _chunk("https://a.example", "shared token text about widgets once more", 2),
            _chunk("https://b.example", "entirely separate discussion of gadgets", 0),
            _chunk("https://z.example", "irrelevant filler", 0),
        ]
        selected = mmr_select(
            items, [1.0, 0.98, 0.97, 0.7, 0.0], top_k=2, lambda_=0.5, max_per_url=None
        )
        assert len({c.url for c in selected}) == 2

    def test_relevance_only_selection_would_be_redundant(self) -> None:
        """The control for the test above: λ=1 takes both chunks from one page."""
        items = [
            _chunk("https://a.example", "shared token text about widgets", 0),
            _chunk("https://a.example", "shared token text about widgets again", 1),
            _chunk("https://b.example", "entirely separate discussion of gadgets", 0),
        ]
        selected = mmr_select(items, [1.0, 0.98, 0.7], top_k=2, lambda_=1.0, max_per_url=None)
        assert {c.url for c in selected} == {"https://a.example"}

    def test_max_per_url_caps_a_single_source(self) -> None:
        items = [_chunk("https://a.example", f"text variant number {i}", i) for i in range(5)]
        items.append(_chunk("https://b.example", "different source entirely"))
        selected = mmr_select(items, [1.0] * 5 + [0.01], top_k=3, max_per_url=2)
        from_a = [c for c in selected if c.url == "https://a.example"]
        assert len(from_a) <= 2

    def test_cap_relaxes_rather_than_returning_short(self) -> None:
        """One source with a cap of 1 must still fill top_k=3, not return 1."""
        items = [_chunk("https://a.example", f"text number {i}", i) for i in range(4)]
        selected = mmr_select(items, [1.0, 0.9, 0.8, 0.7], top_k=3, max_per_url=1)
        assert len(selected) == 3

    def test_scores_are_attached_to_results(self, chunks: list[Chunk]) -> None:
        scores = [0.5, 0.25, 0.75, 0.1]
        selected = mmr_select(chunks, scores, top_k=4, max_per_url=None)
        for chunk in selected:
            original = next(i for i, c in enumerate(chunks) if c.source_id == chunk.source_id)
            assert chunk.score == pytest.approx(scores[original])

    def test_no_chunk_is_selected_twice(self, chunks: list[Chunk]) -> None:
        selected = mmr_select(chunks, [1.0, 1.0, 1.0, 1.0], top_k=4, max_per_url=None)
        ids = [c.source_id for c in selected]
        assert len(ids) == len(set(ids))

    def test_is_deterministic_on_ties(self) -> None:
        items = [_chunk(f"https://s{i}.example", "identical text here") for i in range(5)]
        scores = [1.0] * 5
        first = [c.source_id for c in mmr_select(items, scores, top_k=3, max_per_url=None)]
        second = [c.source_id for c in mmr_select(items, scores, top_k=3, max_per_url=None)]
        assert first == second

    def test_empty_input_returns_empty(self) -> None:
        assert mmr_select([], [], top_k=5) == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_non_positive_top_k_returns_empty(self, chunks: list[Chunk], top_k: int) -> None:
        assert mmr_select(chunks, [1.0] * len(chunks), top_k=top_k) == []

    def test_length_mismatch_is_rejected(self, chunks: list[Chunk]) -> None:
        with pytest.raises(RetrievalError, match="same length"):
            mmr_select(chunks, [1.0, 2.0], top_k=2)

    @pytest.mark.parametrize("lambda_", [-0.1, 1.1])
    def test_invalid_lambda_is_rejected(self, chunks: list[Chunk], lambda_: float) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            mmr_select(chunks, [1.0] * len(chunks), top_k=2, lambda_=lambda_)

    def test_identical_scores_do_not_break_normalisation(self, chunks: list[Chunk]) -> None:
        selected = mmr_select(chunks, [0.5] * len(chunks), top_k=3, max_per_url=None)
        assert len(selected) == 3


class TestRetriever:
    def test_retrieves_at_most_top_k(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20, top_k=3)
        assert len(retriever.retrieve("interpreter lock", documents)) <= 3

    def test_top_k_can_be_overridden_per_call(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20, top_k=6)
        assert len(retriever.retrieve("interpreter lock", documents, top_k=2)) == 2

    def test_surfaces_the_on_topic_document(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20, top_k=3)
        selected = retriever.retrieve("global interpreter lock free threaded", documents)
        assert "https://a.example/python" in {c.url for c in selected}

    def test_excludes_the_off_topic_document(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20, top_k=2)
        selected = retriever.retrieve("global interpreter lock free threaded", documents)
        assert "https://c.example/gardening" not in {c.url for c in selected}

    def test_results_carry_scores(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20, top_k=3)
        selected = retriever.retrieve("interpreter lock", documents)
        assert any(c.score > 0.0 for c in selected)

    def test_no_documents_returns_empty(self) -> None:
        assert Retriever().retrieve("anything", []) == []

    def test_non_positive_top_k_returns_empty(self, documents: list[Document]) -> None:
        assert Retriever(top_k=1).retrieve("q", documents, top_k=0) == []

    def test_documents_without_usable_text_return_empty(self) -> None:
        docs = [Document(url="https://x.example", title="x", text="   ")]
        assert Retriever(chunk_size=200, chunk_overlap=20).retrieve("q", docs) == []

    def test_rank_orders_by_descending_score(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20)
        ranked = retriever.rank("interpreter lock", retriever.chunk(documents))
        scores = [c.score for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_is_stable_for_equal_scores(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20)
        chunk_list = retriever.chunk(documents)
        first = [c.source_id for c in retriever.rank("zzz-no-match", chunk_list)]
        second = [c.source_id for c in retriever.rank("zzz-no-match", chunk_list)]
        assert first == second

    def test_rank_of_empty_chunk_list_is_empty(self) -> None:
        assert Retriever().rank("q", []) == []

    def test_bm25_only_mode_skips_tfidf(self, documents: list[Document]) -> None:
        retriever = Retriever(chunk_size=200, chunk_overlap=20, bm25_weight=1.0, top_k=2)
        assert len(retriever.retrieve("interpreter lock", documents)) == 2

    def test_injected_scorer_is_used(self, documents: list[Document]) -> None:
        calls: list[str] = []

        class _Recording:
            name = "recording"

            def score(self, query: str, corpus: list[str]) -> list[float]:
                calls.append(query)
                import numpy as np

                return np.linspace(1.0, 0.0, len(corpus))

        retriever = Retriever(chunk_size=200, chunk_overlap=20, scorer=_Recording(), top_k=2)
        retriever.retrieve("my query", documents)
        assert calls == ["my query"]

    def test_diversifies_sources_by_default(self, documents: list[Document]) -> None:
        """With one chunk allowed per URL and top_k equal to the source count,
        every returned chunk must come from a different page."""
        retriever = Retriever(chunk_size=150, chunk_overlap=20, top_k=3, max_chunks_per_url=1)
        selected = retriever.retrieve("interpreter lock threads", documents)
        urls = [c.url for c in selected]
        assert len(urls) == len(set(urls)) == 3

    def test_fills_top_k_by_relaxing_the_cap(self, documents: list[Document]) -> None:
        """Asking for more chunks than sources spends the budget rather than
        returning a short list."""
        retriever = Retriever(chunk_size=150, chunk_overlap=20, top_k=5, max_chunks_per_url=1)
        selected = retriever.retrieve("interpreter lock threads", documents)
        assert len(selected) == 5
        assert len({c.url for c in selected}) == 3

    def test_scorer_returning_wrong_length_is_rejected(self, documents: list[Document]) -> None:
        class _Broken:
            name = "broken"

            def score(self, query: str, corpus: list[str]) -> list[float]:
                return [0.0]

        retriever = Retriever(chunk_size=200, chunk_overlap=20, scorer=_Broken())
        with pytest.raises(RetrievalError, match="wrong number"):
            retriever.retrieve("q", documents)
