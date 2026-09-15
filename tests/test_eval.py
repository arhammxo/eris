"""The offline eval harness: parsing, metrics, determinism and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eris.config import Settings, load_settings
from eris.eval import (
    EvalCase,
    EvalError,
    EvalReport,
    load_cases,
    run_eval,
    run_eval_file,
)
from eris.llm import EchoCitationClient, FakeClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_QA = REPO_ROOT / "examples" / "qa.jsonl"


def _case_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": "What is the payload limit?",
        "expected_url": "https://docs.example/limits",
        "sources": [
            {
                "url": "https://docs.example/limits",
                "title": "Limits",
                "snippet": "Payload limits.",
                "text": "The maximum request payload size is 10 MiB per request. "
                "Larger uploads must use the multipart endpoint. " * 4,
            },
            {
                "url": "https://docs.example/quickstart",
                "title": "Quickstart",
                "snippet": "Getting started.",
                "text": "Create a key in the console and send a signed request. "
                "Authentication headers are described below. " * 4,
            },
        ],
    }
    payload.update(overrides)
    return payload


def _write(path: Path, *payloads: dict[str, object]) -> Path:
    path.write_text("\n".join(json.dumps(p) for p in payloads), encoding="utf-8")
    return path


@pytest.fixture
def eval_settings(tmp_path: Path) -> Settings:
    return load_settings(
        search_backend="fake",
        cache_ttl_s=0,
        cache_dir=tmp_path,
        top_k=4,
        chunk_size=300,
        chunk_overlap=50,
    )


class TestEvalCaseParsing:
    def test_parses_a_valid_case(self) -> None:
        case = EvalCase.from_dict(_case_payload(), line_no=1)
        assert case.question == "What is the payload limit?"
        assert len(case.documents) == 2
        assert len(case.results) == 2

    def test_defaults_expected_url_to_the_first_source(self) -> None:
        payload = _case_payload()
        payload.pop("expected_url")
        assert EvalCase.from_dict(payload, line_no=1).expected_url == "https://docs.example/limits"

    def test_snippet_is_used_when_text_is_absent(self) -> None:
        payload = _case_payload(
            expected_url="https://e.com/a",
            sources=[{"url": "https://e.com/a", "title": "T", "snippet": "Body from snippet."}],
        )
        assert EvalCase.from_dict(payload, line_no=1).documents[0].text == "Body from snippet."

    def test_title_defaults_to_the_url(self) -> None:
        payload = _case_payload(
            expected_url="https://e.com/a",
            sources=[{"url": "https://e.com/a", "text": "Body text here."}],
        )
        assert EvalCase.from_dict(payload, line_no=1).documents[0].title == "https://e.com/a"

    def test_note_is_optional(self) -> None:
        assert EvalCase.from_dict(_case_payload(), line_no=1).note == ""

    def test_note_is_preserved(self) -> None:
        case = EvalCase.from_dict(_case_payload(note="tricky"), line_no=1)
        assert case.note == "tricky"

    @pytest.mark.parametrize("question", ["", "   "])
    def test_missing_question_is_rejected(self, question: str) -> None:
        with pytest.raises(EvalError, match="'question' is required"):
            EvalCase.from_dict(_case_payload(question=question), line_no=3)

    @pytest.mark.parametrize("sources", [[], None, "not a list"])
    def test_missing_sources_is_rejected(self, sources: object) -> None:
        with pytest.raises(EvalError, match="'sources' must be"):
            EvalCase.from_dict(_case_payload(sources=sources), line_no=4)

    def test_source_without_url_is_rejected(self) -> None:
        with pytest.raises(EvalError, match="url is required"):
            EvalCase.from_dict(_case_payload(sources=[{"text": "body"}]), line_no=5)

    def test_source_without_body_is_rejected(self) -> None:
        with pytest.raises(EvalError, match="needs 'text' or 'snippet'"):
            EvalCase.from_dict(_case_payload(sources=[{"url": "https://e.com/a"}]), line_no=6)

    def test_non_object_source_is_rejected(self) -> None:
        with pytest.raises(EvalError, match="must be an object"):
            EvalCase.from_dict(_case_payload(sources=["nope"]), line_no=7)

    def test_expected_url_outside_sources_is_rejected(self) -> None:
        """Catches a typo that would otherwise show up as a silent 0% hit-rate."""
        with pytest.raises(EvalError, match="not among the sources"):
            EvalCase.from_dict(_case_payload(expected_url="https://typo.example"), line_no=8)

    def test_error_message_names_the_line(self) -> None:
        with pytest.raises(EvalError, match="line 42"):
            EvalCase.from_dict(_case_payload(question=""), line_no=42)


class TestLoadCases:
    def test_loads_every_line(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "qa.jsonl", _case_payload(), _case_payload(question="Another?"))
        assert len(load_cases(path)) == 2

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        path.write_text(f"\n{json.dumps(_case_payload())}\n\n", encoding="utf-8")
        assert len(load_cases(path)) == 1

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        path.write_text(f"# a comment\n{json.dumps(_case_payload())}\n", encoding="utf-8")
        assert len(load_cases(path)) == 1

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match="not found"):
            load_cases(tmp_path / "absent.jsonl")

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(EvalError, match="line 1: invalid JSON"):
            load_cases(path)

    def test_non_object_line_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(EvalError, match="must be a JSON object"):
            load_cases(path)

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        path.write_text("# only comments\n", encoding="utf-8")
        with pytest.raises(EvalError, match="No eval cases"):
            load_cases(path)


class TestRunEval:
    def test_finds_the_expected_source(self, eval_settings: Settings) -> None:
        report = run_eval([EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings)
        assert report.hit_rate == 1.0

    def test_citations_are_valid_with_the_echo_client(self, eval_settings: Settings) -> None:
        report = run_eval([EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings)
        assert report.citation_validity == 1.0

    def test_reports_per_case_detail(self, eval_settings: Settings) -> None:
        report = run_eval([EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings)
        case = report.results[0]
        assert case.question == "What is the payload limit?"
        assert case.n_chunks > 0
        assert case.retrieved_urls

    def test_is_deterministic(self, eval_settings: Settings) -> None:
        cases = [EvalCase.from_dict(_case_payload(), line_no=1)]
        first = run_eval(cases, settings=eval_settings)
        second = run_eval(cases, settings=eval_settings)
        assert first.to_dict() == second.to_dict()

    def test_hallucinated_citation_fails_validity_not_hit_rate(
        self, eval_settings: Settings
    ) -> None:
        """The two metrics must move independently."""
        report = run_eval(
            [EvalCase.from_dict(_case_payload(), line_no=1)],
            settings=eval_settings,
            llm_client=FakeClient(["Claim [1] and [99]."]),
        )
        assert report.hit_rate == 1.0
        assert report.citation_validity == 0.0

    def test_missed_retrieval_lowers_hit_rate(self, eval_settings: Settings) -> None:
        """top_k=1 with the answer in the lower-ranked source must miss."""
        payload = _case_payload(expected_url="https://docs.example/quickstart")
        report = run_eval([EvalCase.from_dict(payload, line_no=1)], settings=eval_settings, top_k=1)
        assert report.hit_rate == 0.0

    def test_top_k_override_is_reported(self, eval_settings: Settings) -> None:
        report = run_eval(
            [EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings, top_k=2
        )
        assert report.top_k == 2
        assert report.results[0].n_chunks <= 2

    def test_no_cases_yields_zero_metrics(self, eval_settings: Settings) -> None:
        report = run_eval([], settings=eval_settings)
        assert report.total == 0
        assert report.hit_rate == 0.0
        assert report.citation_validity == 0.0

    def test_pipeline_error_is_recorded_not_raised(self, eval_settings: Settings) -> None:
        from eris.errors import LLMError

        report = run_eval(
            [EvalCase.from_dict(_case_payload(), line_no=1)],
            settings=eval_settings,
            llm_client=FakeClient(error=LLMError("model down")),
        )
        assert report.errors == 1
        assert "model down" in report.results[0].error

    def test_runs_without_an_api_key(self, eval_settings: Settings) -> None:
        assert eval_settings.has_api_key is False
        assert run_eval([EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings)


class TestEvalReport:
    def test_table_lists_every_case(self, eval_settings: Settings) -> None:
        report = run_eval(
            [
                EvalCase.from_dict(_case_payload(), line_no=1),
                EvalCase.from_dict(_case_payload(question="Second question?"), line_no=2),
            ],
            settings=eval_settings,
        )
        table = report.to_table()
        assert "hit@k" in table
        assert "PASS" in table
        assert "cases=2" in table

    def test_table_reports_both_metrics(self, eval_settings: Settings) -> None:
        table = run_eval(
            [EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings
        ).to_table()
        assert "hit_rate@4=100.0%" in table
        assert "citation_validity=100.0%" in table

    def test_table_truncates_long_questions(self, eval_settings: Settings) -> None:
        long_question = "Why " * 40 + "?"
        table = run_eval(
            [EvalCase.from_dict(_case_payload(question=long_question), line_no=1)],
            settings=eval_settings,
        ).to_table()
        assert "…" in table

    def test_empty_report_says_so(self) -> None:
        assert "No eval cases" in EvalReport().to_table()

    def test_json_payload_includes_metrics_and_cases(self, eval_settings: Settings) -> None:
        payload = run_eval(
            [EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings
        ).to_dict()
        assert payload["total"] == 1
        assert payload["hit_rate"] == 1.0
        assert len(payload["cases"]) == 1

    def test_json_payload_is_serialisable(self, eval_settings: Settings) -> None:
        payload = run_eval(
            [EvalCase.from_dict(_case_payload(), line_no=1)], settings=eval_settings
        ).to_dict()
        assert json.loads(json.dumps(payload))["total"] == 1


class TestShippedFixtures:
    def test_the_shipped_file_parses(self) -> None:
        assert len(load_cases(SHIPPED_QA)) >= 5

    def test_every_case_has_a_distractor_source(self) -> None:
        """A single-source case cannot distinguish good ranking from luck."""
        for case in load_cases(SHIPPED_QA):
            assert len(case.documents) >= 2, case.question

    def test_shipped_fixtures_score_perfectly(self, eval_settings: Settings) -> None:
        report = run_eval_file(SHIPPED_QA, settings=eval_settings, llm_client=EchoCitationClient())
        assert report.hit_rate == 1.0
        assert report.citation_validity == 1.0
        assert report.errors == 0

    def test_shipped_fixtures_are_deterministic(self, eval_settings: Settings) -> None:
        first = run_eval_file(SHIPPED_QA, settings=eval_settings).to_dict()
        second = run_eval_file(SHIPPED_QA, settings=eval_settings).to_dict()
        assert first == second

    def test_retrieval_is_discriminating_at_top_k_one(self, eval_settings: Settings) -> None:
        """With a single chunk of context the ranker must still find the answer.

        A weaker guarantee than the default top_k, and the one that would break
        first if BM25 scoring regressed.
        """
        report = run_eval_file(SHIPPED_QA, settings=eval_settings, top_k=1)
        assert report.hit_rate >= 0.8, report.to_table()
