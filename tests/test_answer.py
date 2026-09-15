"""Prompt construction, citation validation, and LLM clients.

The citation tests are the important ones. The prompt asks the model to cite
only real sources, but an instruction is not a guarantee, so these confirm that
a hallucinated ``[9]`` is detected and reported rather than passed through.
"""

from __future__ import annotations

import pytest

from eris.answer import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    AnswerBuilder,
    build_prompt,
    extract_citations,
    format_context,
    strip_invalid_citations,
    validate_citations,
)
from eris.errors import DependencyMissingError, LLMError
from eris.llm import AnthropicClient, EchoCitationClient, FakeClient, LLMClient
from eris.models import Chunk


def _chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            url=f"https://s{i}.example/page",
            title=f"Source {i}",
            text=f"Body text of source {i}.",
            index=0,
        )
        for i in range(1, count + 1)
    ]


class TestFormatContext:
    def test_numbers_sources_from_one(self) -> None:
        rendered = format_context(_chunks(3))
        assert "[1] Source 1" in rendered
        assert "[3] Source 3" in rendered
        assert "[0]" not in rendered

    def test_includes_urls_for_attribution(self) -> None:
        assert "URL: https://s1.example/page" in format_context(_chunks(1))

    def test_includes_body_text(self) -> None:
        assert "Body text of source 1." in format_context(_chunks(1))

    def test_separates_sources(self) -> None:
        assert "---" in format_context(_chunks(2))

    def test_truncates_overlong_chunks(self) -> None:
        chunk = Chunk(url="https://e.com", title="T", text="x" * 10_000)
        rendered = format_context([chunk], max_chars=100)
        assert len(rendered) < 400
        assert rendered.endswith("...")

    def test_falls_back_to_url_when_title_blank(self) -> None:
        chunk = Chunk(url="https://e.com/p", title="   ", text="body")
        assert "[1] https://e.com/p" in format_context([chunk])

    def test_no_chunks_yields_empty_string(self) -> None:
        assert format_context([]) == ""


class TestBuildPrompt:
    def test_contains_question_and_context(self) -> None:
        prompt = build_prompt("What is the payload limit?", _chunks(2))
        assert "What is the payload limit?" in prompt
        assert "Source 1" in prompt
        assert "CONTEXT" in prompt

    def test_instructs_bracket_citation(self) -> None:
        assert "[n]" in build_prompt("q", _chunks(1))

    def test_instructs_refusal_when_uncovered(self) -> None:
        assert "do not cover" in build_prompt("q", _chunks(1))

    def test_marks_absent_context_explicitly(self) -> None:
        assert "(no sources retrieved)" in build_prompt("q", [])

    @pytest.mark.parametrize("question", ["", "   "])
    def test_blank_question_is_rejected(self, question: str) -> None:
        with pytest.raises(ValueError, match="question"):
            build_prompt(question, _chunks(1))


class TestSystemPrompt:
    def test_forbids_outside_knowledge(self) -> None:
        assert "ONLY" in SYSTEM_PROMPT

    def test_forbids_invented_citations(self) -> None:
        assert "Never cite a number that does not appear" in SYSTEM_PROMPT

    def test_requires_admitting_gaps(self) -> None:
        assert "do not contain the answer" in SYSTEM_PROMPT


class TestExtractCitations:
    def test_finds_all_markers(self) -> None:
        assert extract_citations("Claim [1] and claim [2].") == [1, 2]

    def test_deduplicates_preserving_order(self) -> None:
        assert extract_citations("[2] then [1] then [2] again") == [2, 1]

    def test_handles_adjacent_markers(self) -> None:
        assert extract_citations("Both agree [1][2].") == [1, 2]

    def test_multi_digit_numbers(self) -> None:
        assert extract_citations("see [12]") == [12]

    def test_ignores_non_numeric_brackets(self) -> None:
        assert extract_citations("an [example] of [text]") == []

    def test_no_markers_yields_empty(self) -> None:
        assert extract_citations("No citations at all.") == []

    def test_empty_input_yields_empty(self) -> None:
        assert extract_citations("") == []


class TestValidateCitations:
    def test_in_range_citations_resolve_to_sources(self) -> None:
        valid, invalid = validate_citations("Claim [1] and [2].", _chunks(3))
        assert invalid == ()
        assert [c.n for c in valid] == [1, 2]
        assert valid[0].url == "https://s1.example/page"
        assert valid[0].title == "Source 1"

    def test_hallucinated_citation_is_rejected(self) -> None:
        """The headline guardrail: [9] cannot survive a 3-source context."""
        valid, invalid = validate_citations("Claim [9].", _chunks(3))
        assert invalid == (9,)
        assert valid == ()

    def test_mixed_valid_and_invalid_are_separated(self) -> None:
        valid, invalid = validate_citations("Real [1] fake [7] real [3].", _chunks(3))
        assert [c.n for c in valid] == [1, 3]
        assert invalid == (7,)

    def test_zero_is_invalid_because_context_is_one_indexed(self) -> None:
        _, invalid = validate_citations("Claim [0].", _chunks(3))
        assert invalid == (0,)

    def test_boundary_citation_is_valid(self) -> None:
        valid, invalid = validate_citations("Claim [3].", _chunks(3))
        assert invalid == ()
        assert valid[0].n == 3

    def test_one_past_the_boundary_is_invalid(self) -> None:
        _, invalid = validate_citations("Claim [4].", _chunks(3))
        assert invalid == (4,)

    def test_uncited_answer_yields_no_citations(self) -> None:
        valid, invalid = validate_citations("No markers here.", _chunks(3))
        assert valid == ()
        assert invalid == ()

    def test_any_citation_is_invalid_without_context(self) -> None:
        valid, invalid = validate_citations("Claim [1].", [])
        assert valid == ()
        assert invalid == (1,)

    def test_blank_title_falls_back_to_url(self) -> None:
        chunk = Chunk(url="https://e.com/p", title="  ", text="body")
        valid, _ = validate_citations("Claim [1].", [chunk])
        assert valid[0].title == "https://e.com/p"

    def test_logs_a_warning_for_hallucinations(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="eris.answer"):
            validate_citations("Claim [9].", _chunks(3))
        assert "do not exist" in caplog.text


class TestStripInvalidCitations:
    def test_removes_only_the_invalid_marker(self) -> None:
        assert strip_invalid_citations("Real [1] fake [9].", [9]) == "Real [1] fake."

    def test_keeps_text_unchanged_when_nothing_invalid(self) -> None:
        assert strip_invalid_citations("Real [1].", []) == "Real [1]."

    def test_tidies_space_before_punctuation(self) -> None:
        assert strip_invalid_citations("A claim [9].", [9]) == "A claim."

    def test_collapses_double_spaces(self) -> None:
        assert "  " not in strip_invalid_citations("A [9] claim here", [9])


class TestAnswerBuilder:
    def test_returns_text_and_citations(self) -> None:
        builder = AnswerBuilder(FakeClient(["Grounded claim [1] and [2]."]))
        answer = builder.build("q", _chunks(3))
        assert answer.text == "Grounded claim [1] and [2]."
        assert [c.n for c in answer.citations] == [1, 2]
        assert answer.invalid_citations == ()

    def test_reports_hallucinated_citations(self) -> None:
        builder = AnswerBuilder(FakeClient(["Claim [1] and [9]."]))
        answer = builder.build("q", _chunks(3))
        assert answer.invalid_citations == (9,)
        assert [c.n for c in answer.citations] == [1]

    def test_hallucinated_markers_can_be_stripped(self) -> None:
        builder = AnswerBuilder(FakeClient(["Claim [1] and [9]."]))
        answer = builder.build("q", _chunks(3), strip_hallucinated=True)
        assert "[9]" not in answer.text
        assert "[1]" in answer.text
        assert answer.invalid_citations == (9,)

    def test_invalid_citations_are_reported_even_when_not_stripped(self) -> None:
        builder = AnswerBuilder(FakeClient(["Claim [9]."]))
        answer = builder.build("q", _chunks(3))
        assert "[9]" in answer.text
        assert answer.invalid_citations == (9,)

    def test_model_is_not_called_without_context(self) -> None:
        """No sources means no grounding, so spending a request is pointless."""
        client = FakeClient(["should never be used"])
        answer = AnswerBuilder(client).build("q", [])
        assert answer.text == NO_CONTEXT_ANSWER
        assert client.call_count == 0

    def test_prompt_carries_the_question_and_sources(self) -> None:
        client = FakeClient(["ok [1]"])
        AnswerBuilder(client).build("what is the limit?", _chunks(2))
        assert "what is the limit?" in client.last_prompt
        assert "Source 2" in client.last_prompt

    def test_system_prompt_is_passed_through(self) -> None:
        client = FakeClient(["ok [1]"])
        AnswerBuilder(client).build("q", _chunks(1))
        assert client.calls[0][0] == SYSTEM_PROMPT

    def test_custom_system_prompt_is_used(self) -> None:
        client = FakeClient(["ok [1]"])
        AnswerBuilder(client, system_prompt="CUSTOM").build("q", _chunks(1))
        assert client.calls[0][0] == "CUSTOM"

    def test_answer_text_is_stripped(self) -> None:
        answer = AnswerBuilder(FakeClient(["  spaced [1]  "])).build("q", _chunks(1))
        assert answer.text == "spaced [1]"

    @pytest.mark.parametrize("reply", ["", "   ", "\n\n"])
    def test_empty_reply_raises(self, reply: str) -> None:
        with pytest.raises(LLMError, match="empty answer"):
            AnswerBuilder(FakeClient([reply])).build("q", _chunks(1))

    def test_client_failure_propagates_as_llm_error(self) -> None:
        builder = AnswerBuilder(FakeClient(error=LLMError("upstream down")))
        with pytest.raises(LLMError, match="upstream down"):
            builder.build("q", _chunks(1))

    def test_cited_urls_are_exposed(self) -> None:
        answer = AnswerBuilder(FakeClient(["Claim [2]."])).build("q", _chunks(3))
        assert answer.cited_urls == ("https://s2.example/page",)

    def test_answer_serialises(self) -> None:
        answer = AnswerBuilder(FakeClient(["Claim [1] and [9]."])).build("q", _chunks(2))
        payload = answer.to_dict()
        assert payload["invalid_citations"] == [9]
        assert payload["citations"][0]["url"] == "https://s1.example/page"


class TestFakeClient:
    def test_satisfies_the_client_protocol(self) -> None:
        assert isinstance(FakeClient(), LLMClient)

    def test_returns_replies_in_order(self) -> None:
        client = FakeClient(["first", "second"])
        assert client.complete("s", "p") == "first"
        assert client.complete("s", "p") == "second"

    def test_repeats_the_last_reply_once_exhausted(self) -> None:
        client = FakeClient(["only"])
        client.complete("s", "p")
        assert client.complete("s", "p") == "only"

    def test_accepts_a_bare_string(self) -> None:
        assert FakeClient("just one").complete("s", "p") == "just one"

    def test_handler_receives_both_prompts(self) -> None:
        client = FakeClient(handler=lambda system, prompt: f"{system}|{prompt}")
        assert client.complete("SYS", "PROMPT") == "SYS|PROMPT"

    def test_records_calls(self) -> None:
        client = FakeClient(["x"])
        client.complete("s", "p")
        assert client.calls == [("s", "p")]
        assert client.call_count == 1

    def test_raises_configured_error(self) -> None:
        with pytest.raises(LLMError):
            FakeClient(error=LLMError("boom")).complete("s", "p")

    def test_last_prompt_before_any_call_raises(self) -> None:
        with pytest.raises(AssertionError):
            _ = FakeClient().last_prompt

    def test_default_reply_admits_it_cannot_answer(self) -> None:
        assert "cannot answer" in FakeClient().complete("s", "p")


class TestEchoCitationClient:
    def test_cites_every_numbered_source(self) -> None:
        prompt = build_prompt("q", _chunks(3))
        reply = EchoCitationClient().complete(SYSTEM_PROMPT, prompt)
        assert "[1]" in reply
        assert "[2]" in reply

    def test_respects_the_citation_cap(self) -> None:
        prompt = build_prompt("q", _chunks(5))
        reply = EchoCitationClient(max_citations=2).complete(SYSTEM_PROMPT, prompt)
        assert extract_citations(reply) == [1, 2]

    def test_admits_when_no_sources_exist(self) -> None:
        reply = EchoCitationClient().complete(SYSTEM_PROMPT, "QUESTION: q")
        assert "do not cover" in reply

    def test_only_cites_valid_numbers(self) -> None:
        chunk_list = _chunks(3)
        reply = EchoCitationClient().complete(SYSTEM_PROMPT, build_prompt("q", chunk_list))
        _, invalid = validate_citations(reply, chunk_list)
        assert invalid == ()


class TestAnthropicClient:
    def test_blank_api_key_is_rejected(self) -> None:
        with pytest.raises(LLMError, match="API key is required"):
            AnthropicClient("   ", model="claude-x")

    def test_key_is_not_exposed_by_repr(self) -> None:
        client = AnthropicClient("dummy-key-redaction", model="claude-x")
        assert "dummy-key-redaction" not in repr(client)

    def test_model_is_recorded(self) -> None:
        assert AnthropicClient("dummy-key", model="claude-y").model == "claude-y"

    def test_missing_sdk_raises_dependency_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = AnthropicClient("dummy-key", model="claude-x")

        def _raise() -> object:
            raise DependencyMissingError("anthropic", "llm")

        monkeypatch.setattr(client, "_ensure_client", _raise)
        with pytest.raises(DependencyMissingError, match="eris\\[llm\\]"):
            client.complete("s", "p")

    def test_sdk_errors_are_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Messages:
            def create(self, **kwargs: object) -> object:
                raise RuntimeError("overloaded")

        class _Stub:
            messages = _Messages()

        client = AnthropicClient("dummy-key", model="claude-x")
        monkeypatch.setattr(client, "_ensure_client", lambda: _Stub())
        with pytest.raises(LLMError, match="Anthropic request failed"):
            client.complete("s", "p")

    def test_request_carries_configured_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        class _Block:
            text = "reply [1]"

        class _Messages:
            def create(self, **kwargs: object) -> object:
                seen.update(kwargs)
                return type("R", (), {"content": [_Block()]})()

        class _Stub:
            messages = _Messages()

        client = AnthropicClient("dummy-key", model="claude-z", max_tokens=77, temperature=0.0)
        monkeypatch.setattr(client, "_ensure_client", lambda: _Stub())
        assert client.complete("SYS", "PROMPT") == "reply [1]"
        assert seen["model"] == "claude-z"
        assert seen["max_tokens"] == 77
        assert seen["temperature"] == 0.0
        assert seen["system"] == "SYS"

    @pytest.mark.parametrize(
        "response",
        [
            {"content": [{"text": "a"}, {"text": "b"}]},
            {"content": "a\nb"},
        ],
    )
    def test_extracts_text_from_supported_shapes(self, response: object) -> None:
        assert AnthropicClient._extract_text(response) == "a\nb"

    @pytest.mark.parametrize("response", [{"content": []}, {"content": None}])
    def test_empty_response_raises(self, response: object) -> None:
        with pytest.raises(LLMError, match="empty response"):
            AnthropicClient._extract_text(response)

    def test_response_without_text_blocks_raises(self) -> None:
        with pytest.raises(LLMError, match="no text blocks"):
            AnthropicClient._extract_text({"content": [{"type": "tool_use"}]})


class TestSdkVersionTolerance:
    """The Messages API surface changes across SDK majors.

    ``temperature`` was removed from ``messages.create`` in anthropic 1.5, so the
    client sends only the arguments the installed signature accepts.
    """

    @staticmethod
    def _explicit(model: str, max_tokens: int, system: str, messages: object) -> None:
        """A create() without a temperature parameter, as in anthropic 1.5+."""

    @staticmethod
    def _with_temperature(
        model: str, max_tokens: int, system: str, messages: object, temperature: float
    ) -> None:
        """A create() that accepts temperature, as in anthropic 0.34."""

    def _request(self) -> dict[str, object]:
        return {
            "model": "m",
            "max_tokens": 10,
            "temperature": 0.0,
            "system": "s",
            "messages": [],
        }

    def test_unsupported_temperature_is_dropped(self) -> None:
        filtered = AnthropicClient._supported_kwargs(self._explicit, self._request())
        assert "temperature" not in filtered
        assert filtered["model"] == "m"
        assert filtered["max_tokens"] == 10

    def test_supported_temperature_is_kept(self) -> None:
        filtered = AnthropicClient._supported_kwargs(self._with_temperature, self._request())
        assert filtered["temperature"] == 0.0

    def test_var_keyword_signature_receives_everything(self) -> None:
        def _create(**kwargs: object) -> None:
            """A stub or future SDK that accepts arbitrary keywords."""

        assert AnthropicClient._supported_kwargs(_create, self._request()) == self._request()

    def test_uninspectable_callable_receives_everything(self) -> None:
        class _NotInspectable:
            """Not callable, so inspect.signature raises TypeError."""

        assert (
            AnthropicClient._supported_kwargs(_NotInspectable(), self._request()) == self._request()
        )

    def test_unrecognised_signature_receives_everything(self) -> None:
        """Filtering must never strip essential arguments.

        A signature missing model/messages/max_tokens means we misread it, so the
        full request is sent and the SDK reports a truthful error.
        """

        def _unrelated(sep: str = " ", end: str = "\n") -> None:
            """Nothing like messages.create."""

        assert AnthropicClient._supported_kwargs(_unrelated, self._request()) == self._request()

    def test_essential_arguments_always_survive_filtering(self) -> None:
        filtered = AnthropicClient._supported_kwargs(self._explicit, self._request())
        assert set(filtered) >= AnthropicClient._ESSENTIAL_KWARGS

    def test_complete_succeeds_against_a_signature_without_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        class _Messages:
            def create(self, model: str, max_tokens: int, system: str, messages: object) -> object:
                seen.update(
                    {
                        "model": model,
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": messages,
                    }
                )
                return {"content": [{"text": "reply [1]"}]}

        class _Stub:
            messages = _Messages()

        client = AnthropicClient("dummy-key", model="claude-new", temperature=0.0)
        monkeypatch.setattr(client, "_ensure_client", lambda: _Stub())
        assert client.complete("SYS", "PROMPT") == "reply [1]"
        assert seen["model"] == "claude-new"
        assert seen["system"] == "SYS"

    def test_dropped_arguments_are_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("DEBUG", logger="eris.llm"):
            AnthropicClient._supported_kwargs(self._explicit, self._request())
        assert "temperature" in caplog.text
