"""Language-model clients.

The :class:`LLMClient` protocol is the seam that keeps the test suite offline:
:class:`FakeClient` satisfies it without a network call or an API key. The
Anthropic SDK is imported lazily inside :class:`AnthropicClient`, so ``import
eris`` never requires it, and every SDK exception is wrapped in
:class:`~eris.errors.LLMError` so callers do not import ``anthropic`` to write
an ``except`` clause.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from eris.errors import DependencyMissingError, LLMError

log = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Turns a system prompt plus a user prompt into text."""

    model: str

    def complete(self, system: str, prompt: str) -> str:
        """Return the model's reply.

        Raises:
            LLMError: on any provider or transport failure.
        """
        ...


class AnthropicClient:
    """Claude client over the official Anthropic SDK.

    Args:
        api_key: Secret value; never logged and never written to disk.
        model: Claude model id.
        max_tokens: Cap on generated tokens.
        temperature: 0.0 by default, since a grounded, citation-bearing answer
            should be reproducible.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError("An Anthropic API key is required to build AnthropicClient")
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise DependencyMissingError("anthropic", "llm") from exc
        try:
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self.timeout_s,
                max_retries=self.max_retries,
            )
        except Exception as exc:
            raise LLMError(f"Could not initialise the Anthropic client: {exc}") from exc
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        client = self._ensure_client()
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        create = client.messages.create
        try:
            response = create(**self._supported_kwargs(create, request))
        except DependencyMissingError:
            raise
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {type(exc).__name__}: {exc}") from exc
        return self._extract_text(response)

    # Without these the request is meaningless, so their absence from a
    # signature means we misread it rather than that the SDK dropped them.
    _ESSENTIAL_KWARGS = frozenset({"model", "messages", "max_tokens"})

    @staticmethod
    def _supported_kwargs(create: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Drop request keys the installed SDK does not accept.

        The Messages API surface moves between major SDK versions - ``temperature``
        was removed from ``messages.create`` in anthropic 1.5, for example. Rather
        than pin a narrow version range, send only what the signature accepts and
        log what was dropped.

        The full request is sent unchanged when the signature takes ``**kwargs``,
        cannot be introspected, or does not mention every essential argument. That
        last case means the signature is not the one we expect, and a truthful
        error from the SDK beats a silently crippled request.
        """
        import inspect

        try:
            parameters = inspect.signature(create).parameters
        except (TypeError, ValueError):
            return request
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return request
        if not AnthropicClient._ESSENTIAL_KWARGS.issubset(parameters):
            log.debug("unrecognised messages.create signature; sending the full request")
            return request

        supported = {key: value for key, value in request.items() if key in parameters}
        dropped = sorted(set(request) - set(supported))
        if dropped:
            log.debug("installed anthropic SDK does not accept: %s", ", ".join(dropped))
        return supported

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Join the text blocks of a Messages API response.

        Tolerates dict-shaped blocks so a stubbed SDK works in tests.
        """
        blocks = getattr(response, "content", None)
        if blocks is None and isinstance(response, dict):
            blocks = response.get("content")
        if isinstance(blocks, str):
            return blocks.strip()
        if not blocks:
            raise LLMError("Anthropic returned an empty response")

        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        if not parts:
            raise LLMError("Anthropic response contained no text blocks")
        return "\n".join(parts).strip()


class FakeClient:
    """Scripted client for tests and offline eval.

    Supply ``replies`` to return them in order (the last one repeats once
    exhausted), or ``handler`` to compute a reply from the prompts. Every call
    is recorded in :attr:`calls` so tests can assert on the prompt.
    """

    def __init__(
        self,
        replies: Sequence[str] | str | None = None,
        *,
        handler: Callable[[str, str], str] | None = None,
        model: str = "fake-model",
        error: Exception | None = None,
    ) -> None:
        if isinstance(replies, str):
            replies = [replies]
        self._replies = list(replies or [])
        self._handler = handler
        self.model = model
        self._error = error
        self.calls: list[tuple[str, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_prompt(self) -> str:
        if not self.calls:
            raise AssertionError("FakeClient has not been called")
        return self.calls[-1][1]

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if self._error is not None:
            raise self._error
        if self._handler is not None:
            return self._handler(system, prompt)
        if not self._replies:
            return "No sources were provided, so I cannot answer."
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return self._replies[index]


class EchoCitationClient:
    """Fake client that cites every source it is shown.

    Gives the eval harness a deterministic, perfect-citation baseline, which
    isolates retrieval quality from model behaviour.
    """

    model = "echo-citation"

    def __init__(self, max_citations: int = 3) -> None:
        self.max_citations = max_citations
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        import re

        self.calls.append((system, prompt))
        numbers = sorted({int(n) for n in re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE)})
        if not numbers:
            return "The provided sources do not cover this question."
        cited = " ".join(f"[{n}]" for n in numbers[: self.max_citations])
        return f"Based on the retrieved sources: {cited}"
