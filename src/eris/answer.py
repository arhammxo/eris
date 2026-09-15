"""Grounded prompt construction and citation validation.

Two jobs. First, build a prompt that numbers the retrieved chunks and tells the
model to cite them as ``[n]`` and to admit when the sources do not answer the
question. Second - and this is the part that matters - *verify* the reply
afterwards. An instruction is a request, not a guarantee: a model can still emit
``[9]`` when only six sources exist. :func:`validate_citations` catches those,
reports them on :attr:`~eris.models.Answer.invalid_citations`, and returns only
citations that resolve to a real source.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from eris.errors import LLMError
from eris.llm import LLMClient
from eris.models import Answer, Chunk, Citation

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Eris, a research assistant that answers strictly from \
supplied web sources.

Rules:
1. Use ONLY the numbered sources in the CONTEXT block. Do not use prior knowledge \
about the topic, and never invent facts, numbers, dates or quotes.
2. Cite every factual claim inline with bracketed source numbers, e.g. [1] or [2][3]. \
Cite the source the claim actually came from.
3. Never cite a number that does not appear in the CONTEXT block.
4. If the sources do not contain the answer, say so plainly and explain what is missing. \
Do not guess, and do not pad the answer.
5. If sources disagree, say so and attribute each position to its source.
6. Be concise and factual. Lead with the answer, then the supporting detail."""

NO_CONTEXT_ANSWER = (
    "I could not retrieve any sources for this question, so I cannot answer it. "
    "This usually means the search returned no usable results or every page failed to fetch."
)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_MAX_CHUNK_CHARS = 4000


def format_context(chunks: Sequence[Chunk], *, max_chars: int = _MAX_CHUNK_CHARS) -> str:
    """Render chunks as a numbered context block, 1-indexed.

    The number, title and URL precede each passage so the model can attribute a
    claim without guessing, and so ``[n]`` maps unambiguously back to a source.
    """
    if not chunks:
        return ""
    blocks: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        title = chunk.title.strip() or chunk.url
        blocks.append(f"[{position}] {title}\nURL: {chunk.url}\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    """Assemble the user prompt from the question and retrieved context."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    context = format_context(chunks)
    if not context:
        context = "(no sources retrieved)"
    return (
        f"CONTEXT\n{context}\n\n"
        f"END CONTEXT\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the sources above, citing them as [n]. "
        "If the sources do not cover the question, say so explicitly."
    )


def extract_citations(text: str) -> list[int]:
    """Return the distinct ``[n]`` numbers in ``text``, in order of appearance."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def validate_citations(
    text: str,
    chunks: Sequence[Chunk],
) -> tuple[tuple[Citation, ...], tuple[int, ...]]:
    """Resolve cited numbers against ``chunks``.

    Returns ``(valid_citations, invalid_numbers)``. A number is invalid when it
    is out of range - including ``[0]``, since context is 1-indexed - and is
    reported rather than silently dropped, so a hallucinating model is visible
    to the caller and to the eval harness.
    """
    cited = extract_citations(text)
    valid: list[Citation] = []
    invalid: list[int] = []
    for number in cited:
        if 1 <= number <= len(chunks):
            chunk = chunks[number - 1]
            valid.append(Citation(n=number, url=chunk.url, title=chunk.title.strip() or chunk.url))
        else:
            invalid.append(number)
    if invalid:
        log.warning(
            "model cited %d source(s) that do not exist: %s (context had %d)",
            len(invalid),
            invalid,
            len(chunks),
        )
    return tuple(valid), tuple(invalid)


def strip_invalid_citations(text: str, invalid: Sequence[int]) -> str:
    """Remove hallucinated ``[n]`` markers, tidying the whitespace they leave."""
    if not invalid:
        return text
    unwanted = set(invalid)

    def _replace(match: re.Match[str]) -> str:
        return "" if int(match.group(1)) in unwanted else match.group(0)

    cleaned = _CITATION_RE.sub(_replace, text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


class AnswerBuilder:
    """Prompts the model over retrieved chunks and validates what comes back."""

    def __init__(self, client: LLMClient, *, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.client = client
        self.system_prompt = system_prompt

    def build(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        strip_hallucinated: bool = False,
    ) -> Answer:
        """Return a validated :class:`Answer` for ``question``.

        With no chunks the model is never called: there is nothing to ground an
        answer in, and a canned refusal is both cheaper and more honest.

        Raises:
            LLMError: if the client fails or returns an empty reply.
        """
        if not chunks:
            return Answer(text=NO_CONTEXT_ANSWER, citations=(), invalid_citations=())

        prompt = build_prompt(question, chunks)
        text = self.client.complete(self.system_prompt, prompt)
        if not isinstance(text, str) or not text.strip():
            raise LLMError("The language model returned an empty answer")
        text = text.strip()

        citations, invalid = validate_citations(text, chunks)
        if invalid and strip_hallucinated:
            text = strip_invalid_citations(text, invalid)
        return Answer(text=text, citations=citations, invalid_citations=invalid)
