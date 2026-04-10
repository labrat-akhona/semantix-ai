"""Streaming support — validate intent once a streamed response is fully assembled."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from semantix.exceptions import SemanticIntentError
from semantix.intent import Intent
from semantix.judges import Judge, Verdict
from semantix.observability import log_validation


class StreamCollector:
    """Accumulates text chunks from a streaming LLM response and validates
    the full output against an ``Intent`` once the stream ends.

    Works as both a **sync** and **async** context manager and an
    **iterator / async-iterator** wrapper.

    Parameters
    ----------
    intent_cls:
        The Intent subclass to validate against.
    judge:
        The Judge to use for validation.

    Example (sync iterator)
    -----------------------
    >>> collector = StreamCollector(ProfessionalDecline, judge=my_judge)
    >>> for chunk in collector.wrap(stream_from_llm()):
    ...     print(chunk, end="")
    >>> result = collector.result()  # Intent instance or raises

    Example (async context manager)
    -------------------------------
    >>> async with StreamCollector(ProfessionalDecline, judge=my_judge) as sc:
    ...     async for chunk in llm_stream:
    ...         sc.feed(chunk)
    >>> result = sc.result()
    """

    def __init__(self, intent_cls: type[Intent], judge: Judge) -> None:
        if not isinstance(intent_cls, type) or not issubclass(intent_cls, Intent):
            raise TypeError(f"Expected Intent subclass, got {intent_cls!r}")
        self._intent_cls = intent_cls
        self._judge = judge
        self._chunks: list[str] = []
        self._verdict: Verdict | None = None
        self._validated = False

    # ── feeding ─────────────────────────────────────────────────

    def feed(self, chunk: str) -> None:
        """Append a text chunk to the buffer."""
        self._chunks.append(chunk)

    @property
    def text(self) -> str:
        """Return the accumulated text so far."""
        return "".join(self._chunks)

    # ── sync iterator wrapper ──────────────────────────────────

    def wrap(self, stream: Iterator[str]) -> Iterator[str]:
        """Wrap a sync string iterator, yielding chunks through and
        validating once exhausted."""
        for chunk in stream:
            self.feed(chunk)
            yield chunk
        self._validate()

    async def awrap(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Wrap an async string iterator, yielding chunks through and
        validating once exhausted."""
        async for chunk in stream:
            self.feed(chunk)
            yield chunk
        self._validate()

    # ── context managers ────────────────────────────────────────

    def __enter__(self) -> StreamCollector:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self._validate()

    async def __aenter__(self) -> StreamCollector:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self._validate()

    # ── result ──────────────────────────────────────────────────

    def result(self) -> Intent:
        """Return the validated Intent instance.

        Raises ``SemanticIntentError`` if validation failed, or
        ``RuntimeError`` if the stream hasn't been fully consumed yet.
        """
        if not self._validated:
            self._validate()
        assert self._verdict is not None
        full_text = self.text
        if not self._verdict.passed:
            raise SemanticIntentError(
                output=full_text,
                intent_name=self._intent_cls.__name__,
                intent_description=self._intent_cls.description(),
                score=self._verdict.score,
            )
        return self._intent_cls(full_text)

    # ── internals ───────────────────────────────────────────────

    def _validate(self) -> None:
        if self._validated:
            return
        full_text = self.text
        description = self._intent_cls.description()
        threshold = self._intent_cls.threshold

        with log_validation(
            self._intent_cls.__name__,
            output_preview=full_text,
        ) as ctx:
            self._verdict = self._judge.evaluate(full_text, description, threshold)
            ctx["passed"] = self._verdict.passed
            ctx["score"] = self._verdict.score
            ctx["reason"] = self._verdict.reason

        self._validated = True
