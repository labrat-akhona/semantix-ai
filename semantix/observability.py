"""Structured logging & observability for Semantix validation events."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Generator

logger = logging.getLogger("semantix")


@dataclass(frozen=True)
class ValidationEvent:
    """Structured record of a single validation attempt."""

    intent_name: str
    passed: bool
    score: float | None = None
    reason: str | None = None
    latency_ms: float = 0.0
    attempt: int = 1
    output_preview: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def log_validation(
    intent_name: str,
    attempt: int = 1,
    output_preview: str = "",
) -> Generator[dict[str, Any], None, None]:
    """Context manager that times a validation call and logs the result.

    Yields a mutable dict; the caller should set ``passed``, ``score``, and
    optionally ``reason`` before exiting the block.

    Example
    -------
    >>> with log_validation("ProfessionalDecline") as ctx:
    ...     verdict = judge.evaluate(...)
    ...     ctx["passed"] = verdict.passed
    ...     ctx["score"] = verdict.score
    """
    ctx: dict[str, Any] = {
        "passed": None,
        "score": None,
        "reason": None,
    }
    start = time.perf_counter()
    try:
        yield ctx
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        event = ValidationEvent(
            intent_name=intent_name,
            passed=bool(ctx.get("passed")),
            score=ctx.get("score"),
            reason=ctx.get("reason"),
            latency_ms=round(elapsed_ms, 2),
            attempt=attempt,
            output_preview=output_preview[:120],
        )
        level = logging.INFO if event.passed else logging.WARNING
        logger.log(
            level,
            "semantix.validation | intent=%s passed=%s score=%s "
            "latency_ms=%.2f attempt=%d",
            event.intent_name,
            event.passed,
            event.score,
            event.latency_ms,
            event.attempt,
        )
        # Also emit a structured record for programmatic consumption.
        logger.debug("semantix.validation.detail | %s", event.as_dict())
