"""Instructor integration — validate fields with semantix Intents.

Usage with semantic_validator::

    from semantix.integrations.instructor import semantic_validator
    from semantix import Intent

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    class Response(BaseModel):
        reply: Annotated[str, AfterValidator(semantic_validator(Polite))]

Usage with SemanticStr shorthand::

    from semantix.integrations.instructor import SemanticStr

    class Response(BaseModel):
        reply: SemanticStr["must be polite", 0.85]
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator

from semantix.intent import Intent
from semantix.judges import Judge


def _default_judge() -> Judge:
    """Resolve the default judge — same logic as @validate_intent."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


def semantic_validator(
    intent: type[Intent],
    judge: Judge | None = None,
) -> callable:
    """Return a Pydantic AfterValidator-compatible callable.

    On success the original value is returned unchanged.
    On failure a ``ValueError`` is raised with the score and reason —
    Instructor catches this and retries automatically.
    """
    _judge = judge or _default_judge()
    description = intent.description()
    threshold = intent.threshold

    def _validate(value: Any) -> Any:
        text = str(value) if not isinstance(value, str) else value
        verdict = _judge.evaluate(text, description, threshold)
        if not verdict.passed:
            score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
            reason_str = f": {verdict.reason}" if verdict.reason else ""
            raise ValueError(
                f"Semantic validation failed (score={score_str}){reason_str}. "
                f"The text must satisfy: {description}"
            )
        return value

    return _validate


class _SemanticStrMeta(type):
    """Metaclass enabling ``SemanticStr["description", threshold]`` syntax."""

    def __getitem__(cls, params: str | tuple) -> Any:
        if isinstance(params, str):
            desc, threshold = params, 0.8
        elif isinstance(params, tuple) and len(params) == 2:
            desc, threshold = params
        else:
            raise TypeError(
                "SemanticStr expects SemanticStr['description'] or "
                "SemanticStr['description', threshold]"
            )

        # Create a dynamic Intent subclass from the description string.
        dynamic_intent = type(
            f"_SemanticStr_{id(desc)}",
            (Intent,),
            {"__doc__": desc, "threshold": threshold},
        )

        return Annotated[str, AfterValidator(semantic_validator(dynamic_intent))]


class SemanticStr(metaclass=_SemanticStrMeta):
    """Shorthand for annotating string fields with semantic validation.

    ``SemanticStr["must be polite"]`` is equivalent to defining an Intent
    subclass with that docstring and wrapping it in an AfterValidator.
    """
