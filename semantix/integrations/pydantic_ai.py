"""Pydantic AI integration — validate agent outputs with semantix Intents.

Usage::

    from semantix.integrations.pydantic_ai import semantix_validator
    from semantix import Intent
    from pydantic_ai import Agent

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    agent = Agent("openai:gpt-4o", output_type=str)
    agent.output_validator(semantix_validator(Polite))
"""

from __future__ import annotations

from typing import Any

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


def semantix_validator(
    intent: type[Intent],
    judge: Judge | None = None,
    judge_from_deps: bool = False,
) -> Any:
    """Return a function compatible with ``@agent.output_validator``.

    Parameters
    ----------
    intent:
        The Intent subclass whose docstring defines the semantic requirement.
    judge:
        Judge to use. Defaults to QuantizedNLIJudge or NLIJudge.
        Ignored when ``judge_from_deps=True``.
    judge_from_deps:
        If True, read the judge from ``ctx.deps`` at runtime instead of
        using a fixed judge. The deps object must be a ``Judge`` instance.
    """
    _judge = None if judge_from_deps else (judge or _default_judge())
    description = intent.description()
    # Use the judge's recommended threshold when the Intent doesn't explicitly set one.
    if (
        _judge is not None
        and "threshold" not in intent.__dict__
        and _judge.recommended_threshold is not None
    ):
        threshold = _judge.recommended_threshold
    else:
        threshold = intent.threshold

    def _validate(ctx: Any, output: Any) -> Any:
        active_judge = ctx.deps if judge_from_deps else _judge
        text = str(output) if not isinstance(output, str) else output
        verdict = active_judge.evaluate(text, description, threshold)
        if not verdict.passed:
            score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
            reason_str = f": {verdict.reason}" if verdict.reason else ""
            try:
                from pydantic_ai import ModelRetry

                raise ModelRetry(
                    f"Semantic validation failed (score={score_str}){reason_str}. "
                    f"The text must satisfy: {description}"
                )
            except ImportError:
                raise ValueError(
                    f"Semantic validation failed (score={score_str}){reason_str}. "
                    f"The text must satisfy: {description}"
                ) from None
        return output

    return _validate
