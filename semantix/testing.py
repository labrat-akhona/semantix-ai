"""Semantic assertions for testing — validate LLM outputs against intents.

Usage::

    from semantix.testing import assert_semantic

    def test_chatbot_is_polite():
        response = my_llm("handle angry customer")
        assert_semantic(response, "polite and professional")
"""

from __future__ import annotations

from semantix.intent import Intent
from semantix.judges import Judge, Verdict


def _default_judge() -> Judge:
    """Resolve the default judge — same logic as @validate_intent."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


def _make_dynamic_intent(description: str) -> type[Intent]:
    """Create a dynamic Intent subclass from a plain string description.

    The resulting class has no explicit ``threshold`` in its ``__dict__``,
    so the judge's ``recommended_threshold`` will apply when available.
    """
    return type(
        f"_DynamicIntent_{id(description)}",
        (Intent,),
        {"__doc__": description},
    )


def assert_semantic(
    output: str,
    intent: str | type[Intent],
    *,
    judge: Judge | None = None,
    threshold: float | None = None,
) -> None:
    """Assert that *output* satisfies a semantic *intent*.

    Parameters
    ----------
    output:
        The text to validate.
    intent:
        Either a plain string description (e.g. ``"must be polite"``)
        or an ``Intent`` subclass with a docstring.
    judge:
        Judge backend. Defaults to QuantizedNLIJudge → NLIJudge fallback.
    threshold:
        Override the threshold. When ``None``, uses the intent's threshold
        or the judge's ``recommended_threshold``.

    Raises
    ------
    AssertionError
        When the output fails semantic validation, with score, intent
        description, output preview, and reason in the message.
    """
    # Resolve intent
    intent_cls = _make_dynamic_intent(intent) if isinstance(intent, str) else intent

    # Resolve judge
    _judge = judge if judge is not None else _default_judge()

    # Resolve threshold: explicit param > intent __dict__ > judge recommended > intent default
    if threshold is not None:
        _threshold = threshold
    elif "threshold" not in intent_cls.__dict__ and _judge.recommended_threshold is not None:
        _threshold = _judge.recommended_threshold
    else:
        _threshold = intent_cls.threshold

    # Evaluate
    description = intent_cls.description()
    verdict: Verdict = _judge.evaluate(output, description, _threshold)

    if not verdict.passed:
        score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
        reason_line = f"\n  Reason:  {verdict.reason}" if verdict.reason else ""
        preview = output[:200] + "..." if len(output) > 200 else output
        raise AssertionError(
            f"Semantic check failed (score={score_str})\n"
            f"  Intent:  {description}\n"
            f'  Output:  "{preview}"'
            f"{reason_line}"
        )
