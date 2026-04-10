"""Guardrails integration — validate outputs with semantix semantic intents.

Usage::

    from guardrails import Guard
    from semantix.integrations.guardrails import SemanticIntent

    guard = Guard().use(SemanticIntent("must be polite and professional"))
    result = guard.validate("Thank you for your patience")
"""

from __future__ import annotations

from typing import Any

from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

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


@register_validator(name="semantix/semantic_intent", data_type="string")
class SemanticIntent(Validator):
    """Validates that text satisfies a natural language intent.

    Uses semantix's local NLI judge for fast (~15ms), offline semantic
    validation with no API cost.

    Parameters
    ----------
    intent:
        Plain English description of what the text should mean.
    threshold:
        Minimum score to pass (0-1). Defaults to the judge's
        recommended threshold, or 0.8 if none.
    judge:
        Optional Judge backend override. Defaults to QuantizedNLIJudge.
    on_fail:
        Guardrails on_fail action (e.g. "reask", "exception").
    """

    def __init__(
        self,
        intent: str,
        *,
        threshold: float | None = None,
        judge: Judge | None = None,
        on_fail: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(on_fail=on_fail, intent=intent, **kwargs)
        self._judge = judge or _default_judge()
        self._intent_cls = type(
            f"_GuardrailsIntent_{id(intent)}",
            (Intent,),
            {"__doc__": intent},
        )
        if threshold is not None:
            self._threshold = threshold
        elif self._judge.recommended_threshold is not None:
            self._threshold = self._judge.recommended_threshold
        else:
            self._threshold = self._intent_cls.threshold

    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        text = str(value) if not isinstance(value, str) else value
        description = self._intent_cls.description()
        verdict = self._judge.evaluate(text, description, self._threshold)

        if verdict.passed:
            return PassResult()

        score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
        reason_str = f": {verdict.reason}" if verdict.reason else ""
        return FailResult(
            error_message=(
                f"Semantic validation failed (score={score_str}){reason_str}. "
                f"The text must satisfy: {description}"
            ),
        )
