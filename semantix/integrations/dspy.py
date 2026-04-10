"""DSPy integration — use semantix Intents as reward functions for Refine/BestOfN.

DSPy v2.6+ uses ``dspy.Refine`` and ``dspy.BestOfN`` with a ``reward_fn``
that scores outputs from 0.0 to 1.0.  ``semantic_reward`` turns any
semantix Intent into a compatible reward function.

Usage with dspy.Refine::

    import dspy
    from semantix import Intent
    from semantix.integrations.dspy import semantic_reward

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    qa = dspy.ChainOfThought("question -> answer")
    refined = dspy.Refine(module=qa, N=3, reward_fn=semantic_reward(Polite))
    result = refined(question="Handle an angry customer")

Usage with dspy.BestOfN::

    best = dspy.BestOfN(module=qa, N=5, reward_fn=semantic_reward(Polite))
    result = best(question="Handle an angry customer")

Specify which output field to validate (defaults to the last field)::

    semantic_reward(Polite, field="answer")
"""

from __future__ import annotations

from collections.abc import Callable
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


def semantic_reward(
    intent: type[Intent] | str,
    *,
    field: str | None = None,
    judge: Judge | None = None,
    threshold: float | None = None,
) -> Callable[[dict[str, Any], Any], float]:
    """Create a DSPy reward function from a semantix Intent.

    Parameters
    ----------
    intent:
        An Intent subclass or a plain-English description string.
    field:
        The prediction field to validate. If ``None``, uses the last
        attribute on the Prediction object.
    judge:
        Judge to use. Defaults to QuantizedNLIJudge or NLIJudge.
    threshold:
        Score threshold — outputs at or above this score get reward 1.0,
        below get the raw score. If ``None``, uses the judge's
        recommended threshold or the intent's default.

    Returns
    -------
    A ``reward_fn(args, pred) -> float`` compatible with
    ``dspy.Refine`` and ``dspy.BestOfN``.
    """
    _judge = judge or _default_judge()

    if isinstance(intent, str):
        intent_cls: type[Intent] = type(
            f"_DSPyIntent_{id(intent)}",
            (Intent,),
            {"__doc__": intent},
        )
    else:
        intent_cls = intent

    description = intent_cls.description()

    if threshold is not None:
        _threshold = threshold
    elif "threshold" not in intent_cls.__dict__ and _judge.recommended_threshold is not None:
        _threshold = _judge.recommended_threshold
    else:
        _threshold = intent_cls.threshold

    def _reward_fn(args: dict[str, Any], pred: Any) -> float:
        if field is not None:
            text = str(getattr(pred, field))
        else:
            # Use the last attribute on the prediction
            keys = list(pred.keys()) if hasattr(pred, "keys") else []
            text = str(pred[keys[-1]]) if keys else str(pred)

        verdict = _judge.evaluate(text, description, _threshold)
        if verdict.score is not None:
            return float(verdict.score)
        return 1.0 if verdict.passed else 0.0

    return _reward_fn


def semantic_metric(
    intent: type[Intent] | str,
    *,
    field: str | None = None,
    judge: Judge | None = None,
    threshold: float | None = None,
) -> Callable[..., float]:
    """Create a DSPy metric function from a semantix Intent.

    A metric function has the signature ``metric(example, pred, ...) -> float``
    and is used in ``dspy.Evaluate`` and optimizers like ``dspy.MIPROv2``.

    Parameters
    ----------
    intent:
        An Intent subclass or a plain-English description string.
    field:
        The prediction field to validate. If ``None``, uses the last
        attribute on the Prediction object.
    judge:
        Judge to use. Defaults to QuantizedNLIJudge or NLIJudge.
    threshold:
        Score threshold. If ``None``, uses the judge's recommended
        threshold or the intent's default.

    Returns
    -------
    A ``metric(example, pred) -> float`` compatible with
    ``dspy.Evaluate`` and DSPy optimizers.
    """
    reward = semantic_reward(intent, field=field, judge=judge, threshold=threshold)

    def _metric(example: Any, pred: Any, *_args: Any, **_kwargs: Any) -> float:
        args = {}
        if hasattr(example, "keys"):
            args = {k: example[k] for k in example.keys()}  # noqa: SIM118
        elif hasattr(example, "__dict__"):
            args = {k: v for k, v in example.__dict__.items() if not k.startswith("_")}
        return reward(args, pred)

    return _metric
