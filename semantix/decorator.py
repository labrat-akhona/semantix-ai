"""The @validate_intent decorator — the core runtime hook."""

from __future__ import annotations

import functools
import inspect
import logging
from contextvars import ContextVar
from typing import Any, Callable, TypeVar, get_type_hints, overload

from semantix.exceptions import SemanticIntentError
from semantix.intent import Intent
from semantix.judges import Judge, Verdict
from semantix.observability import log_validation

# Stores the most recent validation failure in the current async/thread context.
# Set before each retry so the decorated LLM function can read it via get_last_failure().
_last_failure: ContextVar[SemanticIntentError | None] = ContextVar(
    "_last_failure", default=None
)


def get_last_failure() -> SemanticIntentError | None:
    """Return the most recent ``SemanticIntentError`` in the current context.

    This is the manual opt-in path for self-healing retries. Call this inside
    your LLM function to discover *why* the previous attempt failed, then add
    that feedback to your prompt.

    For automatic injection, declare ``semantix_feedback: Optional[str] = None``
    in your function signature instead — the decorator will pass the feedback
    string directly without any boilerplate.

    Example
    -------
    >>> @validate_intent(judge=NLIJudge(), retries=2)
    ... def decline(event: str) -> ProfessionalDecline:
    ...     hint = ""
    ...     if failure := get_last_failure():
    ...         hint = f"\\n\\nPrevious score: {failure.score:.2f}. Be more polite."
    ...     return call_llm(f"Decline this invite: {event}{hint}")
    """
    return _last_failure.get()


F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger("semantix")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_intent_class(func: Callable[..., Any]) -> type[Intent] | None:
    """Return the Intent subclass from *func*'s return annotation, or None."""
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001
        return None
    ret = hints.get("return")
    if ret is None:
        return None
    if isinstance(ret, type) and issubclass(ret, Intent) and ret is not Intent:
        return ret
    return None


def _accepts_feedback(func: Callable[..., Any]) -> bool:
    """Return True if *func* declares a ``semantix_feedback`` keyword parameter.

    Checked once at decoration time — never called on the hot path.
    """
    try:
        return "semantix_feedback" in inspect.signature(func).parameters
    except (ValueError, TypeError):
        return False


def _build_feedback(err: SemanticIntentError, attempt: int) -> str:
    """Build a Markdown-formatted feedback string suitable for LLM consumption.

    The string is designed to be appended to a prompt so the model understands
    exactly what it got wrong and what is required.

    Parameters
    ----------
    err:
        The ``SemanticIntentError`` from the failed attempt.
    attempt:
        The 1-based attempt number that just failed.
    """
    score_str = f"{err.score:.4f}" if err.score is not None else "N/A"
    reason_line = f"\n- **Judge reason:** {err.reason}" if err.reason else ""
    return (
        f"## Semantix Self-Healing Feedback\n\n"
        f"Attempt **{attempt}** failed validation.\n\n"
        f"### What went wrong\n"
        f"- **Intent:** `{err.intent_name}`\n"
        f"- **Score:** {score_str} (threshold not met){reason_line}\n\n"
        f"### What is required\n"
        f"{err.intent_description.strip()}\n\n"
        f"### Your previous output (rejected)\n"
        f"```\n{err.output[:400]}\n```\n\n"
        f"Please generate a new response that satisfies the requirement above."
    )


def _run_judge(
    judge: Judge,
    raw_output: str,
    intent_cls: type[Intent],
    attempt: int = 1,
) -> Intent:
    """Validate *raw_output* against *intent_cls* using *judge*.

    Returns an ``Intent`` instance on success; raises ``SemanticIntentError`` on failure.
    """
    description = intent_cls.description()
    threshold = intent_cls.threshold

    with log_validation(
        intent_cls.__name__,
        attempt=attempt,
        output_preview=raw_output,
    ) as ctx:
        verdict: Verdict = judge.evaluate(raw_output, description, threshold)
        ctx["passed"] = verdict.passed
        ctx["score"] = verdict.score
        ctx["reason"] = verdict.reason

    if not verdict.passed:
        raise SemanticIntentError(
            output=raw_output,
            intent_name=intent_cls.__name__,
            intent_description=description,
            score=verdict.score,
            reason=verdict.reason,
        )
    return intent_cls(raw_output)


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


@overload
def validate_intent(func: F) -> F: ...


@overload
def validate_intent(
    *,
    judge: Judge,
    retries: int = ...,
) -> Callable[[F], F]: ...


def validate_intent(
    func: F | None = None,
    *,
    judge: Judge | None = None,
    retries: int = 0,
) -> F | Callable[[F], F]:
    """Decorator that validates an LLM call's return value against its Intent.

    Can be used bare (``@validate_intent``) or with parameters
    (``@validate_intent(judge=NLIJudge(), retries=2)``).

    Parameters
    ----------
    judge:
        The ``Judge`` backend to use.  Defaults to ``NLIJudge()``
        (local, no API key required).
    retries:
        Number of **additional** attempts if the first call fails validation.
        The decorated function is re-invoked on each retry so the LLM has a
        chance to produce a different output.  ``0`` means no retries (fail
        immediately).

    When the decorated function's return type is an ``Intent`` subclass the
    decorator will:

    1. Call the original function and capture the **raw string** output.
    2. Forward the string + the Intent's docstring to the configured *Judge*.
    3. If the output fails validation and retries remain, go back to step 1.
    4. If no retries remain, raise ``SemanticIntentError``.
    5. Otherwise wrap the string in an instance of the Intent subclass and
       return it.

    If the return type is **not** an Intent subclass the decorator is a no-op.

    Self-Healing Retries
    --------------------
    Declare ``semantix_feedback: Optional[str] = None`` in your function
    signature and the decorator will automatically inject a Markdown-formatted
    failure report on each retry — no boilerplate required:

    .. code-block:: python

        from typing import Optional
        from semantix import Intent, NLIJudge, validate_intent

        class ProfessionalDecline(Intent):
            \"\"\"The text must politely decline an invitation without being rude.\"\"\"

        @validate_intent(judge=NLIJudge(), retries=2)
        def decline(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
            prompt = f"Decline this invite: {event}"
            if semantix_feedback:
                prompt += f"\\n\\n{semantix_feedback}"
            return call_llm(prompt)

    On the first call ``semantix_feedback`` is ``None``.  If validation fails,
    the next call receives a structured Markdown block explaining what went
    wrong, the score, the requirement, and the rejected output.

    The ``get_last_failure()`` ContextVar approach also remains available for
    manual control or custom feedback formatting.
    """

    def decorator(fn: F) -> F:
        intent_cls = _resolve_intent_class(fn)

        # Fast path — nothing to validate.
        if intent_cls is None:
            return fn

        # Choose the judge once at decoration time.
        # NLIJudge is the default — runs locally, no API key needed,
        # and entailment is more accurate than cosine similarity.
        _judge = judge
        if _judge is None:
            from semantix.judges.nli import NLIJudge

            _judge = NLIJudge()

        max_attempts = 1 + retries

        # Detect feedback support once — avoids inspect overhead on every call.
        _fn_accepts_feedback: bool = _accepts_feedback(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_err: SemanticIntentError | None = None

                for attempt in range(1, max_attempts + 1):
                    raw = await fn(*args, **kwargs)
                    raw_str = str(raw) if not isinstance(raw, str) else raw

                    try:
                        result = _run_judge(_judge, raw_str, intent_cls, attempt)  # type: ignore[arg-type]
                        # Success — clean up any injected state.
                        _last_failure.set(None)
                        kwargs.pop("semantix_feedback", None)
                        return result

                    except SemanticIntentError as err:
                        last_err = err
                        if attempt < max_attempts:
                            # Keep ContextVar for backward-compat manual access.
                            _last_failure.set(err)
                            # Direct injection if function opted in.
                            if _fn_accepts_feedback:
                                kwargs["semantix_feedback"] = _build_feedback(err, attempt)
                            logger.info(
                                "Retry %d/%d for %s (score=%.4f)",
                                attempt,
                                retries,
                                intent_cls.__name__,  # type: ignore[union-attr]
                                err.score or 0.0,
                            )

                # All attempts exhausted — clean up and raise.
                kwargs.pop("semantix_feedback", None)
                raise last_err  # type: ignore[misc]

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_err: SemanticIntentError | None = None

            for attempt in range(1, max_attempts + 1):
                raw = fn(*args, **kwargs)
                raw_str = str(raw) if not isinstance(raw, str) else raw

                try:
                    result = _run_judge(_judge, raw_str, intent_cls, attempt)  # type: ignore[arg-type]
                    # Success — clean up any injected state.
                    _last_failure.set(None)
                    kwargs.pop("semantix_feedback", None)
                    return result

                except SemanticIntentError as err:
                    last_err = err
                    if attempt < max_attempts:
                        # Keep ContextVar for backward-compat manual access.
                        _last_failure.set(err)
                        # Direct injection if function opted in.
                        if _fn_accepts_feedback:
                            kwargs["semantix_feedback"] = _build_feedback(err, attempt)
                        logger.info(
                            "Retry %d/%d for %s (score=%.4f)",
                            attempt,
                            retries,
                            intent_cls.__name__,  # type: ignore[union-attr]
                            err.score or 0.0,
                        )

            # All attempts exhausted — clean up and raise.
            kwargs.pop("semantix_feedback", None)
            raise last_err  # type: ignore[misc]

        return sync_wrapper  # type: ignore[return-value]

    # Support both @validate_intent and @validate_intent(judge=...)
    if func is not None:
        return decorator(func)
    return decorator
