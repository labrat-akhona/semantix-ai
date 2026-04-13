# pytest-semantix (`assert_semantic`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `assert_semantic()` — a one-function testing primitive that lets any developer validate LLM outputs against semantic intents in pytest (or any test runner).

**Architecture:** Single module `semantix/testing.py` exposing `assert_semantic(output, intent, *, judge, threshold)`. Accepts either a string description (creates a dynamic Intent) or an Intent subclass. Uses the same default judge resolution and threshold logic as `@validate_intent`. On failure, raises `AssertionError` with score, intent, truncated output, and reason. Re-exported from `semantix/__init__.py`.

**Tech Stack:** Pure Python, no new dependencies. Uses existing `semantix.intent.Intent`, `semantix.judges.Judge`, `semantix.judges.Verdict`.

---

### Task 1: Failing tests for `assert_semantic`

**Files:**
- Create: `semantix/tests/test_testing.py`

These tests use `MockJudge` from `semantix/tests/conftest.py` (already available as pytest fixtures aren't needed — just import directly).

- [ ] **Step 1: Write the test file with all test cases**

```python
"""Tests for semantix.testing — the assert_semantic() function."""

from __future__ import annotations

import pytest

from semantix.intent import Intent
from semantix.judges import Verdict
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


class Strict(Intent):
    """The text must be extremely formal."""
    threshold = 0.95


def test_passing_assertion_with_string_intent():
    """String intent + passing judge → no error."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.90)
    assert_semantic("Thank you for your patience", "must be polite", judge=judge)
    assert judge.call_count == 1
    assert judge.last_output == "Thank you for your patience"
    assert "must be polite" in judge.last_description


def test_passing_assertion_with_intent_class():
    """Intent class → uses its docstring as the description."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.90)
    assert_semantic("Thank you", Polite, judge=judge)
    assert judge.call_count == 1
    assert "polite and professional" in judge.last_description


def test_failing_assertion_raises_assertion_error():
    """Failing judge → AssertionError with score, intent, output, reason."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=False, score=0.12, reason="Text is aggressive")
    with pytest.raises(AssertionError, match=r"score=0\.12"):
        assert_semantic("You're an idiot", "must be polite", judge=judge)


def test_failure_message_contains_intent_description():
    """The error message includes what the text was supposed to satisfy."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=False, score=0.30, reason="Not formal enough")
    with pytest.raises(AssertionError, match="must be polite"):
        assert_semantic("hey dude", "must be polite", judge=judge)


def test_failure_message_contains_output_preview():
    """The error message includes a preview of the offending output."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=False, score=0.10)
    with pytest.raises(AssertionError, match="hey dude"):
        assert_semantic("hey dude", "must be polite", judge=judge)


def test_failure_message_contains_reason():
    """When the judge provides a reason, it appears in the error."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=False, score=0.10, reason="Informal greeting")
    with pytest.raises(AssertionError, match="Informal greeting"):
        assert_semantic("hey dude", "must be polite", judge=judge)


def test_long_output_is_truncated_in_message():
    """Output longer than 200 chars is truncated in the error message."""
    from semantix.testing import assert_semantic

    long_text = "x" * 300
    judge = MockJudge(passed=False, score=0.10)
    with pytest.raises(AssertionError) as exc_info:
        assert_semantic(long_text, "must be short", judge=judge)
    # The full 300-char string should NOT appear — it should be truncated
    assert "x" * 300 not in str(exc_info.value)
    assert "x" * 200 in str(exc_info.value)


def test_explicit_threshold_is_forwarded_to_judge():
    """When threshold is passed, it overrides intent/judge defaults."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.90)
    assert_semantic("hello", "must be polite", judge=judge, threshold=0.5)
    assert judge.last_threshold == 0.5


def test_intent_class_threshold_is_used():
    """When an Intent class has explicit threshold, it's used."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.96)
    assert_semantic("Dear Sir/Madam", Strict, judge=judge)
    assert judge.last_threshold == 0.95


def test_explicit_threshold_overrides_intent_class():
    """Explicit threshold param beats Intent class threshold."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.90)
    assert_semantic("Dear Sir/Madam", Strict, judge=judge, threshold=0.5)
    assert judge.last_threshold == 0.5


def test_judge_recommended_threshold_used_for_string_intent():
    """String intents have no explicit threshold, so judge's recommended_threshold applies."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.90)
    judge.recommended_threshold = 0.65
    assert_semantic("hello", "must be polite", judge=judge)
    assert judge.last_threshold == 0.65


def test_none_score_in_failure_message():
    """When judge returns score=None, message shows N/A."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=False, score=None)
    # Override the Verdict to have score=None
    original_evaluate = judge.evaluate

    def evaluate_none_score(output, desc, threshold=0.8):
        v = original_evaluate(output, desc, threshold)
        return Verdict(passed=False, score=None, reason=v.reason)

    judge.evaluate = evaluate_none_score
    with pytest.raises(AssertionError, match="score=N/A"):
        assert_semantic("bad text", "must be good", judge=judge)


def test_return_value_is_none_on_success():
    """assert_semantic returns None on success (like assert)."""
    from semantix.testing import assert_semantic

    judge = MockJudge(passed=True, score=0.95)
    result = assert_semantic("good text", "must be good", judge=judge)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_testing.py -v 2>&1 | head -30`
Expected: ERRORS — `ImportError: cannot import name 'assert_semantic' from 'semantix.testing'` (module doesn't exist yet)

---

### Task 2: Implement `assert_semantic`

**Files:**
- Create: `semantix/testing.py`

- [ ] **Step 3: Write the implementation**

```python
"""Semantic assertions for testing — validate LLM outputs against intents.

Usage::

    from semantix.testing import assert_semantic

    def test_chatbot_is_polite():
        response = my_llm("handle angry customer")
        assert_semantic(response, "polite and professional")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from semantix.intent import Intent
from semantix.judges import Judge, Verdict

if TYPE_CHECKING:
    pass


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
    if isinstance(intent, str):
        intent_cls = _make_dynamic_intent(intent)
    else:
        intent_cls = intent

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
            f"  Output:  \"{preview}\""
            f"{reason_line}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_testing.py -v`
Expected: All 14 tests PASS

---

### Task 3: Re-export from `__init__.py` and commit

**Files:**
- Modify: `semantix/__init__.py`

- [ ] **Step 5: Add assert_semantic to the public API**

In `semantix/__init__.py`, add the import after the existing imports (line ~29):

```python
from semantix.testing import assert_semantic
```

And add `"assert_semantic"` to the `__all__` list in the `# Core` section:

```python
__all__ = [
    # Core
    "Intent",
    "SemanticIntentError",
    "validate_intent",
    "get_last_failure",
    "assert_semantic",
    ...
]
```

- [ ] **Step 6: Run full test suite**

Run: `python3 -m pytest semantix/tests/ -v 2>&1 | tail -20`
Expected: All tests pass (180 existing + 14 new = 194)

- [ ] **Step 7: Run lint**

Run: `ruff check semantix/testing.py semantix/tests/test_testing.py && ruff format --check semantix/testing.py semantix/tests/test_testing.py`
Expected: All checks passed, files already formatted

- [ ] **Step 8: Commit**

```bash
git add semantix/testing.py semantix/tests/test_testing.py semantix/__init__.py
git commit -m "feat: add assert_semantic() for testing LLM outputs against intents"
```
