# Guardrails Hub Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `SemanticIntent` Guardrails validator so semantix shows up as a drop-in validator in any Guard pipeline.

**Architecture:** Single file `semantix/integrations/guardrails.py` implementing Guardrails' `Validator` base class. Accepts a plain string intent, creates a dynamic Intent subclass internally, evaluates via semantix's judge system, returns `PassResult`/`FailResult`. Registered as `"semantix/semantic_intent"`.

**Tech Stack:** Python, guardrails-ai (`Validator`, `PassResult`, `FailResult`, `register_validator`), semantix internals (`Intent`, `Judge`, `Verdict`).

---

## File Structure

- **Create:** `semantix/integrations/guardrails.py` — the `SemanticIntent` validator
- **Create:** `semantix/tests/test_guardrails.py` — tests using MockJudge
- **Modify:** `pyproject.toml:46-79` — add `guardrails` optional dependency

---

### Task 1: Write failing tests

**Files:**
- Create: `semantix/tests/test_guardrails.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for semantix.integrations.guardrails — SemanticIntent validator."""

from __future__ import annotations

import pytest

from semantix.tests.conftest import MockJudge


def test_passing_validation_returns_pass_result():
    """When judge passes, _validate returns PassResult."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    validator = SemanticIntent("must be polite", judge=judge)
    result = validator._validate("Thank you for your patience", {})

    from guardrails.validators import PassResult

    assert isinstance(result, PassResult)


def test_failing_validation_returns_fail_result():
    """When judge fails, _validate returns FailResult with error message."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=False, score=0.12, reason="Text is aggressive")
    validator = SemanticIntent("must be polite", judge=judge)
    result = validator._validate("You're an idiot", {})

    from guardrails.validators import FailResult

    assert isinstance(result, FailResult)
    assert "0.12" in result.error_message
    assert "must be polite" in result.error_message
    assert "Text is aggressive" in result.error_message


def test_threshold_override():
    """Explicit threshold is forwarded to judge."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.60)
    validator = SemanticIntent("must be polite", threshold=0.5, judge=judge)
    validator._validate("hello", {})
    assert judge.last_threshold == 0.5


def test_default_threshold_uses_judge_recommended():
    """When no threshold given, judge's recommended_threshold applies."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    judge.recommended_threshold = 0.65
    validator = SemanticIntent("must be polite", judge=judge)
    validator._validate("hello", {})
    assert judge.last_threshold == 0.65


def test_intent_description_forwarded_to_judge():
    """The intent string becomes the judge's intent_description."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    validator = SemanticIntent("must be polite and professional", judge=judge)
    validator._validate("Thank you", {})
    assert "must be polite and professional" in judge.last_description


def test_non_string_value_is_coerced():
    """Non-string values are converted to string before validation."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    validator = SemanticIntent("must be a number", judge=judge)
    validator._validate(42, {})
    assert judge.last_output == "42"


def test_fail_result_includes_reason_when_present():
    """FailResult error_message includes judge reason when available."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=False, score=0.30, reason="Informal tone")
    validator = SemanticIntent("must be formal", judge=judge)
    result = validator._validate("hey dude", {})

    from guardrails.validators import FailResult

    assert isinstance(result, FailResult)
    assert "Informal tone" in result.error_message


def test_fail_result_no_reason():
    """FailResult works when judge returns no reason."""
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=False, score=0.20)
    validator = SemanticIntent("must be formal", judge=judge)
    result = validator._validate("hey", {})

    from guardrails.validators import FailResult

    assert isinstance(result, FailResult)
    assert "0.20" in result.error_message


def test_none_score_shows_na():
    """When score is None, error message shows N/A."""
    from semantix.integrations.guardrails import SemanticIntent
    from semantix.judges import Verdict

    judge = MockJudge(passed=False, score=0.10)
    original_evaluate = judge.evaluate

    def evaluate_none_score(output, desc, threshold=0.8):
        original_evaluate(output, desc, threshold)
        return Verdict(passed=False, score=None, reason=None)

    judge.evaluate = evaluate_none_score
    validator = SemanticIntent("must be good", judge=judge)
    result = validator._validate("bad", {})

    from guardrails.validators import FailResult

    assert isinstance(result, FailResult)
    assert "N/A" in result.error_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_guardrails.py -v 2>&1 | head -20`
Expected: ERRORS — `ModuleNotFoundError: No module named 'semantix.integrations.guardrails'`

---

### Task 2: Implement SemanticIntent validator

**Files:**
- Create: `semantix/integrations/guardrails.py`

- [ ] **Step 3: Write the implementation**

```python
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
    Validator,
    ValidationResult,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_guardrails.py -v`
Expected: All 9 tests PASS

---

### Task 3: Add optional dependency and commit

**Files:**
- Modify: `pyproject.toml:46-79`

- [ ] **Step 5: Add guardrails optional dependency**

In `pyproject.toml`, add to `[project.optional-dependencies]` after the `langchain` line:

```toml
guardrails = ["guardrails-ai>=0.5.0"]
```

And add `"guardrails-ai>=0.5.0"` to the `all` and `dev` arrays.

- [ ] **Step 6: Run full test suite and lint**

Run: `python3 -m pytest semantix/tests/ -v 2>&1 | tail -15`
Expected: All tests pass (193 existing + 9 new = 202)

Run: `ruff check semantix/integrations/guardrails.py semantix/tests/test_guardrails.py && ruff format --check semantix/integrations/guardrails.py semantix/tests/test_guardrails.py`
Expected: All checks passed

- [ ] **Step 7: Commit**

```bash
git add semantix/integrations/guardrails.py semantix/tests/test_guardrails.py pyproject.toml
git commit -m "feat: add Guardrails Hub validator integration (SemanticIntent)"
```
