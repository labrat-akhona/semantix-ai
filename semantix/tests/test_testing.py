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
