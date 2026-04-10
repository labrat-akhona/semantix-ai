"""Tests for Not() intent negation."""

import pytest

from semantix.composite import Not
from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""


class Polite(Intent):
    """The text must be polite and professional."""

    threshold = 0.9


def test_not_creates_negated_intent():
    """Not() creates an intent with negated description."""
    Safe = Not(MedicalAdvice)
    assert "must NOT satisfy" in Safe.description()
    assert "medical diagnoses" in Safe.description()


def test_not_class_name():
    """Negated intent has descriptive class name."""
    Safe = Not(MedicalAdvice)
    assert Safe.__name__ == "Not[MedicalAdvice]"


def test_invert_operator():
    """~MedicalAdvice is equivalent to Not(MedicalAdvice)."""
    Safe = ~MedicalAdvice
    assert "must NOT satisfy" in Safe.description()
    assert "medical diagnoses" in Safe.description()
    assert Safe.__name__ == "Not[MedicalAdvice]"


def test_not_preserves_explicit_threshold():
    """When the original intent has explicit threshold, Not preserves it."""
    NegPolite = Not(Polite)
    assert "threshold" in NegPolite.__dict__
    assert NegPolite.threshold == 0.9


def test_not_no_threshold_when_original_has_none():
    """When original has no explicit threshold, Not doesn't set one either."""
    Safe = Not(MedicalAdvice)
    assert "threshold" not in Safe.__dict__


def test_not_stores_negated_intent_reference():
    """The negated intent class is stored as _negated_intent."""
    Safe = Not(MedicalAdvice)
    assert Safe._negated_intent is MedicalAdvice


def test_not_rejects_bare_intent():
    """Cannot negate the bare Intent base class."""
    with pytest.raises(TypeError, match="bare Intent"):
        Not(Intent)


def test_not_rejects_non_intent():
    """Cannot negate a non-Intent class."""
    with pytest.raises(TypeError, match="Expected an Intent subclass"):
        Not(str)


def test_not_composable_with_allof():
    """Not can be composed with AllOf: Polite & ~MedicalAdvice."""
    from semantix.composite import AllOf

    Combined = AllOf(Polite, ~MedicalAdvice)
    desc = Combined.description()
    assert "polite and professional" in desc
    assert "must NOT satisfy" in desc


def test_not_usable_with_validate_intent():
    """Not intent works end-to-end with a mock judge."""
    from semantix.decorator import validate_intent

    Safe = ~MedicalAdvice
    judge = MockJudge(passed=True, score=0.95)

    @validate_intent(judge=judge)
    def chatbot(msg: str) -> Safe:
        return "I can't provide medical advice, please see a doctor."

    result = chatbot("my head hurts")
    assert str(result) == "I can't provide medical advice, please see a doctor."
    assert judge.call_count == 1
