"""Tests for composite intents (AllOf / AnyOf) and & / | operators."""

import pytest

from semantix.composite import AllOf, AnyOf
from semantix.decorator import validate_intent
from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite."""


class Positive(Intent):
    """The text must be positive."""

    threshold = 0.9


class Formal(Intent):
    """The text must be formal."""

    threshold = 0.7


# ── AllOf ───────────────────────────────────────────────────────────────


def test_allof_creates_combined_class():
    Combined = AllOf(Polite, Positive)
    assert issubclass(Combined, Intent)
    desc = Combined.description()
    assert "polite" in desc.lower()
    assert "positive" in desc.lower()
    assert "AND" in desc


def test_allof_threshold_is_min():
    Combined = AllOf(Polite, Positive, Formal)
    assert Combined.threshold == 0.7  # min(0.8, 0.9, 0.7)


def test_allof_needs_two():
    with pytest.raises(TypeError):
        AllOf(Polite)


# ── AnyOf ───────────────────────────────────────────────────────────────


def test_anyof_creates_combined_class():
    Combined = AnyOf(Polite, Positive)
    assert issubclass(Combined, Intent)
    desc = Combined.description()
    assert "OR" in desc


def test_anyof_threshold_is_max():
    Combined = AnyOf(Polite, Positive, Formal)
    assert Combined.threshold == 0.9  # max(0.8, 0.9, 0.7)


# ── operator syntax ────────────────────────────────────────────────────


def test_and_operator():
    Combined = Polite & Positive
    assert issubclass(Combined, Intent)
    assert "AND" in Combined.description()


def test_or_operator():
    Combined = Polite | Positive
    assert issubclass(Combined, Intent)
    assert "OR" in Combined.description()


# ── integration with decorator ──────────────────────────────────────────


def test_composite_intent_with_decorator():
    judge = MockJudge(passed=True, score=0.95)
    Combined = Polite & Positive

    @validate_intent(judge=judge)
    def respond(msg: str) -> Combined:  # type: ignore[valid-type]
        return "Thank you, that's wonderful!"

    result = respond("hello")
    assert isinstance(result, Combined)
    assert judge.call_count == 1
