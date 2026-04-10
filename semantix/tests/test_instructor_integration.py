"""Tests for the Instructor integration adapter."""

from __future__ import annotations

import pytest

from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


class Helpful(Intent):
    """The text must be helpful and informative."""

    threshold = 0.9


# ── semantic_validator ──────────────────────────────────────────────


def test_semantic_validator_passes():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=True, score=0.95)
    validator = semantic_validator(Polite, judge=judge)

    result = validator("Thank you for reaching out.")
    assert result == "Thank you for reaching out."
    assert judge.call_count == 1
    assert judge.last_description == Polite.description()


def test_semantic_validator_fails_raises_value_error():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=False, score=0.3, reason="Too aggressive")
    validator = semantic_validator(Polite, judge=judge)

    with pytest.raises(ValueError, match="Semantic validation failed"):
        validator("Get lost!")


def test_semantic_validator_includes_score_and_reason_in_error():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=False, score=0.42, reason="Not polite enough")
    validator = semantic_validator(Polite, judge=judge)

    with pytest.raises(ValueError, match="0.42") as exc_info:
        validator("Whatever.")
    assert "Not polite enough" in str(exc_info.value)


def test_semantic_validator_respects_intent_threshold():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=True, score=0.85)
    validator = semantic_validator(Helpful, judge=judge)

    validator("Here is the information you need.")
    assert judge.call_count == 1


def test_semantic_validator_converts_non_string():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=True, score=0.95)
    validator = semantic_validator(Polite, judge=judge)

    result = validator(42)
    assert result == 42
    assert judge.last_output == "42"


# ── SemanticStr ─────────────────────────────────────────────────────


def test_semantic_str_basic():
    from semantix.integrations.instructor import SemanticStr

    annotated_type = SemanticStr["must be polite"]
    assert hasattr(annotated_type, "__metadata__")


def test_semantic_str_with_threshold():
    from semantix.integrations.instructor import SemanticStr

    annotated_type = SemanticStr["must be polite", 0.9]
    assert hasattr(annotated_type, "__metadata__")
