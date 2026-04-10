"""Tests for the LangChain integration adapter."""

from __future__ import annotations

import pytest

from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


# ── SemanticValidator ───────────────────────────────────────────────


def test_semantic_validator_invoke_passes():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=True, score=0.95)
    validator = SemanticValidator(Polite, judge=judge)

    result = validator.invoke("Thank you for your patience.")
    assert result == "Thank you for your patience."
    assert judge.call_count == 1


def test_semantic_validator_invoke_fails():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=False, score=0.3, reason="Too rude")
    validator = SemanticValidator(Polite, judge=judge)

    with pytest.raises(Exception, match="Semantic validation failed"):
        validator.invoke("Get lost!")


def test_semantic_validator_error_includes_score():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=False, score=0.42, reason="Not polite")
    validator = SemanticValidator(Polite, judge=judge)

    with pytest.raises(Exception, match="0.42"):
        validator.invoke("Whatever.")


def test_semantic_validator_handles_non_string():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=True, score=0.95)
    validator = SemanticValidator(Polite, judge=judge)

    result = validator.invoke(42)
    assert result == 42
    assert judge.last_output == "42"


def test_semantic_validator_batch():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=True, score=0.95)
    validator = SemanticValidator(Polite, judge=judge)

    results = validator.batch(["Hello.", "Thank you."])
    assert results == ["Hello.", "Thank you."]
    assert judge.call_count == 2
