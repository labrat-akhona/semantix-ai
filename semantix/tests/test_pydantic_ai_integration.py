"""Tests for the Pydantic AI integration adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


# ── semantix_validator ──────────────────────────────────────────────


def test_semantix_validator_passes_string_output():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=True, score=0.95)
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()
    result = validator_fn(ctx, "Thank you for your patience.")
    assert result == "Thank you for your patience."
    assert judge.call_count == 1


def test_semantix_validator_fails_raises_model_retry():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=False, score=0.3, reason="Too rude")
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()

    with pytest.raises(Exception, match="Semantic validation failed"):
        validator_fn(ctx, "Get lost!")


def test_semantix_validator_includes_score_in_retry_message():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=False, score=0.42, reason="Not professional")
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()
    with pytest.raises(Exception, match="0.42"):
        validator_fn(ctx, "Whatever.")


def test_semantix_validator_handles_non_string_output():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=True, score=0.95)
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()
    result = validator_fn(ctx, 42)
    assert result == 42
    assert judge.last_output == "42"


def test_semantix_validator_with_judge_from_deps():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=True, score=0.95)
    validator_fn = semantix_validator(Polite, judge_from_deps=True)

    ctx = MagicMock()
    ctx.deps = judge
    result = validator_fn(ctx, "Thank you.")
    assert result == "Thank you."
    assert judge.call_count == 1
