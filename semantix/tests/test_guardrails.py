"""Tests for semantix.integrations.guardrails — SemanticIntent validator."""

from __future__ import annotations

from semantix.tests.conftest import MockJudge


def test_passing_validation_returns_pass_result():
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    validator = SemanticIntent("must be polite", judge=judge)
    result = validator._validate("Thank you for your patience", {})
    from guardrails.validators import PassResult

    assert isinstance(result, PassResult)


def test_failing_validation_returns_fail_result():
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
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.60)
    validator = SemanticIntent("must be polite", threshold=0.5, judge=judge)
    validator._validate("hello", {})
    assert judge.last_threshold == 0.5


def test_default_threshold_uses_judge_recommended():
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    judge.recommended_threshold = 0.65
    validator = SemanticIntent("must be polite", judge=judge)
    validator._validate("hello", {})
    assert judge.last_threshold == 0.65


def test_intent_description_forwarded_to_judge():
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    validator = SemanticIntent("must be polite and professional", judge=judge)
    validator._validate("Thank you", {})
    assert "must be polite and professional" in judge.last_description


def test_non_string_value_is_coerced():
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=True, score=0.90)
    validator = SemanticIntent("must be a number", judge=judge)
    validator._validate(42, {})
    assert judge.last_output == "42"


def test_fail_result_includes_reason_when_present():
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=False, score=0.30, reason="Informal tone")
    validator = SemanticIntent("must be formal", judge=judge)
    result = validator._validate("hey dude", {})
    from guardrails.validators import FailResult

    assert isinstance(result, FailResult)
    assert "Informal tone" in result.error_message


def test_fail_result_no_reason():
    from semantix.integrations.guardrails import SemanticIntent

    judge = MockJudge(passed=False, score=0.20)
    validator = SemanticIntent("must be formal", judge=judge)
    result = validator._validate("hey", {})
    from guardrails.validators import FailResult

    assert isinstance(result, FailResult)
    assert "0.20" in result.error_message


def test_none_score_shows_na():
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
