"""Tests for the DSPy integration — semantic_reward and semantic_metric."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from semantix.intent import Intent
from semantix.judges import Judge, Verdict


class Polite(Intent):
    """The text must be polite and professional."""


class _FakeJudge(Judge):
    """Returns a configurable verdict for testing."""

    recommended_threshold = 0.5

    def __init__(self, score: float = 0.9, passed: bool = True, reason: str = "") -> None:
        self._score = score
        self._passed = passed
        self._reason = reason

    def evaluate(self, text: str, description: str, threshold: float) -> Verdict:
        return Verdict(passed=self._passed, score=self._score, reason=self._reason)


class _FakePrediction:
    """Mimics a dspy.Prediction with attribute and key access."""

    def __init__(self, **kwargs: str) -> None:
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, key: str) -> str:
        return self._data[key]


# ---------- semantic_reward ----------


def test_reward_returns_score():
    from semantix.integrations.dspy import semantic_reward

    fn = semantic_reward(Polite, judge=_FakeJudge(score=0.85))
    result = fn({}, _FakePrediction(answer="Thank you"))
    assert result == pytest.approx(0.85)


def test_reward_with_string_intent():
    from semantix.integrations.dspy import semantic_reward

    fn = semantic_reward("must be polite", judge=_FakeJudge(score=0.7))
    result = fn({}, _FakePrediction(answer="Hello"))
    assert result == pytest.approx(0.7)


def test_reward_reads_specified_field():
    from semantix.integrations.dspy import semantic_reward

    judge = _FakeJudge(score=0.9)
    judge.evaluate = MagicMock(return_value=Verdict(passed=True, score=0.9, reason=""))
    fn = semantic_reward(Polite, field="reply", judge=judge)
    fn({}, _FakePrediction(answer="wrong", reply="correct"))
    judge.evaluate.assert_called_once()
    assert judge.evaluate.call_args[0][0] == "correct"


def test_reward_uses_last_field_by_default():
    from semantix.integrations.dspy import semantic_reward

    judge = _FakeJudge(score=0.9)
    judge.evaluate = MagicMock(return_value=Verdict(passed=True, score=0.9, reason=""))
    fn = semantic_reward(Polite, judge=judge)
    fn({}, _FakePrediction(reasoning="step by step", answer="final"))
    judge.evaluate.assert_called_once()
    assert judge.evaluate.call_args[0][0] == "final"


def test_reward_fallback_when_no_score():
    from semantix.integrations.dspy import semantic_reward

    fn = semantic_reward(Polite, judge=_FakeJudge(score=None, passed=True))
    # score=None but passed → 1.0
    result = fn({}, _FakePrediction(answer="ok"))
    assert result == 1.0


def test_reward_fallback_when_no_score_failed():
    from semantix.integrations.dspy import semantic_reward

    fn = semantic_reward(Polite, judge=_FakeJudge(score=None, passed=False))
    result = fn({}, _FakePrediction(answer="bad"))
    assert result == 0.0


def test_reward_uses_explicit_threshold():
    from semantix.integrations.dspy import semantic_reward

    judge = _FakeJudge(score=0.9)
    judge.evaluate = MagicMock(return_value=Verdict(passed=True, score=0.9, reason=""))
    fn = semantic_reward(Polite, judge=judge, threshold=0.95)
    fn({}, _FakePrediction(answer="hi"))
    assert judge.evaluate.call_args[0][2] == 0.95


def test_reward_uses_judge_recommended_threshold():
    from semantix.integrations.dspy import semantic_reward

    judge = _FakeJudge(score=0.9)
    judge.recommended_threshold = 0.6
    judge.evaluate = MagicMock(return_value=Verdict(passed=True, score=0.9, reason=""))
    fn = semantic_reward(Polite, judge=judge)
    fn({}, _FakePrediction(answer="hi"))
    # Polite has no explicit threshold in __dict__, so judge's 0.6 should be used
    assert judge.evaluate.call_args[0][2] == 0.6


# ---------- semantic_metric ----------


def test_metric_returns_score():
    from semantix.integrations.dspy import semantic_metric

    fn = semantic_metric(Polite, judge=_FakeJudge(score=0.75))
    example = _FakePrediction(question="How are you?")
    pred = _FakePrediction(answer="Fine, thanks!")
    assert fn(example, pred) == pytest.approx(0.75)


def test_metric_with_simple_namespace_example():
    from semantix.integrations.dspy import semantic_metric

    fn = semantic_metric(Polite, judge=_FakeJudge(score=0.8))
    example = SimpleNamespace(question="test")
    pred = _FakePrediction(answer="response")
    assert fn(example, pred) == pytest.approx(0.8)


def test_metric_accepts_extra_args():
    """DSPy sometimes passes extra kwargs like trace to metrics."""
    from semantix.integrations.dspy import semantic_metric

    fn = semantic_metric(Polite, judge=_FakeJudge(score=0.9))
    example = _FakePrediction(question="test")
    pred = _FakePrediction(answer="response")
    # Should not raise even with extra positional/keyword args
    result = fn(example, pred, "extra", trace=None)
    assert result == pytest.approx(0.9)
