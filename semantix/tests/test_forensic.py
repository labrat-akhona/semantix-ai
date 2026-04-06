"""Tests for ForensicJudge — mocked base judge + saliency, no model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from semantix.judges import Judge, Verdict
from semantix.judges.forensic import ForensicJudge, _mask_perturbation_saliency


# ---------------------------------------------------------------------------
# Unit: mask perturbation saliency
# ---------------------------------------------------------------------------


class TestMaskPerturbationSaliency:
    def test_returns_list_of_tuples(self):
        def score_fn(text):
            return 0.9 if "bad" in text else 0.2

        tokens = ["this", "is", "bad", "text"]
        result = _mask_perturbation_saliency(tokens, score_fn, top_k=3)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_identifies_high_contribution_token(self):
        def score_fn(text):
            return 0.9 if "bad" in text else 0.1

        tokens = ["this", "is", "bad", "text"]
        result = _mask_perturbation_saliency(tokens, score_fn, top_k=1)
        assert result[0][0] == "bad"

    def test_top_k_limits_output(self):
        def score_fn(text):
            return 0.5

        tokens = ["a", "b", "c", "d", "e"]
        result = _mask_perturbation_saliency(tokens, score_fn, top_k=2)
        assert len(result) <= 2

    def test_score_drop_is_positive_for_causal_token(self):
        def score_fn(text):
            return 0.9 if "toxic" in text else 0.1

        tokens = ["very", "toxic", "message"]
        result = _mask_perturbation_saliency(tokens, score_fn, top_k=3)
        toxic_entry = [t for t in result if t[0] == "toxic"]
        assert len(toxic_entry) == 1
        assert toxic_entry[0][1] > 0.5

    def test_empty_tokens_returns_empty(self):
        result = _mask_perturbation_saliency([], lambda t: 0.5, top_k=3)
        assert result == []

    def test_skips_subword_fragments(self):
        def score_fn(text):
            return 0.5

        tokens = ["hello", "##ing", "world"]
        result = _mask_perturbation_saliency(tokens, score_fn, top_k=3)
        token_names = [t[0] for t in result]
        assert "##ing" not in token_names

    def test_skips_special_tokens(self):
        def score_fn(text):
            return 0.5

        tokens = ["[CLS]", "hello", "[SEP]"]
        result = _mask_perturbation_saliency(tokens, score_fn, top_k=3)
        token_names = [t[0] for t in result]
        assert "[CLS]" not in token_names
        assert "[SEP]" not in token_names


# ---------------------------------------------------------------------------
# Integration: ForensicJudge
# ---------------------------------------------------------------------------


class _StubJudge(Judge):
    """Judge that returns a fixed verdict and tracks calls."""

    def __init__(self, passed: bool, score: float, reason: str | None = None):
        self._verdict = Verdict(passed=passed, score=score, reason=reason)
        self.call_count = 0

    def evaluate(self, output, intent_description, threshold=0.8):
        self.call_count += 1
        return self._verdict


class TestForensicJudge:
    @patch("semantix.judges.forensic._run_forensics")
    def test_passing_verdict_skips_forensics(self, mock_forensics):
        base = _StubJudge(passed=True, score=0.9)
        judge = ForensicJudge(base)
        verdict = judge.evaluate("good text", "be polite", 0.5)
        assert verdict.passed is True
        mock_forensics.assert_not_called()

    @patch("semantix.judges.forensic._run_forensics")
    def test_failing_verdict_triggers_forensics(self, mock_forensics):
        mock_forensics.return_value = [("bruh", 0.7), ("whatever", 0.5)]
        base = _StubJudge(passed=False, score=0.2)
        judge = ForensicJudge(base)
        verdict = judge.evaluate("bruh whatever", "be polite", 0.5)
        assert verdict.passed is False
        mock_forensics.assert_called_once()

    @patch("semantix.judges.forensic._run_forensics")
    def test_breach_report_in_reason(self, mock_forensics):
        mock_forensics.return_value = [("bruh", 0.72), ("whatever", 0.51)]
        base = _StubJudge(passed=False, score=0.2)
        judge = ForensicJudge(base)
        verdict = judge.evaluate("bruh whatever", "be polite", 0.5)
        assert "Breach Report" in verdict.reason
        assert "bruh" in verdict.reason
        assert "whatever" in verdict.reason

    @patch("semantix.judges.forensic._run_forensics")
    def test_breach_report_contains_score(self, mock_forensics):
        mock_forensics.return_value = [("bruh", 0.72)]
        base = _StubJudge(passed=False, score=0.15)
        judge = ForensicJudge(base)
        verdict = judge.evaluate("bruh", "be polite", 0.5)
        assert "0.15" in verdict.reason

    @patch("semantix.judges.forensic._run_forensics")
    def test_original_score_preserved(self, mock_forensics):
        mock_forensics.return_value = []
        base = _StubJudge(passed=False, score=0.23)
        judge = ForensicJudge(base)
        verdict = judge.evaluate("text", "intent", 0.5)
        assert verdict.score == 0.23

    @patch("semantix.judges.forensic._run_forensics")
    def test_preserves_base_reason(self, mock_forensics):
        mock_forensics.return_value = [("bad", 0.6)]
        base = _StubJudge(passed=False, score=0.2, reason="entailment low")
        judge = ForensicJudge(base)
        verdict = judge.evaluate("bad text", "be polite", 0.5)
        assert "entailment low" in verdict.reason

    @patch("semantix.judges.forensic._run_forensics")
    def test_empty_breach_tokens_still_has_report(self, mock_forensics):
        mock_forensics.return_value = []
        base = _StubJudge(passed=False, score=0.2)
        judge = ForensicJudge(base)
        verdict = judge.evaluate("text", "intent", 0.5)
        assert "Breach Report" in verdict.reason

    def test_wraps_any_judge_subclass(self):
        base = _StubJudge(passed=True, score=0.9)
        judge = ForensicJudge(base)
        assert isinstance(judge, Judge)
