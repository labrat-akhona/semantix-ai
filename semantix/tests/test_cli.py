"""Tests for the semantix CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from semantix.cli import _build_parser, _resolve_threshold, _run_check, main

# ---------------------------------------------------------------------------
# _resolve_threshold
# ---------------------------------------------------------------------------

class TestResolveThreshold:
    def test_explicit_wins(self):
        judge = MagicMock(recommended_threshold=0.6)
        assert _resolve_threshold(0.9, judge) == 0.9

    def test_judge_recommended(self):
        judge = MagicMock(recommended_threshold=0.75)
        assert _resolve_threshold(None, judge) == 0.75

    def test_fallback_default(self):
        judge = MagicMock(recommended_threshold=None)
        assert _resolve_threshold(None, judge) == 0.8


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_check_basic(self):
        parser = _build_parser()
        args = parser.parse_args(["check", "hello", "--intent", "polite"])
        assert args.command == "check"
        assert args.text == "hello"
        assert args.intent == "polite"
        assert args.threshold is None
        assert args.judge is None
        assert args.negate is False

    def test_check_all_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "check", "hello",
            "--intent", "polite",
            "--threshold", "0.85",
            "--judge", "nli",
            "--negate",
        ])
        assert args.threshold == 0.85
        assert args.judge == "nli"
        assert args.negate is True


# ---------------------------------------------------------------------------
# _run_check
# ---------------------------------------------------------------------------

class TestRunCheck:
    def _make_args(self, **overrides):
        defaults = {
            "text": "hello world",
            "intent": "polite",
            "threshold": None,
            "judge": None,
            "negate": False,
        }
        defaults.update(overrides)
        return MagicMock(**defaults)

    @patch("semantix.cli._resolve_judge")
    def test_pass_returns_zero(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=True, score=0.92, reason=None)
        mock_resolve.return_value = judge

        code = _run_check(self._make_args())
        assert code == 0
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "0.9200" in out

    @patch("semantix.cli._resolve_judge")
    def test_fail_returns_one(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=False, score=0.3, reason="not polite")
        mock_resolve.return_value = judge

        code = _run_check(self._make_args())
        assert code == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "not polite" in out

    @patch("semantix.cli._resolve_judge")
    def test_negate_inverts(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=True, score=0.95, reason=None)
        mock_resolve.return_value = judge

        # With negate, a passing verdict becomes FAIL
        code = _run_check(self._make_args(negate=True))
        assert code == 1

    @patch("semantix.cli._resolve_judge")
    def test_negate_fail_becomes_pass(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=False, score=0.2, reason=None)
        mock_resolve.return_value = judge

        code = _run_check(self._make_args(negate=True))
        assert code == 0

    @patch("semantix.cli._resolve_judge")
    def test_explicit_threshold_forwarded(self, mock_resolve):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=True, score=0.9)
        mock_resolve.return_value = judge

        _run_check(self._make_args(threshold=0.85))
        judge.evaluate.assert_called_once_with("hello world", "polite", threshold=0.85)

    @patch("semantix.cli._resolve_judge")
    def test_score_none_displays_na(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=True, score=None)
        mock_resolve.return_value = judge

        _run_check(self._make_args())
        out = capsys.readouterr().out
        assert "n/a" in out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_command_prints_help(self, capsys):
        code = main([])
        assert code == 0

    @patch("semantix.cli._run_check", return_value=0)
    def test_check_dispatches(self, mock_run):
        code = main(["check", "hi", "--intent", "greeting"])
        assert code == 0
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_judge
# ---------------------------------------------------------------------------

class TestResolveJudge:
    @patch("semantix.cli._default_judge")
    def test_none_uses_default(self, mock_default):
        from semantix.cli import _resolve_judge

        _resolve_judge(None)
        mock_default.assert_called_once()

    @patch("semantix.judges.nli.NLIJudge")
    def test_nli(self, mock_cls):
        from semantix.cli import _resolve_judge

        _resolve_judge("nli")
        mock_cls.assert_called_once()

    @patch("semantix.judges.embedding.EmbeddingJudge")
    def test_embedding(self, mock_cls):
        from semantix.cli import _resolve_judge

        _resolve_judge("embedding")
        mock_cls.assert_called_once()

    def test_unknown_exits(self):
        from semantix.cli import _resolve_judge

        with pytest.raises(SystemExit) as exc_info:
            _resolve_judge("bogus")
        assert exc_info.value.code == 2
