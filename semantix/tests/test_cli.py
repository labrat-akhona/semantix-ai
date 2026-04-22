"""Tests for the semantix CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from semantix.cli import (
    _build_parser,
    _load_audit_entries,
    _percentile,
    _resolve_threshold,
    _run_check,
    _run_demo,
    _run_prove,
    _run_verify,
    _verify_chain,
    main,
)

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
        args = parser.parse_args(
            [
                "check",
                "hello",
                "--intent",
                "polite",
                "--threshold",
                "0.85",
                "--judge",
                "nli",
                "--negate",
            ]
        )
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


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([5.0], 99) == 5.0

    def test_interpolation(self):
        # p50 of [0, 10] is 5.0
        assert _percentile([0.0, 10.0], 50) == 5.0

    def test_p99_of_uniform(self):
        vals = sorted(float(x) for x in range(1, 101))
        # p99 is near 99.0
        assert abs(_percentile(vals, 99) - 99.01) < 0.1


class TestRunProve:
    def _make_args(self, **overrides):
        defaults = {
            "text": "hello",
            "intent": "greeting",
            "n": 10,
            "threshold": None,
            "judge": None,
            "no_color": True,  # tests expect plain output
        }
        defaults.update(overrides)
        return MagicMock(**defaults)

    @patch("semantix.cli._resolve_judge")
    def test_deterministic_returns_zero(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=True, score=0.87654, reason=None)
        mock_resolve.return_value = judge

        code = _run_prove(self._make_args(n=25))
        assert code == 0
        out = capsys.readouterr().out
        assert "DETERMINISM VERIFIED" in out
        assert "25/25" in out
        assert "0.87654" in out

    @patch("semantix.cli._resolve_judge")
    def test_non_deterministic_returns_one(self, mock_resolve, capsys):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        # Alternate between two scores to force non-determinism.
        judge.evaluate.side_effect = [
            Verdict(passed=True, score=0.90, reason=None),
            Verdict(passed=True, score=0.91, reason=None),
        ] * 5
        mock_resolve.return_value = judge

        code = _run_prove(self._make_args(n=10))
        assert code == 1
        out = capsys.readouterr().out
        assert "NON-DETERMINISTIC" in out
        assert "2 distinct scores" in out

    @patch("semantix.cli._resolve_judge")
    def test_forwards_threshold(self, mock_resolve):
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = None
        judge.evaluate.return_value = Verdict(passed=True, score=0.9, reason=None)
        mock_resolve.return_value = judge

        _run_prove(self._make_args(n=3, threshold=0.7))
        # Each of the 3 calls should use threshold=0.7.
        for call in judge.evaluate.call_args_list:
            assert call.kwargs.get("threshold") == 0.7

    def test_parser_prove_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["prove"])
        assert args.command == "prove"
        assert args.n == 100
        assert args.text  # has a default
        assert args.intent  # has a default

    def test_parser_prove_custom(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "prove",
                "--text",
                "x",
                "--intent",
                "y",
                "--n",
                "5",
                "--judge",
                "nli",
                "--threshold",
                "0.5",
                "--no-color",
            ]
        )
        assert args.text == "x"
        assert args.intent == "y"
        assert args.n == 5
        assert args.judge == "nli"
        assert args.threshold == 0.5
        assert args.no_color is True


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


# ---------------------------------------------------------------------------
# verify subcommand
# ---------------------------------------------------------------------------


def _write_audit(path, *, tamper_index: int | None = None, tamper_field: str = "score"):
    """Helper: write a valid 4-entry hash-chained trail, optionally tampered."""
    import hashlib
    import json
    from datetime import datetime, timezone

    entries = []
    prev = "GENESIS"
    records = [
        ("Polite", True, 0.92),
        ("NoMedicalAdvice", True, 0.88),
        ("Polite", False, 0.12),
        ("Polite", True, 0.85),
    ]
    for i, (intent, passed, score) in enumerate(records):
        cert = {
            "@context": "https://schema.semantix.ai/v1",
            "@type": "SemanticCertificate",
            "id": f"urn:semantix:cert:test-{i}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "score": score,
            "passed": passed,
            "reason": None,
            "output_hash": hashlib.sha256(f"out{i}".encode()).hexdigest(),
            "previous_hash": prev,
        }
        entries.append(cert)
        prev = hashlib.sha256(json.dumps(cert, sort_keys=True).encode()).hexdigest()

    if tamper_index is not None:
        entries[tamper_index][tamper_field] = 0.999

    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    return entries


class TestLoadAuditEntries:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_audit_entries(str(tmp_path / "nope.jsonl"))

    def test_malformed_raises(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{not json\n")
        with pytest.raises(ValueError) as exc:
            _load_audit_entries(str(p))
        assert "line 1" in str(exc.value)

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "ok.jsonl"
        p.write_text('{"a":1}\n\n{"a":2}\n')
        entries = _load_audit_entries(str(p))
        assert entries == [{"a": 1}, {"a": 2}]


class TestVerifyChain:
    def test_valid_chain(self, tmp_path):
        entries = _write_audit(tmp_path / "audit.jsonl")
        ok, broken = _verify_chain(entries)
        assert ok is True
        assert broken is None

    def test_tamper_detected(self, tmp_path):
        entries = _write_audit(tmp_path / "audit.jsonl", tamper_index=1)
        ok, broken = _verify_chain(entries)
        assert ok is False
        # Tampering entry 1 invalidates entry 2's previous_hash.
        assert broken == 2

    def test_empty_list_is_valid(self):
        ok, broken = _verify_chain([])
        assert ok is True
        assert broken is None


class TestRunVerify:
    def _make_args(self, path, **overrides):
        defaults = {"path": str(path), "top": 5, "no_color": True}
        defaults.update(overrides)
        return MagicMock(**defaults)

    def test_valid_trail_returns_zero(self, tmp_path, capsys):
        p = tmp_path / "audit.jsonl"
        _write_audit(p)
        code = _run_verify(self._make_args(p))
        assert code == 0
        out = capsys.readouterr().out
        assert "CHAIN VERIFIED" in out
        assert "4/4" in out
        assert "Polite" in out

    def test_tampered_trail_returns_one(self, tmp_path, capsys):
        p = tmp_path / "audit.jsonl"
        _write_audit(p, tamper_index=1)
        code = _run_verify(self._make_args(p))
        assert code == 1
        out = capsys.readouterr().out
        assert "CHAIN BROKEN" in out
        assert "#2" in out

    def test_missing_file_returns_two(self, tmp_path, capsys):
        code = _run_verify(self._make_args(tmp_path / "nope.jsonl"))
        assert code == 2
        err = capsys.readouterr().err
        assert "ERROR" in err

    def test_empty_file_returns_zero(self, tmp_path, capsys):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        code = _run_verify(self._make_args(p))
        assert code == 0
        out = capsys.readouterr().out
        assert "no entries" in out

    def test_top_flag_limits_intents(self, tmp_path, capsys):
        p = tmp_path / "audit.jsonl"
        _write_audit(p)
        code = _run_verify(self._make_args(p, top=1))
        assert code == 0
        out = capsys.readouterr().out
        # "Polite" appears 3x, should be the only one shown.
        assert "Polite" in out
        assert "NoMedicalAdvice" not in out.split("top 1")[1].split("CHAIN")[0]

    def test_parser_verify(self):
        parser = _build_parser()
        args = parser.parse_args(["verify", "/tmp/x.jsonl", "--top", "3", "--no-color"])
        assert args.command == "verify"
        assert args.path == "/tmp/x.jsonl"
        assert args.top == 3
        assert args.no_color is True


# ---------------------------------------------------------------------------
# demo subcommand
# ---------------------------------------------------------------------------


class TestRunDemo:
    def _make_args(self, **overrides):
        defaults = {"judge": None, "no_color": True}
        defaults.update(overrides)
        return MagicMock(**defaults)

    @patch("semantix.cli._resolve_judge")
    def test_all_expected_returns_zero(self, mock_resolve, capsys):
        from semantix.cli import _DEMO_SCENARIOS
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = 0.3
        # Tailor return values so each scenario hits its expected verdict.
        # warmup + 3 scenarios = 4 evaluate() calls.
        scenario_scores = []
        for s in _DEMO_SCENARIOS:
            if s["expect"] == "PASS" and not s["negate"]:
                scenario_scores.append(Verdict(passed=True, score=0.9))
            elif s["expect"] == "FAIL" and not s["negate"]:
                scenario_scores.append(Verdict(passed=False, score=0.05, reason="nope"))
            else:  # negated FAIL: raw check must pass (so negation flips it)
                scenario_scores.append(Verdict(passed=True, score=0.7, reason="entailed"))
        judge.evaluate.side_effect = [Verdict(passed=True, score=1.0)] + scenario_scores
        mock_resolve.return_value = judge

        code = _run_demo(self._make_args())
        assert code == 0
        out = capsys.readouterr().out
        assert "Semantix demo" in out
        assert "PASS" in out
        assert "FAIL" in out
        assert "0 API calls" in out

    @patch("semantix.cli._resolve_judge")
    def test_unexpected_result_returns_one(self, mock_resolve, capsys):
        from semantix.cli import _DEMO_SCENARIOS
        from semantix.judges import Verdict

        judge = MagicMock()
        judge.recommended_threshold = 0.3
        # Force all scenarios to PASS regardless of expected verdict.
        judge.evaluate.side_effect = [Verdict(passed=True, score=0.9)] * (1 + len(_DEMO_SCENARIOS))
        mock_resolve.return_value = judge

        code = _run_demo(self._make_args())
        # Some scenarios expect FAIL, so all-PASS is wrong.
        assert code == 1

    def test_parser_demo(self):
        parser = _build_parser()
        args = parser.parse_args(["demo", "--judge", "nli", "--no-color"])
        assert args.command == "demo"
        assert args.judge == "nli"
        assert args.no_color is True
