"""Unit tests for `semantix eval popia`."""

from __future__ import annotations

import json

from semantix.cli import main as cli_main
from semantix.eval.popia import EvalReport


def _fake_report(gate: bool, delta: float = 0.15) -> EvalReport:
    return EvalReport(
        n_pairs=150,
        stock_accuracy=0.62,
        stock_f1_macro=0.59,
        popia_accuracy=0.78,
        popia_f1_macro=0.59 + delta,
        per_clause={
            "POPIA consent": (0.60, 0.75),
            "POPIA cross-border transfers": (0.55, 0.82),
        },
        delta_f1=delta,
        release_gate_passed=gate,
    )


def test_eval_popia_exits_zero_when_gate_passes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(True))

    rc = cli_main(["eval", "popia"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_eval_popia_exits_one_when_gate_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(False, delta=0.05))

    rc = cli_main(["eval", "popia"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_eval_popia_json_flag_emits_valid_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(True))

    rc = cli_main(["eval", "popia", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["release_gate_passed"] is True
    assert data["n_pairs"] == 150
    assert data["delta_f1"] == 0.15


def test_eval_popia_download_failure_exits_two(capsys, monkeypatch):
    def boom():
        raise FileNotFoundError("HF unreachable")
    monkeypatch.setattr("semantix.cli._download_popia_eval", boom)

    rc = cli_main(["eval", "popia"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unreachable" in err.lower() or "not found" in err.lower()
