"""Unit tests for semantix.eval.popia."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantix.eval.popia import EvalReport, evaluate_popia
from semantix.judges import Judge, Verdict


class ScriptedJudge(Judge):
    """Judge that returns pre-scripted verdicts keyed by (premise, hypothesis)."""

    def __init__(self, script: dict[tuple[str, str], bool]):
        self._script = script

    def evaluate(self, output: str, intent_description: str, threshold: float = 0.8) -> Verdict:
        passed = self._script.get((output, intent_description), False)
        return Verdict(passed=passed, score=0.9 if passed else 0.1, reason=None)


def _write_eval(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_perfect_popia_beats_random_stock(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "contradiction"},
        {"clause": "POPIA security safeguards", "premise": "p3", "hypothesis": "h3", "label": "entailment"},
        {"clause": "POPIA security safeguards", "premise": "p4", "hypothesis": "h4", "label": "neutral"},
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): False, ("p3", "h3"): True, ("p4", "h4"): False})
    stock = ScriptedJudge({("p1", "h1"): False, ("p2", "h2"): True, ("p3", "h3"): False, ("p4", "h4"): True})

    report = evaluate_popia(eval_path, popia, stock)
    assert report.n_pairs == 4
    assert report.popia_accuracy == 1.0
    assert report.stock_accuracy == 0.0
    assert report.delta_f1 > 0.5


def test_release_gate_requires_both_delta_and_no_per_clause_regression(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "entailment"},
        {"clause": "POPIA security safeguards", "premise": "p3", "hypothesis": "h3", "label": "entailment"},
        {"clause": "POPIA security safeguards", "premise": "p4", "hypothesis": "h4", "label": "entailment"},
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): True, ("p3", "h3"): False, ("p4", "h4"): False})
    stock = ScriptedJudge({("p1", "h1"): False, ("p2", "h2"): False, ("p3", "h3"): True, ("p4", "h4"): True})

    report = evaluate_popia(eval_path, popia, stock)
    assert report.per_clause["POPIA consent"][1] > report.per_clause["POPIA consent"][0]
    assert report.per_clause["POPIA security safeguards"][1] < report.per_clause["POPIA security safeguards"][0]
    assert report.release_gate_passed is False


def test_gate_passes_when_delta_ge_10pp_and_no_regression(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": f"p{i}", "hypothesis": f"h{i}", "label": "entailment"}
        for i in range(4)
    ] + [
        {"clause": "POPIA security safeguards", "premise": f"s{i}", "hypothesis": f"sh{i}", "label": "entailment"}
        for i in range(4)
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia_script = {(f"p{i}", f"h{i}"): True for i in range(4)} | {(f"s{i}", f"sh{i}"): True for i in range(4)}
    stock_script = {(f"p{i}", f"h{i}"): i < 3 for i in range(4)} | {(f"s{i}", f"sh{i}"): i < 2 for i in range(4)}
    popia = ScriptedJudge(popia_script)
    stock = ScriptedJudge(stock_script)

    report = evaluate_popia(eval_path, popia, stock)
    assert report.delta_f1 >= 0.10
    assert all(popia_f >= stock_f for stock_f, popia_f in report.per_clause.values())
    assert report.release_gate_passed is True


def test_gate_fails_when_delta_below_10pp(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "entailment"},
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): False})
    stock = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): False})

    report = evaluate_popia(eval_path, popia, stock)
    assert report.delta_f1 == 0.0
    assert report.release_gate_passed is False


def test_missing_eval_file_raises_filenotfound(tmp_path):
    popia = ScriptedJudge({})
    stock = ScriptedJudge({})
    with pytest.raises(FileNotFoundError):
        evaluate_popia(tmp_path / "nope.jsonl", popia, stock)
