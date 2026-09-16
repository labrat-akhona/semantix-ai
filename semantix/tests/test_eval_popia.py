"""Unit tests for semantix.eval.popia."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantix.eval.popia import GATE_THRESHOLD, evaluate_popia, evaluate_popia_matrix
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
        {
            "clause": "POPIA security safeguards",
            "premise": "p3",
            "hypothesis": "h3",
            "label": "entailment",
        },
        {
            "clause": "POPIA security safeguards",
            "premise": "p4",
            "hypothesis": "h4",
            "label": "neutral",
        },
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge(
        {("p1", "h1"): True, ("p2", "h2"): False, ("p3", "h3"): True, ("p4", "h4"): False}
    )
    stock = ScriptedJudge(
        {("p1", "h1"): False, ("p2", "h2"): True, ("p3", "h3"): False, ("p4", "h4"): True}
    )

    report = evaluate_popia(eval_path, popia, stock)
    assert report.n_pairs == 4
    assert report.popia_accuracy == 1.0
    assert report.stock_accuracy == 0.0
    assert report.delta_f1 > 0.5


def test_release_gate_requires_both_delta_and_no_per_clause_regression(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "entailment"},
        {
            "clause": "POPIA security safeguards",
            "premise": "p3",
            "hypothesis": "h3",
            "label": "entailment",
        },
        {
            "clause": "POPIA security safeguards",
            "premise": "p4",
            "hypothesis": "h4",
            "label": "entailment",
        },
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge(
        {("p1", "h1"): True, ("p2", "h2"): True, ("p3", "h3"): False, ("p4", "h4"): False}
    )
    stock = ScriptedJudge(
        {("p1", "h1"): False, ("p2", "h2"): False, ("p3", "h3"): True, ("p4", "h4"): True}
    )

    report = evaluate_popia(eval_path, popia, stock)
    assert report.per_clause["POPIA consent"][1] > report.per_clause["POPIA consent"][0]
    assert (
        report.per_clause["POPIA security safeguards"][1]
        < report.per_clause["POPIA security safeguards"][0]
    )
    assert report.release_gate_passed is False


def test_gate_passes_when_delta_ge_10pp_and_no_regression(tmp_path):
    rows = [
        {
            "clause": "POPIA consent",
            "premise": f"p{i}",
            "hypothesis": f"h{i}",
            "label": "entailment",
        }
        for i in range(4)
    ] + [
        {
            "clause": "POPIA security safeguards",
            "premise": f"s{i}",
            "hypothesis": f"sh{i}",
            "label": "entailment",
        }
        for i in range(4)
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia_script = {(f"p{i}", f"h{i}"): True for i in range(4)} | {
        (f"s{i}", f"sh{i}"): True for i in range(4)
    }
    stock_script = {(f"p{i}", f"h{i}"): i < 3 for i in range(4)} | {
        (f"s{i}", f"sh{i}"): i < 2 for i in range(4)
    }
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


class RecordingJudge(Judge):
    """Passes everything and records the threshold it was called with."""

    def __init__(self) -> None:
        self.thresholds: list[float] = []

    def evaluate(self, output: str, intent_description: str, threshold: float = 0.8) -> Verdict:
        self.thresholds.append(threshold)
        return Verdict(passed=True, score=0.9)


class ConstantJudge(Judge):
    def __init__(self, passed: bool) -> None:
        self._passed = passed

    def evaluate(self, output: str, intent_description: str, threshold: float = 0.8) -> Verdict:
        return Verdict(passed=self._passed, score=0.9 if self._passed else 0.1)


_TWO_ROWS = [
    {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
    {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "neutral"},
]


def _entailment_rows():
    # A judge that passes everything scores macro-F1 1.0 here; one that fails everything, 0.0.
    return [
        {
            "clause": "POPIA consent",
            "premise": f"p{i}",
            "hypothesis": f"h{i}",
            "label": "entailment",
        }
        for i in range(4)
    ]


def test_gate_passes_its_threshold_explicitly(tmp_path):
    popia, stock = RecordingJudge(), RecordingJudge()
    report = evaluate_popia(_write_eval(tmp_path, _TWO_ROWS), popia, stock)
    assert popia.thresholds == [GATE_THRESHOLD, GATE_THRESHOLD]
    assert stock.thresholds == [GATE_THRESHOLD, GATE_THRESHOLD]
    assert report.threshold == GATE_THRESHOLD == 0.5


def test_custom_threshold_is_used_and_recorded(tmp_path):
    popia, stock = RecordingJudge(), RecordingJudge()
    report = evaluate_popia(_write_eval(tmp_path, _TWO_ROWS), popia, stock, threshold=0.75)
    assert set(popia.thresholds) == {0.75}
    assert report.threshold == 0.75


def test_report_as_dict_is_json_serialisable(tmp_path):
    report = evaluate_popia(_write_eval(tmp_path, _TWO_ROWS), RecordingJudge(), RecordingJudge())
    data = json.loads(json.dumps(report.as_dict()))
    assert data["per_clause"]["POPIA consent"] == list(report.per_clause["POPIA consent"])


def test_matrix_fails_when_any_file_fails(tmp_path):
    eval_path = _write_eval(tmp_path, _entailment_rows())
    good = (ConstantJudge(True), ConstantJudge(False))  # POPIA right, stock wrong
    bad = (ConstantJudge(False), ConstantJudge(True))  # POPIA worse than stock
    loaded: list[str] = []

    def load(variant):
        loaded.append(variant)
        return bad if variant == "onnx/avx2.onnx" else good

    matrix = evaluate_popia_matrix(
        eval_path, load, variants=["onnx/avx2.onnx", "onnx/vnni.onnx"], runtime={"os": "test"}
    )
    assert loaded == ["onnx/avx2.onnx", "onnx/vnni.onnx"]
    assert matrix.results["onnx/vnni.onnx"].release_gate_passed is True
    assert matrix.results["onnx/avx2.onnx"].regressed_clauses == ["POPIA consent"]
    assert matrix.failing == ["onnx/avx2.onnx"]
    assert matrix.all_passed is False
    assert matrix.runtime == {"os": "test"}


def test_matrix_passes_when_every_file_passes(tmp_path):
    matrix = evaluate_popia_matrix(
        _write_eval(tmp_path, _entailment_rows()),
        lambda variant: (ConstantJudge(True), ConstantJudge(False)),
        variants=["a.onnx", "b.onnx"],
        runtime={},
    )
    assert matrix.all_passed is True
    assert matrix.failing == []


def test_matrix_with_no_files_does_not_pass(tmp_path):
    matrix = evaluate_popia_matrix(
        _write_eval(tmp_path, _entailment_rows()), lambda v: None, variants=[], runtime={}
    )
    assert matrix.all_passed is False


def test_matrix_checks_eval_file_before_loading_models(tmp_path):
    def load(variant):
        raise AssertionError("models must not load when the eval file is missing")

    with pytest.raises(FileNotFoundError):
        evaluate_popia_matrix(tmp_path / "nope.jsonl", load, variants=["a.onnx"], runtime={})


def test_matrix_as_dict_is_json_serialisable(tmp_path):
    matrix = evaluate_popia_matrix(
        _write_eval(tmp_path, _entailment_rows()),
        lambda v: (ConstantJudge(True), ConstantJudge(False)),
        variants=["a.onnx"],
        runtime={"onnxruntime": "1.0"},
    )
    data = json.loads(json.dumps(matrix.as_dict()))
    assert data["all_passed"] is True
    assert data["results"]["a.onnx"]["threshold"] == 0.5
    assert data["runtime"] == {"onnxruntime": "1.0"}


def test_as_dict_reports_the_regressed_clauses(tmp_path):
    rows = [
        {"clause": "consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "minimality", "premise": "p2", "hypothesis": "h2", "label": "entailment"},
    ]
    report = evaluate_popia(_write_eval(tmp_path, rows), ConstantJudge(False), ConstantJudge(True))
    data = json.loads(json.dumps(report.as_dict()))
    assert data["regressed_clauses"] == ["consent", "minimality"]
    assert data["regressed_clauses"] == report.regressed_clauses


def test_load_judges_for_variant_builds_one_popia_and_one_stock_judge(monkeypatch):
    # Shared by semantix.cli and scripts/eval_popia.py so both gate the same way.
    from semantix.eval.popia import load_judges_for_variant

    built: list[tuple[str, str]] = []

    class FakePOPIA:
        def __init__(self, model_variant=None):
            built.append(("popia", model_variant))

    class FakeStock:
        def __init__(self, model_variant=None):
            built.append(("stock", model_variant))

    monkeypatch.setattr("semantix.judges.popia.POPIAJudge", FakePOPIA)
    monkeypatch.setattr("semantix.judges.quantized_nli.QuantizedNLIJudge", FakeStock)

    popia, stock = load_judges_for_variant("onnx/model_qint8_arm64.onnx")
    assert isinstance(popia, FakePOPIA) and isinstance(stock, FakeStock)
    assert built == [
        ("popia", "onnx/model_qint8_arm64.onnx"),
        ("stock", "onnx/model_qint8_arm64.onnx"),
    ]
