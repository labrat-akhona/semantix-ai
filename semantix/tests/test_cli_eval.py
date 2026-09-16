"""Unit tests for `semantix eval popia`."""

from __future__ import annotations

import json

from semantix.cli import main as cli_main
from semantix.eval.popia import EvalReport, MatrixReport


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
    monkeypatch.setattr(
        "semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(False, delta=0.05)
    )

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


_AVX2 = "onnx/model_quint8_avx2.onnx"
_VNNI = "onnx/model_qint8_avx512_vnni.onnx"


def _fake_matrix(failing_variant: str | None) -> MatrixReport:
    results = {
        v: _fake_report(v != failing_variant, delta=0.15 if v != failing_variant else -0.02)
        for v in (_AVX2, _VNNI)
    }
    runtime = {
        "onnxruntime": "1.24.4",
        "os": "Linux",
        "machine": "x86_64",
        "cpu_flags": "avx2",
        "auto_variant": _AVX2,
    }
    return MatrixReport(threshold=0.5, runtime=runtime, results=results)


def _patch_all_files(monkeypatch, tmp_path, matrix):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli.evaluate_popia_matrix", lambda *a, **k: matrix)


def test_all_files_exits_one_and_names_the_failing_file(tmp_path, capsys, monkeypatch):
    _patch_all_files(monkeypatch, tmp_path, _fake_matrix(_AVX2))
    assert cli_main(["eval", "popia", "--all-files"]) == 1
    out = capsys.readouterr().out
    assert f"Release gate across all files: FAIL ({_AVX2})" in out


def test_all_files_exits_zero_when_every_file_passes(tmp_path, capsys, monkeypatch):
    _patch_all_files(monkeypatch, tmp_path, _fake_matrix(None))
    assert cli_main(["eval", "popia", "--all-files"]) == 0
    assert "Release gate across all files: PASS" in capsys.readouterr().out


def test_all_files_json_lists_every_file_and_the_runtime(tmp_path, capsys, monkeypatch):
    _patch_all_files(monkeypatch, tmp_path, _fake_matrix(_AVX2))
    assert cli_main(["eval", "popia", "--all-files", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["all_passed"] is False
    assert data["failing"] == [_AVX2]
    assert set(data["results"]) == {_AVX2, _VNNI}
    assert data["runtime"]["onnxruntime"] == "1.24.4"


def test_single_file_json_includes_runtime_and_threshold(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(True))
    monkeypatch.setattr("semantix.cli._runtime_info", lambda: {"auto_variant": "onnx/x.onnx"})
    assert cli_main(["eval", "popia", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runtime"] == {"auto_variant": "onnx/x.onnx"}
    assert data["threshold"] == 0.5


def _boom(*_a, **_k):
    raise OSError("HF unreachable while loading model files")


def test_all_files_model_load_failure_exits_two(tmp_path, capsys, monkeypatch):
    # "could not run" must not read as "the gate failed" -- CI branches on the exit code.
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli.evaluate_popia_matrix", _boom)
    assert cli_main(["eval", "popia", "--all-files"]) == 2
    assert "unreachable" in capsys.readouterr().err.lower()


def test_single_file_model_load_failure_exits_two(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", _boom)
    assert cli_main(["eval", "popia"]) == 2
    assert "unreachable" in capsys.readouterr().err.lower()


def test_all_files_json_names_the_regressed_clause(tmp_path, capsys, monkeypatch):
    # The uploaded report.json must explain its own red gate, like the terminal output does.
    # v1's AVX2 file in miniature: strong overall delta, but below stock on minimality.
    regressed = EvalReport(
        n_pairs=150,
        stock_accuracy=0.62,
        stock_f1_macro=0.55,
        popia_accuracy=0.78,
        popia_f1_macro=0.80,
        per_clause={"POPIA consent": (0.60, 0.75), "POPIA minimality": (0.74, 0.71)},
        delta_f1=0.25,
        release_gate_passed=False,
    )
    matrix = MatrixReport(
        threshold=0.5,
        runtime={"onnxruntime": "1.30.0"},
        results={_AVX2: regressed, _VNNI: _fake_report(True)},
    )
    _patch_all_files(monkeypatch, tmp_path, matrix)
    assert cli_main(["eval", "popia", "--all-files", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["results"][_AVX2]["regressed_clauses"] == ["POPIA minimality"]
    assert data["results"][_VNNI]["regressed_clauses"] == []
