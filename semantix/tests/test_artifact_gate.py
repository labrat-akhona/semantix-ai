"""Unit tests for the post-export artifact gate in ``scripts/train_popia_v2.py``.

The gate is the safeguard that a bad *quantized* model cannot ship: it scores the
SHIPPED ONNX file (not the PyTorch weights the release gate saw) and refuses a
constant predictor or a >10pp macro-F1 regression. Before these tests it had been
exercised exactly once — remotely, on Modal, in anger. A gate nobody has tested is
a gate trusted on faith, which is the exact failure mode that shipped v3's first
artifact (a constant predictor that passed a gate scoring the wrong file).

These tests prove it locally and GPU-free by stubbing ``onnxruntime`` so we can
feed ``onnx_macro_f1`` known logits, plus exercising the pure ``artifact_gate_verdict``
decision directly.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

# scripts/ is not a package — load train_popia_v2.py by path.
_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "train_popia_v2.py"
_spec = importlib.util.spec_from_file_location("train_popia_v2", _SCRIPTS)
tpv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpv2)

# Eval rows spanning all three classes. The premise encodes the intended label id
# so a "healthy" stub session can predict perfectly from the tokenized input.
_LABELS = ["contradiction", "entailment", "neutral", "contradiction", "entailment", "neutral"]
_ROWS = [
    {"premise": str(tpv2.label_to_id(lab)), "hypothesis": "h", "label": lab, "clause": "c"}
    for lab in _LABELS
]


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Stands in for ort.InferenceSession — returns caller-supplied logits."""

    def __init__(self, predict) -> None:
        self._predict = predict

    def get_inputs(self):
        # No token_type_ids -> onnx_macro_f1 skips that feed branch.
        return [_FakeInput("input_ids"), _FakeInput("attention_mask")]

    def run(self, _outputs, feeds):
        return [self._predict(feeds["input_ids"])]


def _fake_tokenizer(premises, hypotheses, **_kw):
    # Encode the intended label id (stringified in the premise) into input_ids[:, 0].
    ids = np.array([[int(p)] for p in premises], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def _healthy(ids):
    """One-hot logits matching the encoded label -> perfect predictions."""
    labels = ids[:, 0]
    logits = np.full((labels.shape[0], 3), -5.0, dtype=np.float32)
    logits[np.arange(labels.shape[0]), labels] = 5.0
    return logits


def _constant(ids):
    """Always argmax class 2 ('neutral') regardless of input -> dead artifact."""
    logits = np.zeros((ids.shape[0], 3), dtype=np.float32)
    logits[:, 2] = 5.0
    return logits


def _patch_ort(monkeypatch, predict):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _FakeSession(predict))


# --- onnx_macro_f1: the distinct-prediction detector -------------------------


def test_onnx_macro_f1_constant_predictor_has_one_distinct(monkeypatch):
    _patch_ort(monkeypatch, _constant)
    _f1, distinct = tpv2.onnx_macro_f1("unused.onnx", _fake_tokenizer, _ROWS, batch_size=4)
    assert distinct == 1  # THE tell for a dead artifact


def test_onnx_macro_f1_healthy_recovers_full_signal(monkeypatch):
    _patch_ort(monkeypatch, _healthy)
    f1, distinct = tpv2.onnx_macro_f1("unused.onnx", _fake_tokenizer, _ROWS, batch_size=4)
    assert distinct == 3
    assert f1 == pytest.approx(1.0)


# --- artifact_gate_verdict: the ship / no-ship decision ----------------------


def test_gate_rejects_constant_predictor():
    passed, reason = tpv2.artifact_gate_verdict(0.50, 1, 0.50, 1, 0.78, 0.96)
    assert passed is False
    assert "CONSTANT" in reason.upper()


def test_gate_rejects_regression_even_when_not_constant():
    # distinct fine, but F1 craters vs PyTorch — the v3 INT8 story (0.78 -> 0.58).
    passed, reason = tpv2.artifact_gate_verdict(0.58, 3, 0.58, 3, 0.78, 0.96)
    assert passed is False
    assert "regress" in reason.lower()


def test_gate_passes_healthy_artifact():
    passed, reason = tpv2.artifact_gate_verdict(0.78, 3, 0.96, 3, 0.78, 0.96)
    assert passed is True
    assert reason == ""


def test_gate_allows_small_quantization_drop_within_tolerance():
    # ~5pp drop on each holdout < 10pp tol -> still ships (honest INT8 rounding).
    passed, _reason = tpv2.artifact_gate_verdict(0.73, 3, 0.91, 3, 0.78, 0.96)
    assert passed is True


def test_gate_constant_check_precedes_regression_check():
    # distinct<=1 wins even if F1 looks acceptable — constant is the worse signal.
    passed, reason = tpv2.artifact_gate_verdict(0.78, 1, 0.96, 3, 0.78, 0.96)
    assert passed is False
    assert "CONSTANT" in reason.upper()
