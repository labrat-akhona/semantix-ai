"""Unit tests for POPIAJudge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantix.judges import Judge, Verdict


def _fake_logits():
    # Label order: {0: contradiction, 1: entailment, 2: neutral}.
    # Use a logit vector where entailment (index 1) dominates after softmax.
    import numpy as np
    return np.array([[0.2, 3.0, 0.5]], dtype=np.float32)


@pytest.fixture
def mocked_onnx(monkeypatch):
    """Patch ONNX session + tokenizer load so no network or model file is needed."""
    fake_session = MagicMock()
    input_ids_mock = MagicMock()
    input_ids_mock.name = "input_ids"
    attention_mock = MagicMock()
    attention_mock.name = "attention_mask"
    fake_session.get_inputs.return_value = [input_ids_mock, attention_mock]
    fake_session.run.return_value = [_fake_logits()]

    fake_tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.ids = [1, 2, 3]
    encoded.attention_mask = [1, 1, 1]
    encoded.type_ids = [0, 0, 0]
    fake_tokenizer.encode.return_value = encoded

    monkeypatch.setattr(
        "semantix.judges.quantized_nli._load_session",
        lambda variant, repo_id=None: fake_session,
    )
    monkeypatch.setattr(
        "semantix.judges.quantized_nli._load_tokenizer",
        lambda repo_id=None: fake_tokenizer,
    )
    yield fake_session


def test_repo_id_is_popia(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    assert POPIAJudge._REPO_ID == "labrat-aiko/nli-popia-v1"


def test_recommended_threshold_is_pinned(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    assert POPIAJudge.recommended_threshold == 0.75


def test_clauses_returns_seven_canonical_strings(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    clauses = POPIAJudge.clauses()
    assert len(clauses) == 7
    assert all(isinstance(c, str) and c.startswith("POPIA") for c in clauses)
    assert len(set(clauses)) == 7


def test_popia_judge_is_subclass_of_quantized_and_base(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    from semantix.judges.quantized_nli import QuantizedNLIJudge
    j = POPIAJudge()
    assert isinstance(j, QuantizedNLIJudge)
    assert isinstance(j, Judge)


def test_evaluate_delegates_and_returns_verdict(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    j = POPIAJudge()
    v = j.evaluate("some output", "some intent", threshold=0.5)
    assert isinstance(v, Verdict)
    assert v.passed is True


def test_download_failure_raises_runtime_error_no_silent_fallback(monkeypatch):
    def boom(variant, repo_id=None):
        raise RuntimeError("HF unreachable (simulated)")
    monkeypatch.setattr("semantix.judges.quantized_nli._load_session", boom)

    from semantix.judges.popia import POPIAJudge
    with pytest.raises(RuntimeError, match="HF unreachable"):
        POPIAJudge()
