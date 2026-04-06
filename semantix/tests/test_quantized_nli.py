"""Tests for QuantizedNLIJudge — mocked ONNX session, no model download."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from semantix.judges import Verdict
from semantix.judges.quantized_nli import (
    QuantizedNLIJudge,
    _detect_onnx_variant,
    _softmax,
)


# ---------------------------------------------------------------------------
# Unit: softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_basic_softmax(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_argmax_preserved(self):
        logits = np.array([-1.0, 5.0, 0.5])
        probs = _softmax(logits)
        assert np.argmax(probs) == 1

    def test_softmax_all_zeros(self):
        logits = np.array([0.0, 0.0, 0.0])
        probs = _softmax(logits)
        assert abs(probs[0] - 1 / 3) < 1e-6


# ---------------------------------------------------------------------------
# Unit: CPU variant detection
# ---------------------------------------------------------------------------


class TestDetectOnnxVariant:
    @patch("semantix.judges.quantized_nli.platform.machine", return_value="aarch64")
    def test_arm64(self, _mock):
        assert _detect_onnx_variant() == "onnx/model_qint8_arm64.onnx"

    @patch("semantix.judges.quantized_nli.platform.machine", return_value="x86_64")
    @patch("semantix.judges.quantized_nli._read_cpuinfo", return_value="avx512vnni avx512f")
    def test_avx512_vnni(self, _cpuinfo, _machine):
        assert _detect_onnx_variant() == "onnx/model_qint8_avx512_vnni.onnx"

    @patch("semantix.judges.quantized_nli.platform.machine", return_value="x86_64")
    @patch("semantix.judges.quantized_nli._read_cpuinfo", return_value="avx512f sse4_2")
    def test_avx512_no_vnni(self, _cpuinfo, _machine):
        assert _detect_onnx_variant() == "onnx/model_qint8_avx512.onnx"

    @patch("semantix.judges.quantized_nli.platform.machine", return_value="x86_64")
    @patch("semantix.judges.quantized_nli._read_cpuinfo", return_value="avx2 sse4_2")
    def test_avx2_fallback(self, _cpuinfo, _machine):
        assert _detect_onnx_variant() == "onnx/model_quint8_avx2.onnx"

    @patch("semantix.judges.quantized_nli.platform.machine", return_value="x86_64")
    @patch("semantix.judges.quantized_nli._read_cpuinfo", return_value="sse4_2")
    def test_no_avx_fallback(self, _cpuinfo, _machine):
        assert _detect_onnx_variant() == "onnx/model_quint8_avx2.onnx"


# ---------------------------------------------------------------------------
# Integration: QuantizedNLIJudge with mocked session
# ---------------------------------------------------------------------------


def _mock_session(logits):
    """Return a mock InferenceSession that returns fixed logits."""
    session = MagicMock()
    session.run.return_value = [np.array([logits], dtype=np.float32)]
    session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
        MagicMock(name="token_type_ids"),
    ]
    return session


def _mock_tokenizer():
    """Return a mock Tokenizer that returns fixed encoding."""
    tokenizer = MagicMock()
    encoding = MagicMock()
    encoding.ids = [101, 2023, 2003, 1037, 3231, 102, 2009, 2442, 2022, 4958, 102]
    encoding.attention_mask = [1] * 11
    encoding.type_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    tokenizer.encode.return_value = encoding
    return tokenizer


class TestQuantizedNLIJudge:
    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_passing_verdict(self, mock_load_session, mock_load_tokenizer):
        # logits: [contradiction, entailment, neutral] — high entailment
        mock_load_session.return_value = _mock_session([0.1, 5.0, 0.5])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        judge = QuantizedNLIJudge()
        verdict = judge.evaluate("polite text", "The text must be polite", 0.5)
        assert verdict.passed is True
        assert verdict.score > 0.9

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_failing_verdict(self, mock_load_session, mock_load_tokenizer):
        # logits: high contradiction
        mock_load_session.return_value = _mock_session([5.0, 0.1, 0.5])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        judge = QuantizedNLIJudge()
        verdict = judge.evaluate("rude text", "The text must be polite", 0.5)
        assert verdict.passed is False
        assert verdict.score < 0.1

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_returns_verdict_type(self, mock_load_session, mock_load_tokenizer):
        mock_load_session.return_value = _mock_session([0.1, 5.0, 0.5])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        judge = QuantizedNLIJudge()
        verdict = judge.evaluate("text", "intent", 0.5)
        assert isinstance(verdict, Verdict)

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_uses_hypothesis_transformation(self, mock_load_session, mock_load_tokenizer):
        mock_load_session.return_value = _mock_session([0.1, 5.0, 0.5])
        tok = _mock_tokenizer()
        mock_load_tokenizer.return_value = tok
        judge = QuantizedNLIJudge()
        judge.evaluate("text", "The text must politely decline", 0.5)
        # _to_hypothesis converts "The text must politely decline" to
        # "Someone is politely declining"
        call_args = tok.encode.call_args
        assert "Someone is politely declining" in call_args[0][1]

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_custom_threshold(self, mock_load_session, mock_load_tokenizer):
        # Score ~0.73 with these logits after softmax
        mock_load_session.return_value = _mock_session([0.1, 1.0, 0.0])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        judge = QuantizedNLIJudge()
        v_low = judge.evaluate("text", "intent", 0.5)
        v_high = judge.evaluate("text", "intent", 0.99)
        assert v_low.passed is True
        assert v_high.passed is False
        assert v_low.score == v_high.score  # same logits, different threshold

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_score_between_0_and_1(self, mock_load_session, mock_load_tokenizer):
        mock_load_session.return_value = _mock_session([2.0, 1.0, 3.0])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        judge = QuantizedNLIJudge()
        verdict = judge.evaluate("text", "intent", 0.5)
        assert 0.0 <= verdict.score <= 1.0
