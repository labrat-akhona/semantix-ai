"""Tests for QuantizedNLIJudge — mocked ONNX session, no model download."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

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


def _make_input_mock(input_name):
    """Create a mock ONNX input with a .name attribute."""
    m = MagicMock()
    m.name = input_name
    return m


def _mock_session(logits):
    """Return a mock InferenceSession that returns fixed logits."""
    session = MagicMock()
    session.run.return_value = [np.array([logits], dtype=np.float32)]
    session.get_inputs.return_value = [
        _make_input_mock("input_ids"),
        _make_input_mock("attention_mask"),
        _make_input_mock("token_type_ids"),
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
        # Label order: [contradiction, entailment, neutral] — high entailment at idx 1
        mock_load_session.return_value = _mock_session([0.1, 5.0, 0.5])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        judge = QuantizedNLIJudge()
        verdict = judge.evaluate("polite text", "The text must be polite", 0.5)
        assert verdict.passed is True
        assert verdict.score > 0.9

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_failing_verdict(self, mock_load_session, mock_load_tokenizer):
        # High contradiction at idx 0, low entailment at idx 1
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
        # Label order: [contradiction, entailment, neutral] — entailment at idx 1
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


# ---------------------------------------------------------------------------
# Unit: temperature scaling
# ---------------------------------------------------------------------------


class TestSoftmaxTemperature:
    def test_temperature_one_is_identity(self):
        logits = np.array([-1.0, 5.0, 0.5])
        assert np.allclose(_softmax(logits), _softmax(logits, temperature=1.0))

    def test_high_temperature_flattens(self):
        logits = np.array([0.0, 5.0, 0.0])
        sharp = _softmax(logits, temperature=1.0)
        flat = _softmax(logits, temperature=10.0)
        # Argmax preserved, but max prob is smaller.
        assert np.argmax(sharp) == np.argmax(flat) == 1
        assert flat[1] < sharp[1]
        # Bounds: flat should be much closer to uniform (1/3).
        assert flat[1] - 1 / 3 < sharp[1] - 1 / 3

    def test_low_temperature_sharpens(self):
        logits = np.array([0.0, 1.0, 0.5])
        normal = _softmax(logits, temperature=1.0)
        sharp = _softmax(logits, temperature=0.1)
        assert sharp[1] > normal[1]

    def test_rejects_zero_or_negative_temperature(self):
        import pytest

        logits = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            _softmax(logits, temperature=0.0)
        with pytest.raises(ValueError):
            _softmax(logits, temperature=-1.0)


# ---------------------------------------------------------------------------
# Unit: _load_temperature_constant
# ---------------------------------------------------------------------------


class TestLoadTemperatureConstant:
    def test_missing_calibration_returns_one(self):
        from huggingface_hub.errors import EntryNotFoundError

        from semantix.judges.quantized_nli import _load_temperature_constant

        with patch("huggingface_hub.hf_hub_download", side_effect=EntryNotFoundError("nope")):
            assert _load_temperature_constant("foo/bar") == 1.0

    def test_network_failure_returns_one(self):
        from semantix.judges.quantized_nli import _load_temperature_constant

        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("offline")):
            assert _load_temperature_constant("foo/bar") == 1.0

    def test_loads_temperature_when_present(self, tmp_path):
        from semantix.judges.quantized_nli import _load_temperature_constant

        cal = tmp_path / "calibration.json"
        cal.write_text('{"temperature": 2.5492, "ece_pre": 0.171, "ece_post": 0.075}')
        with patch("huggingface_hub.hf_hub_download", return_value=str(cal)):
            assert _load_temperature_constant("foo/bar") == 2.5492

    def test_malformed_calibration_returns_one(self, tmp_path):
        from semantix.judges.quantized_nli import _load_temperature_constant

        cal = tmp_path / "calibration.json"
        cal.write_text("not valid json {{")
        with patch("huggingface_hub.hf_hub_download", return_value=str(cal)):
            assert _load_temperature_constant("foo/bar") == 1.0


# ---------------------------------------------------------------------------
# Integration: calibrated QuantizedNLIJudge
# ---------------------------------------------------------------------------


class TestQuantizedNLIJudgeCalibration:
    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    @patch("semantix.judges.quantized_nli._load_temperature_constant", return_value=2.5)
    def test_calibrated_true_flattens_score(self, mock_T, mock_load_session, mock_load_tokenizer):
        # Same logits, threshold 0.5 — un-calibrated passes, calibrated fails.
        mock_load_session.return_value = _mock_session([0.0, 1.6, 0.0])
        mock_load_tokenizer.return_value = _mock_tokenizer()

        un_cal = QuantizedNLIJudge()
        cal = QuantizedNLIJudge(calibrated=True)

        v_un = un_cal.evaluate("output", "Intent must hold", 0.5)
        v_cal = cal.evaluate("output", "Intent must hold", 0.5)

        # un-calibrated: peaked at entailment (idx 1); score > 0.5
        # calibrated (T=2.5): flattened; score < 0.5
        assert v_un.score > 0.5
        assert v_cal.score < 0.5
        assert v_un.passed is True
        assert v_cal.passed is False

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    @patch("semantix.judges.quantized_nli._load_temperature_constant", return_value=1.0)
    def test_calibrated_true_with_T_one_is_no_op(
        self, mock_T, mock_load_session, mock_load_tokenizer
    ):
        # When the model has no calibration constant, T=1.0 is returned and
        # behaviour matches calibrated=False.
        mock_load_session.return_value = _mock_session([0.1, 2.0, 0.5])
        mock_load_tokenizer.return_value = _mock_tokenizer()

        un_cal = QuantizedNLIJudge()
        cal = QuantizedNLIJudge(calibrated=True)

        v_un = un_cal.evaluate("output", "Intent must hold", 0.5)
        v_cal = cal.evaluate("output", "Intent must hold", 0.5)

        assert v_un.score == v_cal.score

    @patch("semantix.judges.quantized_nli._load_tokenizer")
    @patch("semantix.judges.quantized_nli._load_session")
    def test_default_is_uncalibrated(self, mock_load_session, mock_load_tokenizer):
        # calibrated defaults to False; _load_temperature_constant should not be called.
        mock_load_session.return_value = _mock_session([0.0, 1.0, 0.0])
        mock_load_tokenizer.return_value = _mock_tokenizer()
        with patch("semantix.judges.quantized_nli._load_temperature_constant") as mock_T:
            QuantizedNLIJudge()
            mock_T.assert_not_called()


# ---------------------------------------------------------------------------
# Unit: _load_tokenizer location fallback
# ---------------------------------------------------------------------------


class TestLoadTokenizerFallback:
    """Published repos put tokenizer.json in different places: v1 at root,
    v2 under onnx/, v3 under pytorch/. The loader must try each in order."""

    def _run(self, available: str):
        from huggingface_hub.errors import EntryNotFoundError

        from semantix.judges.quantized_nli import _load_tokenizer

        tried: list[str] = []

        def fake_download(repo_id, filename):
            tried.append(filename)
            if filename != available:
                raise EntryNotFoundError(f"{filename} missing")
            return f"/tmp/{filename}"

        sentinel = object()
        with (
            patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
            patch("tokenizers.Tokenizer.from_file", return_value=sentinel),
        ):
            result = _load_tokenizer("repo/x")
        assert result is sentinel
        return tried

    def test_root_layout_v1(self):
        tried = self._run("tokenizer.json")
        assert tried == ["tokenizer.json"]

    def test_onnx_layout_v2(self):
        tried = self._run("onnx/tokenizer.json")
        assert tried == ["tokenizer.json", "onnx/tokenizer.json"]

    def test_pytorch_layout_v3(self):
        tried = self._run("pytorch/tokenizer.json")
        assert tried == ["tokenizer.json", "onnx/tokenizer.json", "pytorch/tokenizer.json"]

    def test_missing_everywhere_raises(self):
        import pytest

        with pytest.raises(FileNotFoundError, match="no tokenizer.json"):
            self._run("nonexistent.json")
