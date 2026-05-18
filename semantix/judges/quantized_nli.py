"""Quantized NLI judge — INT8 ONNX inference without PyTorch.

Uses onnxruntime + tokenizers for ~50% faster CPU inference than the
PyTorch-based NLIJudge while producing identical scores.

Requires: pip install onnxruntime tokenizers huggingface-hub
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from semantix.judges import Judge, Verdict
from semantix.judges.nli import _to_hypothesis

_REPO_ID = "cross-encoder/nli-MiniLM2-L6-H768"
_CALIBRATION_FILENAME = "calibration.json"


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over a 1-D array.

    ``temperature > 1`` flattens the distribution (use when raw logits are
    over-peaked / over-confident). ``temperature == 1`` is the standard
    softmax. ``temperature <= 0`` is rejected.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    scaled = logits / temperature
    shifted = scaled - scaled.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _load_temperature_constant(repo_id: str) -> float:
    """Load the softmax temperature from the HF model's calibration.json.

    Returns ``1.0`` (no scaling) when the file is absent or malformed —
    base models and un-calibrated fine-tunes work unchanged. This is the
    *probability calibration* constant (Guo et al., 2017 temperature
    scaling), distinct from the *threshold calibration* in
    ``semantix.training.calibrate`` which sets per-Intent decision cutoffs.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        return 1.0
    try:
        path = hf_hub_download(repo_id=repo_id, filename=_CALIBRATION_FILENAME)
    except EntryNotFoundError:
        return 1.0
    except Exception:
        # Network/auth issues shouldn't break inference — fall through.
        return 1.0
    try:
        data = json.loads(Path(path).read_text())
        T = float(data.get("temperature", 1.0))
        return T if T > 0 else 1.0
    except (ValueError, OSError, json.JSONDecodeError):
        return 1.0


def _read_cpuinfo() -> str:
    """Read CPU flags from /proc/cpuinfo (Linux) or return empty string."""
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("flags"):
                return line.lower()
        return text.lower()
    except OSError:
        return ""


def _detect_onnx_variant() -> str:
    """Pick the best pre-quantized ONNX variant for this CPU."""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "onnx/model_qint8_arm64.onnx"

    cpuinfo = _read_cpuinfo()
    if "avx512vnni" in cpuinfo or "avx512_vnni" in cpuinfo:
        return "onnx/model_qint8_avx512_vnni.onnx"
    if "avx512f" in cpuinfo or "avx512" in cpuinfo:
        return "onnx/model_qint8_avx512.onnx"
    # AVX2 or generic fallback
    return "onnx/model_quint8_avx2.onnx"


def _load_session(variant: str, repo_id: str = _REPO_ID):
    """Download the ONNX model and create an InferenceSession."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=variant)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def _load_tokenizer(repo_id: str = _REPO_ID):
    """Load the Rust-based tokenizer from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    return Tokenizer.from_file(path)


class QuantizedNLIJudge(Judge):
    """INT8 quantized NLI judge — fast CPU inference, no PyTorch.

    Downloads the pre-quantized ONNX model from HuggingFace Hub on first
    use and auto-selects the best variant for the host CPU architecture.

    Default threshold is 0.5 (not 0.8) because NLI entailment probabilities
    are calibrated differently than cosine similarity scores.

    Parameters
    ----------
    model_variant:
        Override the ONNX filename within the repo (e.g.
        ``"onnx/model_qint8_arm64.onnx"``).  By default the variant
        is auto-detected from CPU flags.
    calibrated:
        When ``True``, fetch ``calibration.json`` from the model repo and
        apply temperature scaling at softmax so ``verdict.score`` is a
        well-calibrated probability rather than the raw (often
        over-confident) entailment likelihood. Opt-in for backwards
        compatibility; defaults to ``False``. Models without a
        ``calibration.json`` (base models, un-calibrated fine-tunes)
        silently fall back to ``T=1.0`` even with ``calibrated=True``.
    """

    recommended_threshold = 0.3
    _REPO_ID: str = _REPO_ID

    def __init__(self, model_variant: str | None = None, *, calibrated: bool = False) -> None:
        variant = model_variant or _detect_onnx_variant()
        self._session = _load_session(variant, repo_id=self._REPO_ID)
        self._tokenizer = _load_tokenizer(repo_id=self._REPO_ID)
        # Discover which inputs the ONNX graph actually accepts.
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._temperature = _load_temperature_constant(self._REPO_ID) if calibrated else 1.0

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.5,
    ) -> Verdict:
        hypothesis = _to_hypothesis(intent_description)
        encoded = self._tokenizer.encode(output, hypothesis)

        feeds = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        }
        # Only include token_type_ids if the model expects it.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)
        logits = self._session.run(None, feeds)[0][0]
        probs = _softmax(logits, temperature=self._temperature)
        # Label order: {0: contradiction, 1: entailment, 2: neutral} -- matches
        # the base model's config.id2label and is preserved verbatim on ONNX export.
        entailment_score = float(probs[1])

        return Verdict(
            passed=entailment_score >= threshold,
            score=entailment_score,
        )
