"""Quantized NLI judge — INT8 ONNX inference without PyTorch.

Uses onnxruntime + tokenizers for ~50% faster CPU inference than the
PyTorch-based NLIJudge while producing identical scores.

Requires: pip install onnxruntime tokenizers huggingface-hub
"""

from __future__ import annotations

import importlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np

from semantix.judges import Judge, Verdict
from semantix.judges.nli import _to_hypothesis

_REPO_ID = "cross-encoder/nli-MiniLM2-L6-H768"
_CALIBRATION_FILENAME = "calibration.json"


def _require_turbo_dep(module: str):
    """Import an optional inference dependency, or raise an actionable error.

    ``pip install semantix-ai`` alone does not pull ``onnxruntime`` /
    ``tokenizers`` / ``huggingface-hub`` — they ship in the ``[popia]`` extra
    (identical deps under ``[turbo]`` / ``[gdpr]``). A bare ``ImportError`` here
    is cryptic; point the caller at the fix instead.
    """
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ModuleNotFoundError(
            f"QuantizedNLIJudge/POPIAJudge needs the optional dependency {module!r}, "
            f"which ships in an extra. Install it with:\n"
            f"    pip install 'semantix-ai[popia]'\n"
            f"(the same inference deps are also under [turbo] and [gdpr])."
        ) from err


def _resolve_temperature(repo_id: str, calibrated: bool) -> float:
    """Softmax temperature for the judge; warn if calibration was asked for but is absent.

    Returns ``1.0`` (no scaling) unless ``calibrated`` is requested *and* a real
    ``calibration.json`` (``T != 1.0``) was found. Requesting calibration on a
    model that has none is a silent trap otherwise — the caller thinks scores are
    calibrated when they are raw — so it emits a ``UserWarning`` and leaves
    ``judge.calibrated`` reporting ``False``.
    """
    if not calibrated:
        return 1.0
    temperature = _load_temperature_constant(repo_id)
    if temperature == 1.0:
        warnings.warn(
            f"calibrated=True but no usable calibration.json was found for {repo_id!r}; "
            f"temperature stays 1.0, so verdict.score is the raw (uncalibrated) entailment "
            f"probability. Check judge.calibrated before quoting score magnitudes.",
            stacklevel=3,
        )
    return temperature


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
    _require_turbo_dep("onnxruntime")
    _require_turbo_dep("huggingface_hub")
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=variant)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def _load_tokenizer(repo_id: str = _REPO_ID):
    """Load the Rust-based tokenizer from HuggingFace Hub.

    Repo layouts differ across model versions (verified live 2026-07-15):
    ``nli-popia-v1`` ships ``tokenizer.json`` at the repo root, ``v2`` carries
    it under ``onnx/``, and ``v3`` under ``pytorch/`` only. The file is the same
    self-contained Rust tokenizer in every case, so try each known location so
    every published model loads regardless of which layout its pipeline used.
    """
    _require_turbo_dep("huggingface_hub")
    _require_turbo_dep("tokenizers")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub exposes it under .utils
        from huggingface_hub.utils import EntryNotFoundError

    last_err: Exception | None = None
    for filename in ("tokenizer.json", "onnx/tokenizer.json", "pytorch/tokenizer.json"):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            return Tokenizer.from_file(path)
        except EntryNotFoundError as err:
            last_err = err
    raise FileNotFoundError(
        f"no tokenizer.json found in {repo_id!r} (tried repo root and onnx/)"
    ) from last_err


class QuantizedNLIJudge(Judge):
    """INT8 quantized NLI judge — fast CPU inference, no PyTorch.

    Requires the optional ``[popia]`` extra: ``pip install semantix-ai[popia]``
    (pulls ``onnxruntime`` + ``tokenizers`` + ``huggingface-hub``; the same deps
    are under ``[turbo]`` / ``[gdpr]``). Downloads the pre-quantized ONNX model
    from HuggingFace Hub on first use and auto-selects the best variant for the
    host CPU architecture.

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
        compatibility; defaults to ``False``. If a model has no
        ``calibration.json`` (base models, un-calibrated fine-tunes such as the
        default v1) the temperature stays ``1.0`` and a ``UserWarning`` is
        raised; query :attr:`calibrated` to see the resulting state.
    """

    recommended_threshold = 0.3
    _REPO_ID: str = _REPO_ID

    def __init__(self, model_variant: str | None = None, *, calibrated: bool = False) -> None:
        variant = model_variant or _detect_onnx_variant()
        self._session = _load_session(variant, repo_id=self._REPO_ID)
        self._tokenizer = _load_tokenizer(repo_id=self._REPO_ID)
        # Discover which inputs the ONNX graph actually accepts.
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._temperature = _resolve_temperature(self._REPO_ID, calibrated)

    @property
    def calibrated(self) -> bool:
        """Whether a real temperature-scaling constant (``T != 1.0``) is in effect.

        ``False`` means ``verdict.score`` is the raw entailment probability. For an
        uncalibrated model (e.g. the default v1) that number is systematically
        over-confident, so lean on the ``threshold`` decision rather than reading
        the magnitude as a probability.
        """
        return self._temperature != 1.0

    def evaluate(
        self,
        output: str | None = None,
        intent_description: str | None = None,
        threshold: float = 0.5,
        *,
        premise: str | None = None,
        hypothesis: str | None = None,
    ) -> Verdict:
        """Score whether *premise* entails *hypothesis* (NLI).

        NLI mapping — read this before wiring a compliance audit
        --------------------------------------------------------
        This is a cross-encoder NLI judge, so the two texts are a **premise**
        (the evidence/text under scrutiny) and a **hypothesis** (the claim
        being tested). The historical parameter names come from the library's
        LLM-output-validation origin and map as::

            output              == premise     (the evidence: a policy, a reply)
            intent_description  == hypothesis   (the claim: "data leaves the country")

        Passing them backwards produces a plausible-looking but meaningless
        score and raises nothing, so for NLI/compliance work prefer the
        explicit aliases::

            judge.evaluate(premise=policy_text, hypothesis=requirement_text)

        Both spellings work; supplying a legacy name **and** its alias for the
        same slot is a wiring error and raises ``TypeError`` rather than
        silently picking one.

        Parameters
        ----------
        output / premise:
            The premise — the text being judged. ``premise`` is the alias.
        intent_description / hypothesis:
            The hypothesis — the claim tested against the premise.
            ``hypothesis`` is the alias.
        threshold:
            Minimum entailment probability for ``passed``.
        """
        if output is not None and premise is not None:
            raise TypeError(
                "evaluate() got both `output` and its alias `premise` -- pass one, not both"
            )
        if intent_description is not None and hypothesis is not None:
            raise TypeError(
                "evaluate() got both `intent_description` and its alias `hypothesis` "
                "-- pass one, not both"
            )
        premise_text = premise if premise is not None else output
        hypothesis_src = hypothesis if hypothesis is not None else intent_description
        if premise_text is None:
            raise TypeError("evaluate() requires the premise (positional `output`, or `premise=`)")
        if hypothesis_src is None:
            raise TypeError(
                "evaluate() requires the hypothesis "
                "(positional `intent_description`, or `hypothesis=`)"
            )

        hypothesis_text = _to_hypothesis(hypothesis_src)
        encoded = self._tokenizer.encode(premise_text, hypothesis_text)

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
