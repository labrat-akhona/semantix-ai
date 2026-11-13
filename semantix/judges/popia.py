"""POPIA-fine-tuned NLI judge.

Loads a ``labrat-aiko/nli-popia-v{N}`` quantized ONNX model from HuggingFace.
Inherits inference and caching logic from
:class:`semantix.judges.quantized_nli.QuantizedNLIJudge`. ``v1`` loads the same
qint8 file on every CPU; ``v2`` still selects a file from the CPU.

Versions:

* ``v1`` (current default): 7-clause coverage, macro F1 0.813 on the
  v1 holdout. Strongest narrow-scope model.
* ``v2``: 10-clause coverage adding children's information (§34-35),
  special personal information (§26-33), and automated decision-making
  (§71). Broader at a small in-domain F1 cost on the v1 holdout.

Requires: ``pip install semantix-ai[popia]`` (same deps as ``[turbo]``).
"""

from __future__ import annotations

from typing import ClassVar

from semantix.judges import quantized_nli as _qnli
from semantix.judges.quantized_nli import QuantizedNLIJudge

_V1_CLAUSES = [
    "POPIA consent",
    "POPIA minimality / purpose limitation",
    "POPIA security safeguards",
    "POPIA breach notification",
    "POPIA cross-border transfers",
    "POPIA general processing",
    "POPIA data subject rights",
]

_V2_CLAUSES = _V1_CLAUSES + [
    "POPIA children's information",
    "POPIA special personal information",
    "POPIA automated decision-making",
]

_VERSION_TO_REPO: dict[str, str] = {
    "v1": "labrat-aiko/nli-popia-v1",
    "v2": "labrat-aiko/nli-popia-v2",
}

# Versions that load one file on every CPU instead of selecting by CPU flags. v1 was
# measured on AVX2-only and VNNI hardware (docs/experiments/2026-09-16-avx2-vs-qint8.md):
# the qint8 file is more accurate and faster on both. v2 has no such measurement yet.
_VERSION_TO_VARIANT: dict[str, str] = {
    "v1": _qnli.QINT8_VARIANT,
}

# Files still in a model repo that this library no longer selects. v1's AVX2 file fails the
# release gate (minimality) and stays published only because semantix <= 0.3.3 requests it
# by name on Windows, Intel macOS, and AVX2-only Linux.
_LEGACY_VARIANTS: dict[str, tuple[str, ...]] = {
    "v1": ("onnx/model_quint8_avx2.onnx",),
}


class POPIAJudge(QuantizedNLIJudge):
    """Semantic judge fine-tuned on POPIA (Protection of Personal Information Act).

    Requires the optional ``[popia]`` extra: ``pip install semantix-ai[popia]``.
    The model (~79 MB INT8 ONNX) downloads from HuggingFace Hub on first use.

    Parameters
    ----------
    version:
        Model version. ``"v1"`` (default) for 7-clause coverage with peak
        in-domain F1; ``"v2"`` for 10-clause coverage including the three
        AI-critical clauses (children's data, special PI, automated
        decision-making).
    model_variant:
        Optional override for the ONNX file to load. By default ``v1`` loads
        ``onnx/model_qint8_avx512_vnni.onnx`` on every CPU and ``v2`` selects a
        file from the CPU flags.
    calibrated:
        When ``True``, fetch the temperature constant from the model's
        ``calibration.json`` on HF and apply it at softmax so
        ``verdict.score`` is a well-calibrated probability. Only ``v2``
        ships a calibration constant (``T*=2.5492``, ECE 0.171 → 0.075
        on a stratified 60% test split); ``v1`` has none, so it stays at
        ``T=1.0`` and a ``UserWarning`` is raised — query ``judge.calibrated``
        for the resulting state. Defaults to ``False`` for backwards
        compatibility.
    """

    _REPO_ID: ClassVar[str] = "labrat-aiko/nli-popia-v1"  # back-compat default
    recommended_threshold: ClassVar[float | None] = 0.75

    def __init__(
        self,
        version: str = "v1",
        model_variant: str | None = None,
        *,
        calibrated: bool = False,
    ) -> None:
        if version not in _VERSION_TO_REPO:
            raise ValueError(
                f"unknown POPIAJudge version {version!r} — choose from {sorted(_VERSION_TO_REPO)}"
            )
        self._version = version
        repo_id = _VERSION_TO_REPO[version]
        self._REPO_ID = repo_id  # let inherited _load_temperature_constant find the right repo
        variant = model_variant or _VERSION_TO_VARIANT.get(version) or _qnli._detect_onnx_variant()
        self._model_file = variant
        self._session = _qnli._load_session(variant, repo_id=repo_id)
        self._tokenizer = _qnli._load_tokenizer(repo_id=repo_id)
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._temperature = _qnli._resolve_temperature(repo_id, calibrated)

    @property
    def version(self) -> str:
        return self._version

    @classmethod
    def gated_variants(cls, version: str = "v1") -> tuple[str, ...]:
        """ONNX files this library can select for ``version`` without an override.

        These are the files the release gate must pass on.
        """
        if version not in _VERSION_TO_REPO:
            raise ValueError(f"unknown POPIAJudge version {version!r}")
        if version in _VERSION_TO_VARIANT:
            return (_VERSION_TO_VARIANT[version],)
        return _qnli.ONNX_VARIANTS

    @classmethod
    def legacy_variants(cls, version: str = "v1") -> tuple[str, ...]:
        """Files still published for ``version`` that only older releases select."""
        if version not in _VERSION_TO_REPO:
            raise ValueError(f"unknown POPIAJudge version {version!r}")
        return _LEGACY_VARIANTS.get(version, ())

    @classmethod
    def clauses(cls, version: str = "v1") -> list[str]:
        """Canonical POPIA concept labels this model was trained on."""
        if version == "v1":
            return list(_V1_CLAUSES)
        if version == "v2":
            return list(_V2_CLAUSES)
        raise ValueError(f"unknown POPIAJudge version {version!r}")
