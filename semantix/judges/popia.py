"""POPIA-fine-tuned NLI judge.

Loads a ``labrat-aiko/nli-popia-v{N}`` quantized ONNX model from HuggingFace.
Inherits inference, CPU variant detection, and caching logic from
:class:`semantix.judges.quantized_nli.QuantizedNLIJudge`.

Versions:

* ``v1`` (default until 0.2.x): 7-clause coverage, macro F1 0.813 on the
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
        Optional override for the ONNX quantization variant to load.
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
        variant = model_variant or _qnli._detect_onnx_variant()
        self._session = _qnli._load_session(variant, repo_id=repo_id)
        self._tokenizer = _qnli._load_tokenizer(repo_id=repo_id)
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._temperature = _qnli._resolve_temperature(repo_id, calibrated)

    @property
    def version(self) -> str:
        return self._version

    @classmethod
    def clauses(cls, version: str = "v1") -> list[str]:
        """Canonical POPIA concept labels this model was trained on."""
        if version == "v1":
            return list(_V1_CLAUSES)
        if version == "v2":
            return list(_V2_CLAUSES)
        raise ValueError(f"unknown POPIAJudge version {version!r}")
