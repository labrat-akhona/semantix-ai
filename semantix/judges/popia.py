"""POPIA-fine-tuned NLI judge.

Loads the ``labrat-aiko/nli-popia-v1`` quantized ONNX model from
HuggingFace. Inherits inference, CPU variant detection, and caching logic
from :class:`semantix.judges.quantized_nli.QuantizedNLIJudge`; this subclass
overrides the model identity and pins a different recommended threshold.

Requires: ``pip install semantix-ai[popia]`` (same deps as ``[turbo]``).
"""

from __future__ import annotations

from typing import ClassVar

from semantix.judges import quantized_nli as _qnli
from semantix.judges.quantized_nli import QuantizedNLIJudge


class POPIAJudge(QuantizedNLIJudge):
    """Semantic judge fine-tuned on POPIA (Protection of Personal Information Act)."""

    _REPO_ID: ClassVar[str] = "labrat-aiko/nli-popia-v1"
    recommended_threshold: ClassVar[float | None] = 0.75

    def __init__(self, model_variant: str | None = None) -> None:
        variant = model_variant or _qnli._detect_onnx_variant()
        self._session = _qnli._load_session(variant, repo_id=self._REPO_ID)
        self._tokenizer = _qnli._load_tokenizer(repo_id=self._REPO_ID)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    @classmethod
    def clauses(cls) -> list[str]:
        """Canonical POPIA concept labels this model was trained on."""
        return [
            "POPIA consent",
            "POPIA minimality / purpose limitation",
            "POPIA security safeguards",
            "POPIA breach notification",
            "POPIA cross-border transfers",
            "POPIA general processing",
            "POPIA data subject rights",
        ]
