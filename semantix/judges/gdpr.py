"""GDPR-fine-tuned NLI judge (scaffold — fine-tuned weights pending).

Intended to load ``labrat-aiko/nli-gdpr-v1`` once published. Until then, this
class falls back to the generic :class:`~semantix.judges.quantized_nli.QuantizedNLIJudge`
weights so downstream code can be written against a stable API.

Requires: ``pip install semantix-ai[gdpr]`` (same deps as ``[turbo]``).

**Roadmap:** ``docs/superpowers/specs/2026-04-22-gdpr-v0-scaffold.md``
"""

from __future__ import annotations

import warnings
from typing import ClassVar

from semantix.judges import quantized_nli as _qnli
from semantix.judges.quantized_nli import QuantizedNLIJudge


class GDPRJudge(QuantizedNLIJudge):
    """Semantic judge targeting the EU General Data Protection Regulation.

    The fine-tuned weights for this judge are not yet published. Until
    `labrat-aiko/nli-gdpr-v1` ships, `GDPRJudge()` instantiates against the
    generic quantized NLI model. Scores will be closer to stock NLI than to
    a clause-specialised judge — contribute seeds via
    https://github.com/labrat-akhona/semantix-ai/issues/gdpr-v0-seeds to
    accelerate the fine-tune.
    """

    _REPO_ID: ClassVar[str] = "labrat-aiko/nli-gdpr-v1"
    _FALLBACK_REPO_ID: ClassVar[str] = "labrat-aiko/nli-popia-v1"
    recommended_threshold: ClassVar[float | None] = 0.70

    def __init__(self, model_variant: str | None = None) -> None:
        variant = model_variant or _qnli._detect_onnx_variant()
        try:
            self._session = _qnli._load_session(variant, repo_id=self._REPO_ID)
            self._tokenizer = _qnli._load_tokenizer(repo_id=self._REPO_ID)
            self._repo_id = self._REPO_ID
        except Exception as exc:  # noqa: BLE001 — HF errors are varied
            warnings.warn(
                f"nli-gdpr-v1 not yet published ({type(exc).__name__}); falling "
                f"back to {self._FALLBACK_REPO_ID}. Per-clause GDPR performance "
                f"will be lower than POPIA until the fine-tune lands.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._session = _qnli._load_session(variant, repo_id=self._FALLBACK_REPO_ID)
            self._tokenizer = _qnli._load_tokenizer(repo_id=self._FALLBACK_REPO_ID)
            self._repo_id = self._FALLBACK_REPO_ID
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    @classmethod
    def clauses(cls) -> list[str]:
        """Canonical GDPR concept labels this model is/will be trained on."""
        return [
            "GDPR consent",
            "GDPR minimality",
            "GDPR security",
            "GDPR breach notification",
            "GDPR cross-border transfers",
            "GDPR general processing",
            "GDPR data subject rights",
        ]
