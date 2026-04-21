"""Judge adapters for benchmark runs.

All judges share the Judge protocol: given a text and an intent description,
return a JudgeResult with a 0-1 score, latency, cost, and optional error.
Errors never raise — they are recorded on the row and the run continues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class JudgeResult:
    score: float  # 0.0-1.0, or float("nan") on error
    latency_ms: float
    cost_usd: float  # 0.0 on free tier
    paid_equivalent_usd: float  # what this would cost at paid rates
    raw: str | None = None
    error: str | None = None


class Judge(Protocol):
    name: str

    def evaluate(self, text: str, intent: str) -> JudgeResult: ...


import time

from semantix.judges.quantized_nli import QuantizedNLIJudge


class SemantixJudge:
    name = "semantix"

    def __init__(self) -> None:
        self._inner = QuantizedNLIJudge()

    def evaluate(self, text: str, intent: str) -> JudgeResult:
        start = time.perf_counter()
        try:
            verdict = self._inner.evaluate(text, intent, threshold=0.5)
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(
                score=float("nan"),
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=0.0,
                paid_equivalent_usd=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - start) * 1000
        return JudgeResult(
            score=float(verdict.score if verdict.score is not None else 0.0),
            latency_ms=latency_ms,
            cost_usd=0.0,
            paid_equivalent_usd=0.0,
        )
