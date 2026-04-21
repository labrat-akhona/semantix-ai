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


import os
import re

import httpx

_SYSTEM_PROMPT = (
    "You are a strict semantic judge. Respond ONLY with a single number "
    "between 0.0 and 1.0 representing the probability the given TEXT "
    "fulfills the given INTENT. No other output."
)

_NUMBER_RE = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b")

# Paid-tier rates (Groq Llama 3.3 70B as of 2026-04): $0.59/M input, $0.79/M output.
_GROQ_INPUT_PER_TOKEN_USD = 0.59 / 1_000_000
_GROQ_OUTPUT_PER_TOKEN_USD = 0.79 / 1_000_000


class GroqJudge:
    name = "groq-llama-3.3-70b"

    def __init__(
        self,
        *,
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout
        self._api_key = os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not set")

    def evaluate(self, text: str, intent: str) -> JudgeResult:
        user = f"INTENT: {intent}\n\nTEXT: {text}"
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 10,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = "https://api.groq.com/openai/v1/chat/completions"

        last_error: str | None = None
        backoff = 1.0
        for attempt in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                resp = httpx.post(url, headers=headers, json=body, timeout=self._timeout)
            except httpx.HTTPError as exc:
                last_error = f"HTTPError: {exc}"
                time.sleep(backoff)
                backoff *= 2
                continue
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                last_error = "429"
                time.sleep(retry_after)
                backoff *= 2
                continue
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(backoff)
                backoff *= 2
                continue

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            match = _NUMBER_RE.search(content)
            if not match:
                return JudgeResult(
                    score=float("nan"),
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                    paid_equivalent_usd=_cost(data.get("usage", {})),
                    raw=content,
                    error=f"non-numeric response: {content!r}",
                )
            score = max(0.0, min(1.0, float(match.group(1))))
            return JudgeResult(
                score=score,
                latency_ms=latency_ms,
                cost_usd=0.0,
                paid_equivalent_usd=_cost(data.get("usage", {})),
                raw=content,
            )

        return JudgeResult(
            score=float("nan"),
            latency_ms=0.0,
            cost_usd=0.0,
            paid_equivalent_usd=0.0,
            error=last_error or "exhausted retries",
        )


def _cost(usage: dict) -> float:
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return prompt * _GROQ_INPUT_PER_TOKEN_USD + completion * _GROQ_OUTPUT_PER_TOKEN_USD


# Paid-tier rates for Gemini 2.5 (per Google as of 2026-04):
# Flash: $0.075/M input, $0.30/M output
# Pro:   $1.25/M input, $5.00/M output
_GEMINI_FLASH_IN_USD = 0.075 / 1_000_000
_GEMINI_FLASH_OUT_USD = 0.30 / 1_000_000
_GEMINI_PRO_IN_USD = 1.25 / 1_000_000
_GEMINI_PRO_OUT_USD = 5.00 / 1_000_000


class _GeminiJudgeBase:
    """Shared Gemini REST-API logic. Subclasses set model + rates."""

    _model: str = ""
    _rate_in: float = 0.0
    _rate_out: float = 0.0

    def __init__(self, *, max_retries: int = 2, timeout: float = 60.0) -> None:
        self._max_retries = max_retries
        self._timeout = timeout
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    @property
    def name(self) -> str:
        return self._model

    def evaluate(self, text: str, intent: str) -> JudgeResult:
        prompt = (
            "You are a strict semantic judge. Respond ONLY with a single number "
            "between 0.0 and 1.0 representing the probability the given TEXT "
            "fulfills the given INTENT. No other output.\n\n"
            f"INTENT: {intent}\n\nTEXT: {text}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 64},
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )

        last_error: str | None = None
        backoff = 1.0
        for attempt in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                resp = httpx.post(url, json=body, timeout=self._timeout)
            except httpx.HTTPError as exc:
                last_error = f"HTTPError: {exc}"
                time.sleep(backoff)
                backoff *= 2
                continue
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                last_error = "429"
                time.sleep(retry_after)
                backoff *= 2
                continue
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(backoff)
                backoff *= 2
                continue

            data = resp.json()
            try:
                content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                return JudgeResult(
                    score=float("nan"),
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                    paid_equivalent_usd=0.0,
                    error=f"malformed response: {str(data)[:200]}",
                )
            match = _NUMBER_RE.search(content)
            if not match:
                return JudgeResult(
                    score=float("nan"),
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                    paid_equivalent_usd=self._cost(data.get("usageMetadata", {})),
                    raw=content,
                    error=f"non-numeric response: {content!r}",
                )
            score = max(0.0, min(1.0, float(match.group(1))))
            return JudgeResult(
                score=score,
                latency_ms=latency_ms,
                cost_usd=0.0,
                paid_equivalent_usd=self._cost(data.get("usageMetadata", {})),
                raw=content,
            )

        return JudgeResult(
            score=float("nan"),
            latency_ms=0.0,
            cost_usd=0.0,
            paid_equivalent_usd=0.0,
            error=last_error or "exhausted retries",
        )

    def _cost(self, usage: dict) -> float:
        prompt = usage.get("promptTokenCount", 0)
        completion = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return prompt * self._rate_in + completion * self._rate_out


class GeminiFlashJudge(_GeminiJudgeBase):
    _model = "gemini-2.5-flash"
    _rate_in = _GEMINI_FLASH_IN_USD
    _rate_out = _GEMINI_FLASH_OUT_USD


class GeminiProJudge(_GeminiJudgeBase):
    _model = "gemini-2.5-pro"
    _rate_in = _GEMINI_PRO_IN_USD
    _rate_out = _GEMINI_PRO_OUT_USD
