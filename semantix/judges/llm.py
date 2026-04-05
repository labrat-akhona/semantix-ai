"""Accurate LLM-based judge using the OpenAI chat completions API."""

from __future__ import annotations

import json
import os

from semantix.judges import Judge, Verdict

_SYSTEM_PROMPT = (
    "You are a strict semantic validator. You will be given a REQUIREMENT "
    "and a TEXT. Score how well the TEXT satisfies the REQUIREMENT.\n\n"
    "Respond with ONLY a JSON object (no markdown fences):\n"
    '{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}\n\n'
    "- 1.0 = fully satisfies the requirement\n"
    "- 0.0 = completely fails the requirement\n"
    "- Be precise: use the full range, not just 0 and 1."
)


class LLMJudge(Judge):
    """Judge that asks a lightweight LLM whether the output satisfies the intent.

    Uses the OpenAI Python SDK (``openai >= 1.0``).  Any OpenAI-compatible
    endpoint (Azure, local vLLM, Ollama, etc.) works — just set ``base_url``.

    Parameters
    ----------
    model:
        Model name to use for the judge call.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` env var.
    base_url:
        Optional base URL for OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # noqa: WPS433

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required. Pass it explicitly or set "
                "the OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._model = model

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"REQUIREMENT:\n{intent_description}\n\n"
                        f"TEXT:\n{output}"
                    ),
                },
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        score, reason = self._parse_response(raw)
        return Verdict(
            passed=score >= threshold,
            score=score,
            reason=reason,
        )

    @staticmethod
    def _parse_response(raw: str) -> tuple[float, str]:
        """Parse the JSON response from the LLM.

        Falls back gracefully to the legacy Yes/No format if JSON parsing fails.
        """
        try:
            data = json.loads(raw)
            score = float(data["score"])
            score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", ""))
            return score, reason
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback: legacy Yes/No response
            lower = raw.lower()
            if lower.startswith("yes"):
                return 1.0, f"Judge answered: {raw}"
            return 0.0, f"Judge answered: {raw}"
