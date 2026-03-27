"""Accurate LLM-based judge using the OpenAI chat completions API."""

from __future__ import annotations

import os

from semantix.judges import Judge, Verdict

_SYSTEM_PROMPT = (
    "You are a strict semantic validator. You will be given a REQUIREMENT "
    "and a TEXT. Decide whether the TEXT fully satisfies the REQUIREMENT.\n"
    "Respond with EXACTLY one word: Yes or No."
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
            max_tokens=4,
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
        answer = (response.choices[0].message.content or "").strip().lower()
        passed = answer.startswith("yes")
        return Verdict(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Judge answered: {answer}",
        )
