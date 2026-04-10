"""LangChain integration — validate chain outputs with semantix Intents.

Usage::

    from semantix.integrations.langchain import SemanticValidator
    from semantix import Intent

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    validator = SemanticValidator(Polite)
    chain = prompt | llm | StrOutputParser() | validator
"""

from __future__ import annotations

from typing import Any

from semantix.intent import Intent
from semantix.judges import Judge


def _default_judge() -> Judge:
    """Resolve the default judge — same logic as @validate_intent."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


class SemanticValidator:
    """A LangChain-compatible Runnable that validates text against an Intent.

    Composes with LangChain's ``|`` pipe syntax. On failure raises
    ``OutputParserException`` if langchain-core is installed, otherwise
    ``ValueError``.
    """

    def __init__(self, intent: type[Intent], judge: Judge | None = None) -> None:
        self._judge = judge or _default_judge()
        self._description = intent.description()
        if "threshold" not in intent.__dict__ and self._judge.recommended_threshold is not None:
            self._threshold = self._judge.recommended_threshold
        else:
            self._threshold = intent.threshold

    def _validate(self, value: Any) -> Any:
        text = str(value) if not isinstance(value, str) else value
        verdict = self._judge.evaluate(text, self._description, self._threshold)
        if not verdict.passed:
            score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
            reason_str = f": {verdict.reason}" if verdict.reason else ""
            msg = (
                f"Semantic validation failed (score={score_str}){reason_str}. "
                f"The text must satisfy: {self._description}"
            )
            try:
                from langchain_core.exceptions import OutputParserException

                raise OutputParserException(msg)
            except ImportError:
                raise ValueError(msg) from None
        return value

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Validate a single input."""
        return self._validate(input)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Async validate a single input."""
        return self._validate(input)

    def batch(self, inputs: list[Any], config: Any = None, **kwargs: Any) -> list[Any]:
        """Validate a batch of inputs."""
        return [self._validate(inp) for inp in inputs]

    def __or__(self, other: Any) -> Any:
        """Support ``validator | next_step`` pipe syntax."""
        try:
            from langchain_core.runnables import RunnableSequence

            return RunnableSequence(first=self, last=other)
        except ImportError:
            raise ImportError("langchain-core is required for pipe composition") from None

    def __ror__(self, other: Any) -> Any:
        """Support ``prev_step | validator`` pipe syntax."""
        try:
            from langchain_core.runnables import RunnableSequence

            return RunnableSequence(first=other, last=self)
        except ImportError:
            raise ImportError("langchain-core is required for pipe composition") from None
