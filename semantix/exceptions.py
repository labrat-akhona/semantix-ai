"""Semantix-specific exceptions."""


class SemanticIntentError(Exception):
    """Raised when an LLM output fails semantic intent validation.

    Attributes
    ----------
    output : str
        The raw text that was validated.
    intent_name : str
        The ``Intent`` subclass name.
    intent_description : str
        The docstring / semantic requirement.
    score : float | None
        The similarity or confidence score returned by the judge
        (``None`` when the judge returns a binary yes/no).
    reason : str | None
        Human-readable explanation from the judge (e.g. "Judge answered: no").
        Used to build the ``semantix_feedback`` string on retries.
    """

    def __init__(
        self,
        output: str,
        intent_name: str,
        intent_description: str,
        score: float | None = None,
        reason: str | None = None,
    ) -> None:
        self.output = output
        self.intent_name = intent_name
        self.intent_description = intent_description
        self.score = score
        self.reason = reason
        score_part = f" (score={score:.4f})" if score is not None else ""
        super().__init__(
            f"Output failed semantic validation for "
            f"'{intent_name}'{score_part}.\n"
            f"Requirement: {intent_description}\n"
            f"Output (truncated): {output[:200]}"
        )
