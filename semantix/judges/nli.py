"""NLI-based judge using a cross-encoder entailment model.

Uses Natural Language Inference (NLI) to check whether an output *satisfies*
an intent requirement — a fundamentally more accurate approach than cosine
similarity for intent validation.

Requires: pip install sentence-transformers
"""

from __future__ import annotations

from semantix.judges import Judge, Verdict


def _to_hypothesis(description: str) -> str:
    """Convert a requirement description into an NLI hypothesis.

    Strips imperative framing so the cross-encoder can reason about it cleanly.
    E.g. "The text must politely decline..." → "The text politely declines..."
    """
    d = description.strip()
    for prefix in ("The text must ", "The response must ", "The output must "):
        if d.startswith(prefix):
            verb = d[len(prefix):]
            # "politely decline" → keep as-is, just change framing
            return f"This text {verb}"
    return d


class NLIJudge(Judge):
    """Judge that uses a cross-encoder NLI model to determine if an output
    satisfies an intent requirement.

    Checks whether the output *entails* the intent description — much more
    accurate than cosine similarity for intent validation, and still runs
    fully locally with no API key.

    Parameters
    ----------
    model_name:
        Any NLI cross-encoder model from sentence-transformers.
        Defaults to ``"cross-encoder/nli-MiniLM2-L6-H768"`` (~85 MB, fast).
    """

    def __init__(
        self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"
    ) -> None:
        from sentence_transformers import CrossEncoder  # noqa: WPS433

        self._model: CrossEncoder = CrossEncoder(model_name)

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.5,
    ) -> Verdict:
        """Score how strongly *output* satisfies *intent_description*.

        The cross-encoder returns [contradiction, neutral, entailment]
        probabilities. We use the entailment score as the pass/fail signal.
        """
        hypothesis = _to_hypothesis(intent_description)
        # premise = output, hypothesis = what we require it to mean
        scores = self._model.predict([(output, hypothesis)])
        # scores shape: (1, 3) → [contradiction, neutral, entailment]
        entailment_score = float(scores[0][2])
        return Verdict(
            passed=entailment_score >= threshold,
            score=entailment_score,
        )
