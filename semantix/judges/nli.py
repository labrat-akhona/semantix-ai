"""NLI-based judge using a cross-encoder entailment model.

Uses Natural Language Inference (NLI) to check whether an output *satisfies*
an intent requirement — a fundamentally more accurate approach than cosine
similarity for intent validation.

Requires: pip install sentence-transformers
"""

from __future__ import annotations

from semantix.judges import Judge, Verdict


def _conjugate_third_person(verb: str) -> str:
    """Conjugate a base-form verb to third-person singular present.

    E.g. "decline" → "declines", "express" → "expresses", "carry" → "carries".
    """
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _to_hypothesis(description: str) -> str:
    """Convert a requirement description into an NLI hypothesis.

    Strips imperative framing and reframes as a progressive action
    so the cross-encoder can reason about it naturally.

    E.g. "The text must politely decline..." → "Someone is politely declining..."

    The progressive ("is declining") framing scores significantly higher on
    NLI cross-encoders than declarative ("declines") because the model treats
    the premise as evidence of an ongoing action.
    """
    d = description.strip()
    for prefix in ("The text must ", "The response must ", "The output must "):
        if d.startswith(prefix):
            remainder = d[len(prefix) :]
            words = remainder.split()
            # Skip adverbs (words ending in -ly) to find the verb.
            for i, word in enumerate(words):
                if not word.endswith("ly"):
                    words[i] = _to_progressive(word)
                    break
            return f"Someone is {' '.join(words)}"
    return d


def _to_progressive(verb: str) -> str:
    """Convert a base-form verb to present participle (-ing form).

    E.g. "decline" → "declining", "express" → "expressing", "carry" → "carrying".
    """
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    if (
        len(verb) >= 3
        and verb[-1] not in "aeiouwxy"
        and verb[-2] in "aeiou"
        and verb[-3] not in "aeiou"
    ):
        return verb + verb[-1] + "ing"
    return verb + "ing"


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

    recommended_threshold = 0.3

    def __init__(self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768") -> None:
        from sentence_transformers import CrossEncoder  # noqa: WPS433

        self._model: CrossEncoder = CrossEncoder(model_name)

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.5,
    ) -> Verdict:
        """Score how strongly *output* satisfies *intent_description*.

        The cross-encoder returns logits for [contradiction, entailment, neutral].
        We softmax them and use the entailment probability as the pass/fail signal.
        """
        hypothesis = _to_hypothesis(intent_description)
        # premise = output, hypothesis = what we require it to mean
        scores = self._model.predict([(output, hypothesis)], apply_softmax=True)
        # Label order: {0: contradiction, 1: entailment, 2: neutral}
        entailment_score = float(scores[0][1])
        return Verdict(
            passed=entailment_score >= threshold,
            score=entailment_score,
        )
