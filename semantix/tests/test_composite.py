"""Tests for composite intents (AllOf / AnyOf) and & / | operators."""

import pytest

from semantix.composite import AllOf, AnyOf, Not
from semantix.decorator import validate_intent
from semantix.exceptions import SemanticIntentError
from semantix.intent import Intent
from semantix.judges import Judge, Verdict
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite."""


class Positive(Intent):
    """The text must be positive."""

    threshold = 0.9


class Formal(Intent):
    """The text must be formal."""

    threshold = 0.7


# ── AllOf ───────────────────────────────────────────────────────────────


def test_allof_creates_combined_class():
    Combined = AllOf(Polite, Positive)
    assert issubclass(Combined, Intent)
    desc = Combined.description()
    assert "polite" in desc.lower()
    assert "positive" in desc.lower()
    assert "AND" in desc


def test_allof_threshold_is_min():
    Combined = AllOf(Polite, Positive, Formal)
    assert Combined.threshold == 0.7  # min(0.8, 0.9, 0.7)


def test_allof_needs_two():
    with pytest.raises(TypeError):
        AllOf(Polite)


# ── AnyOf ───────────────────────────────────────────────────────────────


def test_anyof_creates_combined_class():
    Combined = AnyOf(Polite, Positive)
    assert issubclass(Combined, Intent)
    desc = Combined.description()
    assert "OR" in desc


def test_anyof_threshold_is_max():
    Combined = AnyOf(Polite, Positive, Formal)
    assert Combined.threshold == 0.9  # max(0.8, 0.9, 0.7)


# ── operator syntax ────────────────────────────────────────────────────


def test_and_operator():
    Combined = Polite & Positive
    assert issubclass(Combined, Intent)
    assert "AND" in Combined.description()


def test_or_operator():
    Combined = Polite | Positive
    assert issubclass(Combined, Intent)
    assert "OR" in Combined.description()


# ── integration with decorator ──────────────────────────────────────────


def test_composite_intent_with_decorator():
    judge = MockJudge(passed=True, score=0.95)
    Combined = Polite & Positive

    @validate_intent(judge=judge)
    def respond(msg: str) -> Combined:  # type: ignore[valid-type]
        return "Thank you, that's wonderful!"

    result = respond("hello")
    assert isinstance(result, Combined)
    # Per-leaf evaluation: one call per leaf, not one for the composite.
    assert judge.call_count == 2


# ── per-leaf evaluation (TrustMesh feedback #2) ─────────────────────────


class _ScriptedJudge(Judge):
    """Returns verdicts keyed by intent description substring.

    Mirrors the TrustMesh calibration shape: a leaf description matches
    "bad" text strongly but "good" text weakly. Lets us prove that the
    decorator decomposes a composite into leaves before scoring, instead
    of feeding the whole multi-clause description as a single hypothesis.
    """

    recommended_threshold = 0.5

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, str]] = []

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.5,
    ) -> Verdict:
        self.calls.append((output, intent_description))
        score = next(
            (s for key, s in self._scores.items() if key in intent_description),
            0.0,
        )
        return Verdict(passed=score >= threshold, score=score)


class _MoneyAdvice(Intent):
    """Text gives advice about money."""


class _Slang(Intent):
    """Text uses emoji or internet slang."""


class _Aggression(Intent):
    """Text is aggressive or insulting."""


def test_composite_decomposes_into_per_leaf_calls():
    """A 3-leaf composite must call the judge once per leaf, not once for
    the concatenated description. See docs/backlog/2026-05-15-trustmesh-feedback.md #2.
    """
    judge = _ScriptedJudge(
        scores={"money": 0.04, "emoji": 0.07, "aggressive": 0.05},
    )
    Safe = ~_MoneyAdvice & ~_Slang & ~_Aggression

    @validate_intent(judge=judge)
    def reply(msg: str) -> Safe:  # type: ignore[valid-type]
        return "Thanks for your message — I'll get back to you shortly."

    result = reply("hi")
    assert str(result).startswith("Thanks")

    # One call per leaf, not one for the composite description.
    assert len(judge.calls) == 3
    descriptions = [desc for _, desc in judge.calls]
    assert any("money" in d for d in descriptions)
    assert any("emoji" in d for d in descriptions)
    assert any("aggressive" in d for d in descriptions)
    # Crucially, no call should pass the whole "ALL of the following" concatenation.
    assert all("ALL of the following" not in d for d in descriptions)


def test_composite_fails_when_any_leaf_matches_negated_property():
    """If the output matches a leaf the composite negates, validation must fail."""
    judge = _ScriptedJudge(
        scores={"money": 0.72, "emoji": 0.07, "aggressive": 0.05},
    )
    Safe = ~_MoneyAdvice & ~_Slang & ~_Aggression

    @validate_intent(judge=judge)
    def reply(msg: str) -> Safe:  # type: ignore[valid-type]
        return "You should invest in index funds."

    with pytest.raises(SemanticIntentError):
        reply("what should I do with my savings?")


def test_anyof_passes_when_one_leaf_passes():
    """AnyOf composite: any single passing leaf is enough."""
    judge = _ScriptedJudge(scores={"polite": 0.9, "positive": 0.1})
    Either = Polite | Positive

    @validate_intent(judge=judge)
    def reply(msg: str) -> Either:  # type: ignore[valid-type]
        return "Good morning, hope you're well."

    result = reply("hi")
    assert isinstance(result, Either)
    # Two calls (one per leaf), polite leaf passes.
    assert len(judge.calls) == 2
