"""Tests for SemanticIntentError."""

from semantix.exceptions import SemanticIntentError


def test_error_attributes():
    err = SemanticIntentError(
        output="bad text",
        intent_name="Polite",
        intent_description="Must be polite.",
        score=0.42,
    )
    assert err.output == "bad text"
    assert err.intent_name == "Polite"
    assert err.intent_description == "Must be polite."
    assert err.score == 0.42
    assert "0.4200" in str(err)
    assert "Polite" in str(err)


def test_error_without_score():
    err = SemanticIntentError(
        output="x",
        intent_name="Test",
        intent_description="desc",
    )
    assert err.score is None
    assert "score=" not in str(err)
