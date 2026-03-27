"""Tests for the @validate_intent decorator including retries."""

import asyncio

import pytest

from semantix.decorator import validate_intent
from semantix.exceptions import SemanticIntentError
from semantix.intent import Intent
from semantix.tests.conftest import FlipFlopJudge, MockJudge


class ProfessionalDecline(Intent):
    """The text must politely decline an invitation."""


# ── basic validation ────────────────────────────────────────────────────


def test_validate_intent_passes():
    judge = MockJudge(passed=True, score=0.9)

    @validate_intent(judge=judge)
    def decline(event: str) -> ProfessionalDecline:
        return f"Sorry, I can't attend {event}."

    result = decline("the gala")
    assert isinstance(result, ProfessionalDecline)
    assert result.text == "Sorry, I can't attend the gala."
    assert judge.call_count == 1


def test_validate_intent_raises_on_failure():
    judge = MockJudge(passed=False, score=0.3)

    @validate_intent(judge=judge)
    def decline(event: str) -> ProfessionalDecline:
        return "No way!"

    with pytest.raises(SemanticIntentError) as exc_info:
        decline("the gala")

    assert exc_info.value.score == 0.3
    assert exc_info.value.intent_name == "ProfessionalDecline"


def test_no_intent_return_type_is_noop():
    judge = MockJudge(passed=True)

    @validate_intent(judge=judge)
    def plain() -> str:
        return "hello"

    assert plain() == "hello"
    assert judge.call_count == 0  # never invoked


# ── retries ─────────────────────────────────────────────────────────────


def test_retry_succeeds_on_second_attempt():
    judge = FlipFlopJudge(fail_count=1)
    call_counter = {"n": 0}

    @validate_intent(judge=judge, retries=2)
    def decline(event: str) -> ProfessionalDecline:
        call_counter["n"] += 1
        return f"Attempt {call_counter['n']}: Sorry about {event}."

    result = decline("the party")
    assert isinstance(result, ProfessionalDecline)
    assert call_counter["n"] == 2
    assert judge.call_count == 2


def test_retry_exhausted_raises():
    judge = MockJudge(passed=False, score=0.2)

    @validate_intent(judge=judge, retries=2)
    def decline(event: str) -> ProfessionalDecline:
        return "Nope"

    with pytest.raises(SemanticIntentError):
        decline("the gala")

    # 1 initial + 2 retries = 3
    assert judge.call_count == 3


# ── async support ───────────────────────────────────────────────────────


def test_async_validate_intent_passes():
    judge = MockJudge(passed=True, score=0.92)

    @validate_intent(judge=judge)
    async def decline(event: str) -> ProfessionalDecline:
        return f"Sorry, I can't attend {event}."

    result = asyncio.run(decline("the gala"))
    assert isinstance(result, ProfessionalDecline)


def test_async_retry_succeeds():
    judge = FlipFlopJudge(fail_count=1)

    @validate_intent(judge=judge, retries=1)
    async def decline(event: str) -> ProfessionalDecline:
        return f"Sorry about {event}."

    result = asyncio.run(decline("the party"))
    assert isinstance(result, ProfessionalDecline)
    assert judge.call_count == 2
