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


def test_unresolvable_annotations_warn_and_noop(caplog):
    """If get_type_hints() raises (forward ref, TYPE_CHECKING-only import),
    the decorator must warn instead of silently no-opping. See TrustMesh
    feedback #1 (docs/backlog/2026-05-15-trustmesh-feedback.md).
    """
    judge = MockJudge(passed=True)

    # Build a function whose annotation references a symbol that doesn't
    # exist at runtime — get_type_hints() will raise NameError.
    def make_fn():
        ns: dict = {}
        exec(
            "def fn(x: 'NonExistentType') -> 'AlsoMissing':\n    return 'hi'\n",
            ns,
        )
        return ns["fn"]

    fn = make_fn()

    with caplog.at_level("WARNING", logger="semantix"):
        wrapped = validate_intent(judge=judge)(fn)

    # Function still callable as a no-op (validation can't run).
    assert wrapped("anything") == "hi"
    assert judge.call_count == 0

    # The user must see a warning explaining why validation was skipped.
    assert any(
        "validate_intent" in rec.message and "no-op" in rec.message for rec in caplog.records
    ), f"expected no-op warning, got: {[r.message for r in caplog.records]}"


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
