"""Tests for the Informed Self-Healing retry mechanism.

Verifies that @validate_intent automatically injects a ``semantix_feedback``
string into functions that declare the parameter, and that functions without
the parameter continue to work unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from semantix import Intent, SemanticIntentError, validate_intent
from semantix.decorator import _accepts_feedback, _build_feedback

from .conftest import FlipFlopJudge, MockJudge


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class SimpleIntent(Intent):
    """The text must be a clear and direct confirmation."""


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestAcceptsFeedback:
    def test_detects_parameter(self):
        def fn(x: str, semantix_feedback: Optional[str] = None) -> None: ...
        assert _accepts_feedback(fn) is True

    def test_returns_false_without_parameter(self):
        def fn(x: str) -> None: ...
        assert _accepts_feedback(fn) is False

    def test_returns_false_for_wrong_param_name(self):
        def fn(x: str, feedback: Optional[str] = None) -> None: ...
        assert _accepts_feedback(fn) is False

    def test_works_on_async_function(self):
        async def fn(x: str, semantix_feedback: Optional[str] = None) -> None: ...
        assert _accepts_feedback(fn) is True


class TestBuildFeedback:
    def setup_method(self):
        self.err = SemanticIntentError(
            output="This is fine.",
            intent_name="SimpleIntent",
            intent_description="The text must be a clear and direct confirmation.",
            score=0.3210,
            reason="Judge answered: no",
        )

    def test_contains_intent_name(self):
        fb = _build_feedback(self.err, attempt=1)
        assert "SimpleIntent" in fb

    def test_contains_score(self):
        fb = _build_feedback(self.err, attempt=1)
        assert "0.3210" in fb

    def test_contains_reason(self):
        fb = _build_feedback(self.err, attempt=1)
        assert "Judge answered: no" in fb

    def test_contains_requirement(self):
        fb = _build_feedback(self.err, attempt=1)
        assert "clear and direct confirmation" in fb

    def test_contains_rejected_output(self):
        fb = _build_feedback(self.err, attempt=1)
        assert "This is fine." in fb

    def test_contains_attempt_number(self):
        fb = _build_feedback(self.err, attempt=2)
        assert "2" in fb

    def test_no_reason_omits_reason_line(self):
        err_no_reason = SemanticIntentError(
            output="x", intent_name="X", intent_description="Y", score=0.1
        )
        fb = _build_feedback(err_no_reason, attempt=1)
        assert "Judge reason" not in fb

    def test_returns_markdown(self):
        fb = _build_feedback(self.err, attempt=1)
        assert "##" in fb  # Markdown heading present


# ---------------------------------------------------------------------------
# Integration tests — sync
# ---------------------------------------------------------------------------


class TestSyncInjection:
    def test_feedback_is_none_on_first_attempt(self):
        """First call must never receive feedback."""
        calls: list[Optional[str]] = []

        @validate_intent(judge=MockJudge(passed=True), retries=1)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        fn("hello")
        assert calls[0] is None

    def test_feedback_injected_on_retry(self):
        """Second attempt must receive a non-None feedback string."""
        calls: list[Optional[str]] = []

        @validate_intent(judge=FlipFlopJudge(fail_count=1), retries=1)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        fn("hello")
        assert len(calls) == 2
        assert calls[0] is None       # first attempt — no feedback
        assert calls[1] is not None   # retry — feedback injected

    def test_feedback_contains_intent_name(self):
        calls: list[Optional[str]] = []

        @validate_intent(judge=FlipFlopJudge(fail_count=1), retries=1)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        fn("hello")
        assert "SimpleIntent" in calls[1]

    def test_feedback_contains_score(self):
        calls: list[Optional[str]] = []

        @validate_intent(judge=FlipFlopJudge(fail_count=1, fail_score=0.1234), retries=1)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        fn("hello")
        assert "0.1234" in calls[1]

    def test_function_without_param_works_unchanged(self):
        """Functions without semantix_feedback must not receive it (no TypeError)."""
        call_count = 0

        @validate_intent(judge=FlipFlopJudge(fail_count=1), retries=1)
        def fn(x: str) -> SimpleIntent:
            nonlocal call_count
            call_count += 1
            return "confirmed"

        fn("hello")
        assert call_count == 2  # retried once, no crash

    def test_no_feedback_after_exhausted_retries(self):
        """kwargs must be cleaned up even when all retries fail."""
        @validate_intent(judge=MockJudge(passed=False), retries=1)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            return "bad output"

        with pytest.raises(SemanticIntentError):
            fn("hello")

    def test_feedback_cleared_after_success(self):
        """After a successful call, a second call must start with feedback=None."""
        calls: list[Optional[str]] = []

        @validate_intent(judge=FlipFlopJudge(fail_count=1), retries=2)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        fn("first")   # fails once then passes
        calls.clear()
        fn("second")  # fresh call — judge already passed from attempt 1
        assert calls[0] is None  # no feedback bleeds into fresh call

    def test_reason_in_feedback_when_judge_provides_it(self):
        calls: list[Optional[str]] = []

        failing_judge = MockJudge(passed=False, score=0.2, reason="too vague")
        passing_judge = MockJudge(passed=True, score=0.9)

        call_num = 0

        class AlternatingJudge(MockJudge):
            def evaluate(self, output, intent_description, threshold=0.8):
                nonlocal call_num
                call_num += 1
                if call_num == 1:
                    return failing_judge.evaluate(output, intent_description, threshold)
                return passing_judge.evaluate(output, intent_description, threshold)

        @validate_intent(judge=AlternatingJudge(), retries=1)
        def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        fn("hello")
        assert calls[1] is not None
        assert "too vague" in calls[1]


# ---------------------------------------------------------------------------
# Integration tests — async
# ---------------------------------------------------------------------------


class TestAsyncInjection:
    def test_async_feedback_is_none_on_first_attempt(self):
        calls: list[Optional[str]] = []

        @validate_intent(judge=MockJudge(passed=True), retries=1)
        async def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        asyncio.run(fn("hello"))
        assert calls[0] is None

    def test_async_feedback_injected_on_retry(self):
        calls: list[Optional[str]] = []

        @validate_intent(judge=FlipFlopJudge(fail_count=1), retries=1)
        async def fn(x: str, semantix_feedback: Optional[str] = None) -> SimpleIntent:
            calls.append(semantix_feedback)
            return "confirmed"

        asyncio.run(fn("hello"))
        assert calls[0] is None
        assert calls[1] is not None
        assert "SimpleIntent" in calls[1]

    def test_async_function_without_param_works_unchanged(self):
        call_count = 0

        @validate_intent(judge=FlipFlopJudge(fail_count=1), retries=1)
        async def fn(x: str) -> SimpleIntent:
            nonlocal call_count
            call_count += 1
            return "confirmed"

        asyncio.run(fn("hello"))
        assert call_count == 2
