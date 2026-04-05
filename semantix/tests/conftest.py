"""Reusable mock judge for tests — no API keys or heavy models needed."""

from __future__ import annotations

from semantix.judges import Judge, Verdict


class MockJudge(Judge):
    """Configurable mock judge for unit tests.

    Parameters
    ----------
    passed:
        The verdict to always return (or a callable for dynamic control).
    score:
        Score to attach to every verdict.
    """

    def __init__(
        self,
        passed: bool = True,
        score: float = 0.95,
        reason: str | None = None,
    ) -> None:
        self._passed = passed
        self._score = score
        self._reason = reason
        self.call_count = 0
        self.last_output: str | None = None
        self.last_description: str | None = None

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        self.call_count += 1
        self.last_output = output
        self.last_description = intent_description
        return Verdict(
            passed=self._passed,
            score=self._score,
            reason=self._reason,
        )


class FlipFlopJudge(Judge):
    """Fails the first *fail_count* calls, then passes.  Useful for testing retries."""

    def __init__(self, fail_count: int = 1, fail_score: float = 0.3) -> None:
        self._fail_count = fail_count
        self._fail_score = fail_score
        self.call_count = 0

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            return Verdict(passed=False, score=self._fail_score)
        return Verdict(passed=True, score=0.95)
