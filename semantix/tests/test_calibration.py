"""Tests for judge-aware threshold calibration."""

from __future__ import annotations

from semantix.composite import AllOf, AnyOf
from semantix.intent import Intent
from semantix.judges import Judge
from semantix.judges.caching import CachingJudge
from semantix.judges.forensic import ForensicJudge

from .conftest import MockJudge

# -- Helpers ------------------------------------------------------------------


class CalibratedJudge(MockJudge):
    """MockJudge with a recommended_threshold set."""

    recommended_threshold = 0.3


class UncalibratedJudge(MockJudge):
    """MockJudge without a recommended_threshold (uses default None)."""

    pass


class DefaultIntent(Intent):
    """A test intent with no explicit threshold."""


class ExplicitIntent(Intent):
    """A test intent with an explicit threshold."""

    threshold = 0.85


class AnotherDefault(Intent):
    """Another test intent with no explicit threshold."""


# -- Tests: _run_judge threshold resolution -----------------------------------


class TestThresholdResolution:
    """Verify that _run_judge uses the judge's recommended_threshold
    when the Intent doesn't explicitly set one."""

    def test_default_intent_uses_judge_threshold(self):
        """When Intent has no explicit threshold, use judge's recommended_threshold."""
        from semantix.decorator import _run_judge

        judge = CalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello world", DefaultIntent)
        assert judge.last_threshold == 0.3

    def test_explicit_intent_overrides_judge_threshold(self):
        """When Intent explicitly sets threshold, ignore judge's recommended_threshold."""
        from semantix.decorator import _run_judge

        judge = CalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello world", ExplicitIntent)
        assert judge.last_threshold == 0.85

    def test_uncalibrated_judge_uses_intent_default(self):
        """When judge has no recommended_threshold, fall back to Intent's default (0.8)."""
        from semantix.decorator import _run_judge

        judge = UncalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello world", DefaultIntent)
        assert judge.last_threshold == 0.8

    def test_explicit_intent_with_uncalibrated_judge(self):
        """Explicit threshold always wins, even with uncalibrated judge."""
        from semantix.decorator import _run_judge

        judge = UncalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello world", ExplicitIntent)
        assert judge.last_threshold == 0.85


# -- Tests: Composite intents ------------------------------------------------


class TestCompositeCalibration:
    """Verify composites don't force threshold into __dict__ when not needed."""

    def test_allof_default_intents_uses_judge_threshold(self):
        """AllOf of default-threshold intents should let judge threshold apply."""
        from semantix.decorator import _run_judge

        Composite = AllOf(DefaultIntent, AnotherDefault)
        judge = CalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello", Composite)
        assert judge.last_threshold == 0.3

    def test_anyof_default_intents_uses_judge_threshold(self):
        """AnyOf of default-threshold intents should let judge threshold apply."""
        from semantix.decorator import _run_judge

        Composite = AnyOf(DefaultIntent, AnotherDefault)
        judge = CalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello", Composite)
        assert judge.last_threshold == 0.3

    def test_allof_with_explicit_uses_explicit(self):
        """AllOf where one component has explicit threshold uses that."""
        from semantix.decorator import _run_judge

        Composite = AllOf(DefaultIntent, ExplicitIntent)
        judge = CalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello", Composite)
        assert judge.last_threshold == 0.85

    def test_anyof_with_explicit_uses_explicit(self):
        """AnyOf where one component has explicit threshold uses that."""
        from semantix.decorator import _run_judge

        Composite = AnyOf(DefaultIntent, ExplicitIntent)
        judge = CalibratedJudge(passed=True, score=0.95)
        _run_judge(judge, "hello", Composite)
        assert judge.last_threshold == 0.85


# -- Tests: Wrapper judges delegate -------------------------------------------


class TestWrapperDelegation:
    """CachingJudge and ForensicJudge should delegate recommended_threshold."""

    def test_caching_judge_delegates(self):
        calibrated = CalibratedJudge(passed=True, score=0.95)
        caching = CachingJudge(calibrated)
        assert caching.recommended_threshold == 0.3

    def test_caching_judge_delegates_none(self):
        uncalibrated = UncalibratedJudge(passed=True, score=0.95)
        caching = CachingJudge(uncalibrated)
        assert caching.recommended_threshold is None

    def test_forensic_judge_delegates(self):
        calibrated = CalibratedJudge(passed=True, score=0.95)
        forensic = ForensicJudge(calibrated)
        assert forensic.recommended_threshold == 0.3

    def test_forensic_judge_delegates_none(self):
        uncalibrated = UncalibratedJudge(passed=True, score=0.95)
        forensic = ForensicJudge(uncalibrated)
        assert forensic.recommended_threshold is None


# -- Tests: Judge class attributes --------------------------------------------


class TestJudgeDefaults:
    """Verify each judge declares the expected recommended_threshold."""

    def test_base_judge_has_none(self):
        assert Judge.recommended_threshold is None

    def test_mock_judge_inherits_none(self):
        j = MockJudge()
        assert j.recommended_threshold is None
