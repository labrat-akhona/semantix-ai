"""End-to-end integration test: download real POPIAJudge and validate semantics.

This test:
  - downloads the real labrat-akhona/nli-popia-v1 model from HuggingFace
  - runs 3 POPIA presets against hand-chosen outputs
  - verifies verdicts match the expected POPIA semantics

Runs only when pytest is invoked with `-m integration`. Excluded from the
default suite because it requires network access and ~25MB of downloads.
"""

from __future__ import annotations

import pytest

from semantix.judges.popia import POPIAJudge
from semantix.presets.popia import (
    POPIA_BREACH,
    POPIA_CONSENT,
    POPIA_CROSS_BORDER,
)


def _check(judge: POPIAJudge, output: str, preset) -> bool:
    """Evaluate `output` against a preset intent, honouring preset.negate."""
    threshold = preset.threshold if preset.threshold is not None else judge.recommended_threshold
    verdict = judge.evaluate(output, preset.description, threshold=threshold)
    return (not verdict.passed) if preset.negate else verdict.passed


@pytest.fixture(scope="module")
def popia_judge():
    return POPIAJudge()


@pytest.mark.integration
def test_consent_positive(popia_judge):
    output = "I confirm I have read and agree to the privacy terms."
    assert _check(popia_judge, output, POPIA_CONSENT) is True


@pytest.mark.integration
def test_cross_border_positive_detected(popia_judge):
    output = (
        "Customer records stored in our Frankfurt AWS region, replicated nightly "
        "to the Virginia us-east-1 cluster for disaster recovery."
    )
    assert _check(popia_judge, output, POPIA_CROSS_BORDER) is True


@pytest.mark.integration
def test_breach_negated_intent_fires_on_delayed_notification(popia_judge):
    output = "We'll notify affected users in the next quarterly newsletter."
    # POPIA_BREACH is a negated intent ("must NOT delay"). A delayed-notification
    # output matches the breach description → negate flips → _check returns False.
    assert _check(popia_judge, output, POPIA_BREACH) is False
