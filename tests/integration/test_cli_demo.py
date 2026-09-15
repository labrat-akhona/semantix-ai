"""Opt-in checks using the actual local model, downloaded once or cached."""

import pytest

from semantix.cli import main


@pytest.mark.integration
def test_demo_outcomes_match_real_quantized_judge():
    assert main(["demo", "--judge", "quantized", "--no-color"]) == 0
