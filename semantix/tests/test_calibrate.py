"""Tests for threshold calibration from training data."""

from __future__ import annotations

import json

import pytest

from semantix.intent import Intent
from semantix.training.calibrate import apply_calibration, calibrate_thresholds


class Polite(Intent):
    """The text must be polite and professional."""


class Formal(Intent):
    """The text must be extremely formal."""


def _write_training_data(path, records):
    """Helper to write JSONL training data."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_calibrate_clean_separation(tmp_path):
    """When rejected and accepted scores don't overlap, midpoint is used."""
    data_path = tmp_path / "data.jsonl"
    _write_training_data(
        data_path,
        [
            {
                "intent": "Polite",
                "rejected_score": 0.20,
                "accepted_score": 0.80,
                "intent_description": "polite",
                "rejected_output": "bad",
                "accepted_output": "good",
                "rejected_reason": None,
                "feedback": "",
                "attempts": 2,
                "timestamp": "2025-01-01",
            },
            {
                "intent": "Polite",
                "rejected_score": 0.30,
                "accepted_score": 0.90,
                "intent_description": "polite",
                "rejected_output": "bad2",
                "accepted_output": "good2",
                "rejected_reason": None,
                "feedback": "",
                "attempts": 2,
                "timestamp": "2025-01-01",
            },
        ],
    )
    result = calibrate_thresholds(data_path)
    # max_rejected=0.30, min_accepted=0.80 → midpoint=0.55
    assert result["Polite"] == pytest.approx(0.55)


def test_calibrate_overlap(tmp_path):
    """When scores overlap, average of all scores is used."""
    data_path = tmp_path / "data.jsonl"
    _write_training_data(
        data_path,
        [
            {
                "intent": "Polite",
                "rejected_score": 0.60,
                "accepted_score": 0.50,
                "intent_description": "polite",
                "rejected_output": "bad",
                "accepted_output": "good",
                "rejected_reason": None,
                "feedback": "",
                "attempts": 2,
                "timestamp": "2025-01-01",
            },
        ],
    )
    result = calibrate_thresholds(data_path)
    # overlap → average of [0.60, 0.50] = 0.55
    assert result["Polite"] == pytest.approx(0.55)


def test_calibrate_multiple_intents(tmp_path):
    """Different intents get different thresholds."""
    data_path = tmp_path / "data.jsonl"
    _write_training_data(
        data_path,
        [
            {
                "intent": "Polite",
                "rejected_score": 0.20,
                "accepted_score": 0.80,
                "intent_description": "polite",
                "rejected_output": "bad",
                "accepted_output": "good",
                "rejected_reason": None,
                "feedback": "",
                "attempts": 2,
                "timestamp": "2025-01-01",
            },
            {
                "intent": "Formal",
                "rejected_score": 0.40,
                "accepted_score": 0.90,
                "intent_description": "formal",
                "rejected_output": "bad",
                "accepted_output": "good",
                "rejected_reason": None,
                "feedback": "",
                "attempts": 2,
                "timestamp": "2025-01-01",
            },
        ],
    )
    result = calibrate_thresholds(data_path)
    assert "Polite" in result
    assert "Formal" in result
    assert result["Polite"] != result["Formal"]


def test_calibrate_missing_file(tmp_path):
    """Missing file returns empty dict."""
    result = calibrate_thresholds(tmp_path / "nonexistent.jsonl")
    assert result == {}


def test_calibrate_empty_file(tmp_path):
    """Empty file returns empty dict."""
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("")
    result = calibrate_thresholds(data_path)
    assert result == {}


def test_calibrate_skips_none_scores(tmp_path):
    """Records with None scores are skipped."""
    data_path = tmp_path / "data.jsonl"
    _write_training_data(
        data_path,
        [
            {
                "intent": "Polite",
                "rejected_score": None,
                "accepted_score": 0.80,
                "intent_description": "polite",
                "rejected_output": "bad",
                "accepted_output": "good",
                "rejected_reason": None,
                "feedback": "",
                "attempts": 2,
                "timestamp": "2025-01-01",
            },
        ],
    )
    result = calibrate_thresholds(data_path)
    # Only accepted scores, no rejected → can't compute threshold
    assert "Polite" not in result


def test_apply_calibration_sets_thresholds():
    """apply_calibration sets threshold on matching Intent classes."""
    thresholds = {"Polite": 0.55, "Unknown": 0.70}
    classes = {"Polite": Polite}
    count = apply_calibration(thresholds, classes)
    assert count == 1
    assert Polite.threshold == 0.55
    # Reset for other tests
    Polite.threshold = 0.8


def test_apply_calibration_returns_count():
    """apply_calibration returns number of intents calibrated."""
    thresholds = {"Polite": 0.55, "Formal": 0.65}
    classes = {"Polite": Polite, "Formal": Formal}
    count = apply_calibration(thresholds, classes)
    assert count == 2
    # Reset
    Polite.threshold = 0.8
    Formal.threshold = 0.8
