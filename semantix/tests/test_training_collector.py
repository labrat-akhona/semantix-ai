"""Tests for the TrainingCollector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantix.training.collector import TrainingCollector


@pytest.fixture
def collector(tmp_path: Path) -> TrainingCollector:
    return TrainingCollector(tmp_path / "training.jsonl")


@pytest.fixture
def sample_record() -> dict:
    return {
        "intent": "ProfessionalDecline",
        "intent_description": "The text must politely decline.",
        "rejected_output": "Get lost.",
        "rejected_score": 0.23,
        "rejected_reason": "Too aggressive",
        "accepted_output": "Thank you, but I must decline.",
        "accepted_score": 0.94,
        "feedback": "## Semantix Self-Healing Feedback\n\nAttempt 1 failed.",
        "attempts": 2,
    }


def test_record_creates_file(collector: TrainingCollector, sample_record: dict):
    collector.record(**sample_record)
    assert collector.path.exists()


def test_record_appends_valid_jsonl(collector: TrainingCollector, sample_record: dict):
    collector.record(**sample_record)
    lines = collector.path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["intent"] == "ProfessionalDecline"
    assert data["rejected_output"] == "Get lost."
    assert data["accepted_output"] == "Thank you, but I must decline."
    assert data["rejected_score"] == 0.23
    assert data["accepted_score"] == 0.94
    assert "timestamp" in data


def test_record_appends_multiple(collector: TrainingCollector, sample_record: dict):
    collector.record(**sample_record)
    collector.record(**sample_record)
    lines = collector.path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_record_adds_timestamp(collector: TrainingCollector, sample_record: dict):
    collector.record(**sample_record)
    data = json.loads(collector.path.read_text().strip())
    assert "timestamp" in data
    assert "T" in data["timestamp"]


def test_stats_empty(collector: TrainingCollector):
    result = collector.stats()
    assert result == {"total_pairs": 0, "intents": {}}


def test_stats_counts(collector: TrainingCollector, sample_record: dict):
    collector.record(**sample_record)
    collector.record(**sample_record)
    sample_record["intent"] = "Polite"
    collector.record(**sample_record)
    result = collector.stats()
    assert result["total_pairs"] == 3
    assert result["intents"]["ProfessionalDecline"] == 2
    assert result["intents"]["Polite"] == 1


def test_stats_no_file(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "nonexistent.jsonl")
    result = collector.stats()
    assert result == {"total_pairs": 0, "intents": {}}


def test_collector_creates_parent_dirs(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "sub" / "dir" / "training.jsonl")
    collector.record(
        intent="Test",
        intent_description="Test intent.",
        rejected_output="bad",
        rejected_score=0.1,
        rejected_reason=None,
        accepted_output="good",
        accepted_score=0.9,
        feedback="fix it",
        attempts=2,
    )
    assert collector.path.exists()
