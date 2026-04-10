"""Tests for training data export formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantix.training.collector import TrainingCollector
from semantix.training.exporters import export_generic, export_openai


@pytest.fixture
def populated_collector(tmp_path: Path) -> TrainingCollector:
    collector = TrainingCollector(tmp_path / "training.jsonl")
    collector.record(
        intent="ProfessionalDecline",
        intent_description="The text must politely decline an invitation.",
        rejected_output="Get lost.",
        rejected_score=0.23,
        rejected_reason="Too aggressive",
        accepted_output="Thank you, but I must decline.",
        accepted_score=0.94,
        feedback="## Feedback\n\nAttempt 1 failed.",
        attempts=2,
    )
    collector.record(
        intent="Polite",
        intent_description="The text must be polite.",
        rejected_output="Whatever.",
        rejected_score=0.4,
        rejected_reason="Not polite",
        accepted_output="I appreciate your time.",
        accepted_score=0.91,
        feedback="## Feedback\n\nAttempt 1 failed.",
        attempts=2,
    )
    return collector


def test_export_generic_copies_all_records(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "output.jsonl"
    export_generic(populated_collector.path, output)
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2


def test_export_generic_preserves_data(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "output.jsonl"
    export_generic(populated_collector.path, output)
    data = json.loads(output.read_text().strip().split("\n")[0])
    assert data["intent"] == "ProfessionalDecline"
    assert data["rejected_output"] == "Get lost."
    assert data["accepted_output"] == "Thank you, but I must decline."


def test_export_generic_filter_by_intent(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "output.jsonl"
    export_generic(populated_collector.path, output, intent_filter="Polite")
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["intent"] == "Polite"


def test_export_openai_format(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "finetune.jsonl"
    export_openai(populated_collector.path, output)
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2
    data = json.loads(lines[0])
    assert "messages" in data
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][1]["role"] == "user"
    assert data["messages"][2]["role"] == "assistant"


def test_export_openai_system_contains_intent(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "finetune.jsonl"
    export_openai(populated_collector.path, output)
    data = json.loads(output.read_text().strip().split("\n")[0])
    assert "politely decline" in data["messages"][0]["content"]


def test_export_openai_assistant_is_accepted_output(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "finetune.jsonl"
    export_openai(populated_collector.path, output)
    data = json.loads(output.read_text().strip().split("\n")[0])
    assert data["messages"][2]["content"] == "Thank you, but I must decline."


def test_export_openai_filter_by_intent(populated_collector: TrainingCollector, tmp_path: Path):
    output = tmp_path / "finetune.jsonl"
    export_openai(populated_collector.path, output, intent_filter="ProfessionalDecline")
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1
