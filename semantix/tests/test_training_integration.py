"""End-to-end tests: @validate_intent with TrainingCollector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from semantix.decorator import validate_intent
from semantix.intent import Intent
from semantix.tests.conftest import FlipFlopJudge, MockJudge
from semantix.training import TrainingCollector, get_default_collector, set_default_collector


class ProfessionalDecline(Intent):
    """The text must politely decline an invitation."""


def test_collector_captures_on_retry_success(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

    @validate_intent(judge=judge, retries=1, collector=collector)
    def decline(event: str) -> ProfessionalDecline:
        return f"I must decline {event}."

    result = decline("the gala")
    assert isinstance(result, ProfessionalDecline)

    lines = collector.path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["intent"] == "ProfessionalDecline"
    assert data["rejected_output"] == "I must decline the gala."
    assert data["accepted_output"] == "I must decline the gala."
    assert data["rejected_score"] == 0.3
    assert data["attempts"] == 2


def test_collector_not_called_on_first_attempt_success(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    judge = MockJudge(passed=True, score=0.95)

    @validate_intent(judge=judge, retries=1, collector=collector)
    def decline(event: str) -> ProfessionalDecline:
        return f"I must decline {event}."

    decline("the gala")
    assert not collector.path.exists()


def test_collector_not_called_when_all_retries_fail(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    judge = MockJudge(passed=False, score=0.2)

    @validate_intent(judge=judge, retries=1, collector=collector)
    def decline(event: str) -> ProfessionalDecline:
        return "No way!"

    with pytest.raises(Exception):
        decline("the gala")
    assert not collector.path.exists()


def test_no_collector_no_error(tmp_path: Path):
    judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

    @validate_intent(judge=judge, retries=1)
    def decline(event: str) -> ProfessionalDecline:
        return f"I must decline {event}."

    result = decline("the gala")
    assert isinstance(result, ProfessionalDecline)


def test_global_collector_captures(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    set_default_collector(collector)

    try:
        judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

        @validate_intent(judge=judge, retries=1)
        def decline(event: str) -> ProfessionalDecline:
            return f"I must decline {event}."

        decline("the gala")
        lines = collector.path.read_text().strip().split("\n")
        assert len(lines) == 1
    finally:
        set_default_collector(None)


def test_per_function_collector_overrides_global(tmp_path: Path):
    global_collector = TrainingCollector(tmp_path / "global.jsonl")
    local_collector = TrainingCollector(tmp_path / "local.jsonl")
    set_default_collector(global_collector)

    try:
        judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

        @validate_intent(judge=judge, retries=1, collector=local_collector)
        def decline(event: str) -> ProfessionalDecline:
            return f"I must decline {event}."

        decline("the gala")
        assert local_collector.path.exists()
        assert not global_collector.path.exists()
    finally:
        set_default_collector(None)


@pytest.mark.asyncio
async def test_async_collector_captures(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

    @validate_intent(judge=judge, retries=1, collector=collector)
    async def decline(event: str) -> ProfessionalDecline:
        return f"I must decline {event}."

    result = await decline("the gala")
    assert isinstance(result, ProfessionalDecline)

    lines = collector.path.read_text().strip().split("\n")
    assert len(lines) == 1


def test_collector_captures_feedback(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

    @validate_intent(judge=judge, retries=1, collector=collector)
    def decline(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
        return f"I must decline {event}."

    decline("the gala")
    data = json.loads(collector.path.read_text().strip())
    assert "feedback" in data
    assert data["feedback"] is not None
    assert "Self-Healing" in data["feedback"]
