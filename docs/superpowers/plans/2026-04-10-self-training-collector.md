# Self-Training Data Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in training data collector that captures correction pairs during self-healing retries and exports them in OpenAI or generic JSONL format.

**Architecture:** A `TrainingCollector` class appends correction pairs to an append-only JSONL file. The `@validate_intent` decorator calls `collector.record()` when a retry succeeds after a failure. Export methods convert the raw data to OpenAI fine-tuning format. A global default collector can be set via `set_default_collector()`.

**Tech Stack:** Python 3.10+, JSON, pathlib, no new dependencies

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `semantix/training/__init__.py` | Package init, re-exports TrainingCollector, set/get_default_collector |
| Create | `semantix/training/collector.py` | TrainingCollector: record(), stats(), file I/O |
| Create | `semantix/training/exporters.py` | export_openai(), export_generic() functions |
| Modify | `semantix/decorator.py` | Add collector hook at retry-success point |
| Create | `semantix/tests/test_training_collector.py` | Tests for TrainingCollector |
| Create | `semantix/tests/test_training_exporters.py` | Tests for export formats |
| Create | `semantix/tests/test_training_integration.py` | End-to-end decorator + collector test |

---

### Task 1: TrainingCollector — tests and implementation

**Files:**
- Create: `semantix/training/__init__.py`
- Create: `semantix/training/collector.py`
- Create: `semantix/tests/test_training_collector.py`

- [ ] **Step 1: Create the training package init**

```python
# semantix/training/__init__.py
"""Training data collection for continuous model improvement."""

from semantix.training.collector import TrainingCollector

_default_collector: TrainingCollector | None = None


def set_default_collector(collector: TrainingCollector | None) -> None:
    """Set the global default training data collector."""
    global _default_collector
    _default_collector = collector


def get_default_collector() -> TrainingCollector | None:
    """Return the global default training data collector, or None."""
    return _default_collector


__all__ = ["TrainingCollector", "set_default_collector", "get_default_collector"]
```

- [ ] **Step 2: Write tests for TrainingCollector**

```python
# semantix/tests/test_training_collector.py
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
    assert "T" in data["timestamp"]  # ISO format


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_training_collector.py -v`
Expected: FAIL — `ModuleNotFoundError` because `semantix.training.collector` doesn't exist yet.

- [ ] **Step 4: Implement TrainingCollector**

```python
# semantix/training/collector.py
"""TrainingCollector — append-only JSONL storage for correction pairs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TrainingCollector:
    """Collects (rejected, accepted) training pairs from self-healing retries.

    Each call to ``record()`` appends one JSONL line to disk.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The JSONL file path."""
        return self._path

    def record(
        self,
        *,
        intent: str,
        intent_description: str,
        rejected_output: str,
        rejected_score: float | None,
        rejected_reason: str | None,
        accepted_output: str,
        accepted_score: float | None,
        feedback: str,
        attempts: int,
    ) -> None:
        """Append a training pair to the JSONL file."""
        entry = {
            "intent": intent,
            "intent_description": intent_description,
            "rejected_output": rejected_output,
            "rejected_score": rejected_score,
            "rejected_reason": rejected_reason,
            "accepted_output": accepted_output,
            "accepted_score": accepted_score,
            "feedback": feedback,
            "attempts": attempts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def stats(self) -> dict:
        """Read the JSONL file and return aggregate counts."""
        if not self._path.exists():
            return {"total_pairs": 0, "intents": {}}
        intents: dict[str, int] = {}
        total = 0
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total += 1
                name = data["intent"]
                intents[name] = intents.get(name, 0) + 1
        return {"total_pairs": total, "intents": intents}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_training_collector.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add semantix/training/__init__.py semantix/training/collector.py semantix/tests/test_training_collector.py
git commit -m "feat: add TrainingCollector with append-only JSONL storage"
```

---

### Task 2: Exporters — tests and implementation

**Files:**
- Create: `semantix/training/exporters.py`
- Create: `semantix/tests/test_training_exporters.py`

- [ ] **Step 1: Write tests for exporters**

```python
# semantix/tests/test_training_exporters.py
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


# ── export_generic ──────────────────────────────────────────────────


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


# ── export_openai ──────────────────────────────────────────────────


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_training_exporters.py -v`
Expected: FAIL — `ImportError` because `semantix.training.exporters` doesn't exist yet.

- [ ] **Step 3: Implement exporters**

```python
# semantix/training/exporters.py
"""Export training data in various fine-tuning formats."""

from __future__ import annotations

import json
from pathlib import Path


def _read_records(source: Path, intent_filter: str | None = None) -> list[dict]:
    """Read JSONL records, optionally filtering by intent name."""
    records = []
    with open(source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if intent_filter and data["intent"] != intent_filter:
                continue
            records.append(data)
    return records


def export_generic(
    source: str | Path,
    destination: str | Path,
    intent_filter: str | None = None,
) -> None:
    """Copy training records to a new JSONL file, optionally filtering by intent."""
    records = _read_records(Path(source), intent_filter)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def export_openai(
    source: str | Path,
    destination: str | Path,
    intent_filter: str | None = None,
) -> None:
    """Convert training records to OpenAI fine-tuning chat JSONL format.

    Each record becomes a chat completion example with:
    - system: the intent description
    - user: a generic instruction
    - assistant: the accepted (corrected) output
    """
    records = _read_records(Path(source), intent_filter)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for record in records:
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You must satisfy the following requirement:\n\n"
                            + record["intent_description"]
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Generate a response that satisfies the above requirement.",
                    },
                    {
                        "role": "assistant",
                        "content": record["accepted_output"],
                    },
                ]
            }
            f.write(json.dumps(example) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_training_exporters.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add semantix/training/exporters.py semantix/tests/test_training_exporters.py
git commit -m "feat: add training data exporters — OpenAI and generic JSONL"
```

---

### Task 3: Decorator integration — tests and implementation

**Files:**
- Create: `semantix/tests/test_training_integration.py`
- Modify: `semantix/decorator.py`

- [ ] **Step 1: Write end-to-end integration tests**

```python
# semantix/tests/test_training_integration.py
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


# ── per-function collector ──────────────────────────────────────────


def test_collector_captures_on_retry_success(tmp_path: Path):
    collector = TrainingCollector(tmp_path / "training.jsonl")
    judge = FlipFlopJudge(fail_count=1, fail_score=0.3)

    @validate_intent(judge=judge, retries=1, collector=collector)
    def decline(event: str) -> ProfessionalDecline:
        return f"I must decline {event}."

    result = decline("the gala")
    assert isinstance(result, ProfessionalDecline)

    # Verify training pair was captured
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


# ── global default collector ────────────────────────────────────────


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


# ── async support ───────────────────────────────────────────────────


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


# ── with self-healing feedback ──────────────────────────────────────


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_training_integration.py -v`
Expected: FAIL — `validate_intent` doesn't accept `collector` parameter yet.

- [ ] **Step 3: Modify the decorator to accept and use a collector**

In `semantix/decorator.py`, make these changes:

**a)** Add `collector` parameter to `validate_intent` function signature. Change the overload signatures and the main function:

At the top of the file, add the import:
```python
from semantix.training import TrainingCollector, get_default_collector
```

Change the overloaded signatures (around lines 147-156):
```python
@overload
def validate_intent(func: F) -> F: ...


@overload
def validate_intent(
    *,
    judge: Judge,
    retries: int = ...,
    collector: TrainingCollector | None = ...,
) -> Callable[[F], F]: ...
```

Change the main function signature (around line 159):
```python
def validate_intent(
    func: F | None = None,
    *,
    judge: Judge | None = None,
    retries: int = 0,
    collector: TrainingCollector | None = None,
) -> F | Callable[[F], F]:
```

**b)** Inside the `decorator` inner function, resolve the collector:

After `_judge` is resolved (around line 235), add:
```python
        _collector = collector

        # Track last feedback string for collector
        _last_feedback: str | None = None
```

**c)** In the sync wrapper, after a successful `_run_judge` call that follows a failure (inside the `for attempt` loop, after `result = _run_judge(...)` succeeds and `last_err is not None`), add the collector hook:

In the sync wrapper, replace the success block (around lines 297-300):
```python
                try:
                    result = _run_judge(_judge, raw_str, intent_cls, attempt)
                    # Success — record training pair if this was a retry
                    if last_err is not None:
                        active_collector = _collector or get_default_collector()
                        if active_collector is not None:
                            active_collector.record(
                                intent=intent_cls.__name__,
                                intent_description=intent_cls.description(),
                                rejected_output=last_err.output,
                                rejected_score=last_err.score,
                                rejected_reason=last_err.reason,
                                accepted_output=raw_str,
                                accepted_score=result_verdict_score,
                                feedback=_last_feedback,
                                attempts=attempt,
                            )
                    _last_failure.set(None)
                    kwargs.pop("semantix_feedback", None)
                    return result
```

Wait — the `_run_judge` function returns an `Intent` instance, not a `Verdict`. We need the score from the verdict. The cleanest approach: modify `_run_judge` to also return the verdict score when we need it.

Actually, looking more carefully at the code, the simplest approach is to run the judge evaluation inline for the collector case. But that would duplicate logic. Instead, let's capture the verdict score by modifying `_run_judge` to return a tuple when needed.

**Better approach:** Extract the verdict score from the successful judge call. Since `_run_judge` calls `judge.evaluate()` internally and only returns the Intent, we need to get the score. The simplest way: store the last successful verdict score on the judge call.

**Simplest approach:** Just call `judge.evaluate()` one more time to get the score. No — that's wasteful.

**Actual simplest approach:** Modify `_run_judge` to return `(Intent, Verdict)` and update callers.

Change `_run_judge` (lines 108-139) to:

```python
def _run_judge(
    judge: Judge,
    raw_output: str,
    intent_cls: type[Intent],
    attempt: int = 1,
) -> tuple[Intent, Verdict]:
    """Validate *raw_output* against *intent_cls* using *judge*.

    Returns ``(Intent, Verdict)`` on success; raises ``SemanticIntentError`` on failure.
    """
    description = intent_cls.description()
    threshold = intent_cls.threshold

    with log_validation(
        intent_cls.__name__,
        attempt=attempt,
        output_preview=raw_output,
    ) as ctx:
        verdict: Verdict = judge.evaluate(raw_output, description, threshold)
        ctx["passed"] = verdict.passed
        ctx["score"] = verdict.score
        ctx["reason"] = verdict.reason

    if not verdict.passed:
        raise SemanticIntentError(
            output=raw_output,
            intent_name=intent_cls.__name__,
            intent_description=description,
            score=verdict.score,
            reason=verdict.reason,
        )
    return intent_cls(raw_output), verdict
```

Then update the sync wrapper success path:

```python
                try:
                    result, verdict = _run_judge(_judge, raw_str, intent_cls, attempt)
                    # Record training pair if this was a retry after failure
                    if last_err is not None:
                        active_collector = _collector or get_default_collector()
                        if active_collector is not None:
                            active_collector.record(
                                intent=intent_cls.__name__,
                                intent_description=intent_cls.description(),
                                rejected_output=last_err.output,
                                rejected_score=last_err.score,
                                rejected_reason=last_err.reason,
                                accepted_output=raw_str,
                                accepted_score=verdict.score,
                                feedback=_last_feedback,
                                attempts=attempt,
                            )
                    _last_failure.set(None)
                    kwargs.pop("semantix_feedback", None)
                    return result
```

And in the failure path, track the feedback:

```python
                except SemanticIntentError as err:
                    last_err = err
                    if attempt < max_attempts:
                        _last_failure.set(err)
                        if _fn_accepts_feedback:
                            _last_feedback = _build_feedback(err, attempt)
                            kwargs["semantix_feedback"] = _last_feedback
                        else:
                            _last_feedback = _build_feedback(err, attempt)
                        logger.info(
                            "Retry %d/%d for %s (score=%.4f)",
                            attempt,
                            retries,
                            intent_cls.__name__,
                            err.score or 0.0,
                        )
```

Do the same for the async wrapper — identical logic, just in `async_wrapper`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_training_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python3 -m pytest semantix/tests/ -v`
Expected: All tests PASS (existing tests may need minor update since `_run_judge` now returns a tuple).

If existing tests fail because `_run_judge` return type changed, update the calls in the sync/async wrappers to unpack the tuple: `result, verdict = _run_judge(...)` — but the `result` is what gets returned to the caller, so the public API is unchanged.

- [ ] **Step 6: Commit**

```bash
git add semantix/decorator.py semantix/tests/test_training_integration.py
git commit -m "feat: wire TrainingCollector into @validate_intent retry loop"
```

---

### Task 4: Full test suite, lint, and version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `semantix/__init__.py`

- [ ] **Step 1: Run the entire test suite**

Run: `python3 -m pytest semantix/tests/ -v`
Expected: All tests PASS. No regressions.

- [ ] **Step 2: Fix any issues found**

If any tests fail, fix the root cause and re-run.

- [ ] **Step 3: Bump version to 0.1.7**

In `pyproject.toml`, change `version = "0.1.6"` to `version = "0.1.7"`.
In `semantix/__init__.py`, change `__version__ = "0.1.6"` to `__version__ = "0.1.7"`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml semantix/__init__.py
git commit -m "chore: bump version to 0.1.7"
```
