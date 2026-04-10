# Self-Training Data Collector Design

**Date:** 2026-04-10
**Author:** Akhona Eland + Claude
**Status:** Approved
**Version target:** v0.1.7

---

## Goal

Add an opt-in training data collector that captures (rejected_output, feedback, accepted_output) triples during `@validate_intent` self-healing retries. Data is appended to a JSONL file on disk and can be exported in OpenAI fine-tuning format or generic format for any pipeline.

## Why

The self-healing retry loop already produces before/after correction pairs — the model fails, gets feedback, and produces a better output. These pairs are fine-tuning gold, but they're thrown away after the call completes. Capturing them creates a continuous improvement flywheel: the more you use semantix, the better your model gets.

## Design Principles

- Opt-in only — training data collection must be explicitly enabled
- Zero impact on validation logic — the collector is a hook, not a change to the core
- Append-only JSONL on disk — crash-safe, grep-friendly, no data loss
- Export in multiple formats — store raw data, convert on export
- `@validate_intent` only — framework integrations don't produce correction pairs

---

## Components

### 1. TrainingCollector

A class that receives training pairs and appends them to a JSONL file.

```python
from semantix.training import TrainingCollector

collector = TrainingCollector("training_data.jsonl")
```

Each record appended to disk:

```json
{
  "intent": "ProfessionalDecline",
  "intent_description": "The text must politely decline an invitation without being rude or aggressive.",
  "rejected_output": "Get lost.",
  "rejected_score": 0.23,
  "rejected_reason": "Too aggressive",
  "accepted_output": "Thank you for the invitation, but I must respectfully decline.",
  "accepted_score": 0.94,
  "feedback": "## Semantix Self-Healing Feedback\n\nAttempt **1** failed validation...",
  "attempts": 2,
  "timestamp": "2026-04-10T12:00:00Z"
}
```

#### Public API

- `TrainingCollector(path: str | Path)` — constructor, creates or appends to the file
- `record(...)` — append a training pair (called by the decorator, not by users)
- `stats() -> dict` — returns `{"total_pairs": int, "intents": {name: count}}`
- `export_generic(path: str | Path)` — copies raw JSONL to a new file (filtering/formatting)
- `export_openai(path: str | Path)` — converts to OpenAI fine-tuning JSONL format

### 2. Opt-in Activation

Two ways to enable collection:

```python
# Per-function — pass collector to decorator
@validate_intent(retries=2, collector=collector)
def decline(event: str) -> ProfessionalDecline: ...

# Global — set a default collector for all decorated functions
from semantix.training import set_default_collector

set_default_collector(collector)

@validate_intent(retries=2)
def decline(event: str) -> ProfessionalDecline: ...
```

When both are set, the per-function collector takes precedence.

The global default is stored as a module-level variable in `semantix/training/__init__.py`, accessed by the decorator at call time (not decoration time), so it can be set/changed after functions are decorated.

### 3. Export Formats

#### Generic JSONL

Raw records as stored. `export_generic(path)` reads the source file and writes to the destination, optionally filtering by intent name.

#### OpenAI Fine-Tuning JSONL

Each training pair becomes a chat completion training example:

```json
{
  "messages": [
    {"role": "system", "content": "You must satisfy the following requirement:\n\nThe text must politely decline an invitation without being rude or aggressive."},
    {"role": "user", "content": "Generate a response that satisfies the above requirement."},
    {"role": "assistant", "content": "Thank you for the invitation, but I must respectfully decline."}
  ]
}
```

Note: semantix does not have access to the original user prompt. The system message uses the intent description as the training signal. The user message is a generic instruction. Only the accepted (successful) output is used as the assistant response — rejected outputs are excluded from the fine-tuning format.

### 4. Stats

```python
collector.stats()
# {"total_pairs": 142, "intents": {"ProfessionalDecline": 89, "Polite": 53}}
```

Reads the JSONL file and aggregates counts. Not cached — reads from disk each time for accuracy.

---

## Decorator Integration

The change to `decorator.py` is minimal. In the retry loop, when a retry succeeds after a previous failure:

1. `last_err` holds the `SemanticIntentError` (rejected output, score, reason, intent info)
2. `_build_feedback(err, attempt)` produced the feedback string
3. The new `raw_str` is the accepted output
4. The new `verdict` has the accepted score

At the success-after-failure point, if a collector is configured (per-function or global default), call:

```python
collector.record(
    intent=intent_cls.__name__,
    intent_description=description,
    rejected_output=last_err.output,
    rejected_score=last_err.score,
    rejected_reason=last_err.reason,
    accepted_output=raw_str,
    accepted_score=verdict.score,
    feedback=feedback_str,
    attempts=attempt,
)
```

This applies to both sync and async wrappers. No changes to the validation logic, judge evaluation, or error handling.

---

## File Structure

```
semantix/
  training/
    __init__.py       # TrainingCollector, set_default_collector, get_default_collector
    collector.py      # Core collection, storage, and stats
    exporters.py      # export_openai(), export_generic() functions
```

---

## Out of Scope

- No automatic fine-tuning API calls — user decides when and how to fine-tune
- No prompt reconstruction — semantix doesn't have access to the original user prompt
- No deduplication or filtering of training data — export everything, let the user curate
- No framework integration collection — only `@validate_intent` retries produce correction pairs
- No SQLite or database storage — append-only JSONL is sufficient
- No in-memory-only mode — always writes to disk for crash safety

---

## Testing Strategy

Test files under `semantix/tests/`:

- `test_training_collector.py` — tests for TrainingCollector: record, stats, file creation, append behavior
- `test_training_exporters.py` — tests for OpenAI and generic export formats
- `test_training_integration.py` — end-to-end test: decorator with collector enabled, verify training pair is captured on retry success

Tests use MockJudge and FlipFlopJudge (already in conftest.py) to control pass/fail behavior. No real LLM calls or model downloads needed.
