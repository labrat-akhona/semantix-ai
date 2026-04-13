# Training

semantix turns your validation pipeline into a training pipeline. Every self-healing retry produces a labeled (rejected, accepted) pair that can be exported for fine-tuning.

```
Validate -> Fail -> Correct -> Capture -> Fine-tune -> Validate (fewer failures)
```

## TrainingCollector

`TrainingCollector` is an append-only JSONL writer that captures correction pairs from `@validate_intent` retries.

```python
from semantix.training import TrainingCollector
from semantix import validate_intent, Intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude."""

collector = TrainingCollector("training_data.jsonl")

@validate_intent(retries=2, collector=collector)
def decline(event: str) -> ProfessionalDecline:
    return call_my_llm(event)
```

When a retry succeeds, the collector records:

| Field | Description |
|---|---|
| `intent` | Intent class name |
| `intent_description` | The docstring |
| `rejected_output` | The output that failed validation |
| `rejected_score` | The score of the failed output |
| `rejected_reason` | Why it failed |
| `accepted_output` | The output that passed on retry |
| `accepted_score` | The score of the accepted output |
| `feedback` | The Markdown feedback that was injected |
| `attempts` | How many attempts it took |
| `timestamp` | ISO 8601 UTC timestamp |

### Stats

```python
stats = collector.stats()
print(stats)
# {"total_pairs": 42, "intents": {"ProfessionalDecline": 30, "Polite": 12}}
```

### Global default collector

Set a global collector that all `@validate_intent` decorators use (unless they specify their own):

```python
from semantix.training import set_default_collector, TrainingCollector

set_default_collector(TrainingCollector("global_training.jsonl"))
```

## Exporting for fine-tuning

### OpenAI format

```python
from semantix.training.exporters import export_openai

export_openai("training_data.jsonl", "finetune.jsonl")
```

Each record becomes a chat completion example:

```json
{
  "messages": [
    {"role": "system", "content": "You must satisfy the following requirement:\n\nThe text must politely decline..."},
    {"role": "user", "content": "Generate a response that satisfies the above requirement."},
    {"role": "assistant", "content": "Thank you for the invitation, but I'm unable to attend..."}
  ]
}
```

### Generic export

Copy and optionally filter records:

```python
from semantix.training.exporters import export_generic

# Export all records
export_generic("training_data.jsonl", "filtered.jsonl")

# Filter by intent
export_generic("training_data.jsonl", "decline_only.jsonl", intent_filter="ProfessionalDecline")
```

Both exporters support `intent_filter` to export data for a specific intent only.

## Threshold calibration

After collecting enough training data, calibrate per-intent thresholds from the data:

```python
from semantix.training.calibrate import calibrate_thresholds, apply_calibration

# Compute optimal thresholds
thresholds = calibrate_thresholds("training_data.jsonl")
print(thresholds)
# {"ProfessionalDecline": 0.45, "Polite": 0.52}
```

**How it works:** For each intent, the threshold is set to the midpoint between the highest rejected score and the lowest accepted score. This maximizes the separation between pass/fail examples. When there's overlap, it uses the average of all scores as a rough boundary.

### Applying calibrated thresholds

```python
apply_calibration(thresholds, {
    "ProfessionalDecline": ProfessionalDecline,
    "Polite": Polite,
})
```

This sets `threshold` directly on the Intent classes, overriding the default.

Returns the number of intents that were calibrated.

## Full workflow

```python
from semantix import Intent, validate_intent
from semantix.training import TrainingCollector
from semantix.training.exporters import export_openai
from semantix.training.calibrate import calibrate_thresholds, apply_calibration

# 1. Define intents
class Polite(Intent):
    """The text must be polite and professional."""

# 2. Collect training data
collector = TrainingCollector("training_data.jsonl")

@validate_intent(retries=2, collector=collector)
def respond(msg: str) -> Polite:
    return call_my_llm(msg)

# 3. Run in production -- correction pairs accumulate
for msg in customer_messages:
    try:
        respond(msg)
    except SemanticIntentError:
        pass  # log and handle

# 4. Export for fine-tuning
export_openai("training_data.jsonl", "finetune.jsonl")

# 5. Calibrate thresholds
thresholds = calibrate_thresholds("training_data.jsonl")
apply_calibration(thresholds, {"Polite": Polite})
```

## Related

- [Intents](concepts/intents.md) -- the semantic contracts being trained against
- [Judges](judges.md) -- the scoring backends that produce training data
- [Getting Started](getting-started.md) -- self-healing retries basics
