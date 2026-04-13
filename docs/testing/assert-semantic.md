# assert_semantic

`assert_semantic()` is a one-line semantic assertion for any test runner -- pytest, unittest, or plain scripts.

## Basic usage

```python
from semantix.testing import assert_semantic

def test_chatbot_is_polite():
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

On failure:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

## Parameters

```python
assert_semantic(
    output: str,
    intent: str | type[Intent],
    *,
    judge: Judge | None = None,
    threshold: float | None = None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `output` | `str` | The text to validate |
| `intent` | `str` or `Intent` subclass | What the text should mean. A plain string or an Intent class. |
| `judge` | `Judge` or `None` | Judge backend. Defaults to QuantizedNLIJudge (falls back to NLIJudge). |
| `threshold` | `float` or `None` | Override the threshold. When `None`, uses the intent's threshold or the judge's `recommended_threshold`. |

## Using with Intent classes

```python
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

def test_polite():
    assert_semantic(response, Polite)
```

## Using with negation

```python
from semantix import Intent

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

def test_no_medical_advice():
    assert_semantic(response, ~MedicalAdvice)
```

## Choosing a judge

```python
from semantix import EmbeddingJudge, LLMJudge

# Fast cosine similarity (~5ms)
assert_semantic(response, "polite", judge=EmbeddingJudge())

# Accurate LLM-based scoring
assert_semantic(response, "polite", judge=LLMJudge(model="gpt-4o-mini"))
```

See [Judges](../judges.md) for all available judges.

## Setting a threshold

```python
# Require very high confidence
assert_semantic(response, "polite", threshold=0.95)

# Lenient check
assert_semantic(response, "polite", threshold=0.2)
```

## Threshold resolution order

1. Explicit `threshold` parameter
2. Intent class `threshold` (if set in the class `__dict__`, not inherited)
3. Judge's `recommended_threshold`
4. Intent base class default (`0.8`)

## How it works internally

When you pass a plain string as the intent, semantix creates a dynamic Intent subclass with that string as the docstring. This means the threshold resolution works the same way -- since the dynamic class has no explicit `threshold` in its `__dict__`, the judge's `recommended_threshold` applies.

## Related

- [pytest-semantix Plugin](pytest-plugin.md) -- fixtures, markers, and reports for pytest
- [Intents](../concepts/intents.md) -- defining reusable semantic contracts
- [Judges](../judges.md) -- available judge backends
