# pytest-semantix Plugin

`pytest-semantix` is a separate pytest plugin that wraps `assert_semantic()` as a fixture with markers, CLI options, and test reports.

## Install

```bash
pip install pytest-semantix
```

This also installs `semantix-ai` as a dependency.

## The `assert_semantic` fixture

```python
def test_chatbot_is_polite(assert_semantic):
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

The fixture wraps `semantix.testing.assert_semantic` and tracks results for reporting.

### Parameters

```python
assert_semantic(
    output: str,
    intent: str | type[Intent] | None = None,
    *,
    judge: Judge | None = None,
    threshold: float | None = None,
)
```

When `intent` is `None`, it falls back to the `@pytest.mark.semantic` marker (see below).

## Markers

Use `@pytest.mark.semantic` to attach an intent to a test:

```python
import pytest

@pytest.mark.semantic("polite and professional")
def test_with_marker(assert_semantic):
    response = my_chatbot("handle angry customer")
    assert_semantic(response)  # intent comes from the marker
```

This is useful when you want to separate the intent definition from the assertion call, or when using parametrized tests.

## Intent classes

Reuse intents across tests:

```python
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

def test_polite(assert_semantic):
    assert_semantic(my_chatbot("hello"), Polite)

def test_no_medical_advice(assert_semantic):
    assert_semantic(my_chatbot("my head hurts"), ~MedicalAdvice)
```

## CLI options

```
--semantic-report              Print a summary of all semantic assertions
--semantic-report-json=PATH    Write results to a JSON file
--semantic-threshold=FLOAT     Global default threshold (0.0-1.0)
```

### Terminal report

```bash
pytest --semantic-report
```

```
======================== semantic assertion report =========================
  Total: 5  |  Passed: 4  |  Failed: 1

  [PASS] tests/test_bot.py::test_polite  [12ms]
  [PASS] tests/test_bot.py::test_helpful  [14ms]
  [FAIL] tests/test_bot.py::test_no_pii  (score=0.67)  Contains email address  [11ms]
  [PASS] tests/test_bot.py::test_on_topic  [13ms]
  [PASS] tests/test_bot.py::test_concise  [15ms]

============================================================================
```

### JSON report

```bash
pytest --semantic-report-json=semantic-results.json
```

```json
{
  "summary": { "total": 5, "passed": 4, "failed": 1 },
  "results": [
    {
      "nodeid": "tests/test_bot.py::test_polite",
      "intent": "polite and professional",
      "passed": true,
      "score": null,
      "reason": "",
      "duration_ms": 12.3
    }
  ]
}
```

## CI integration

Add semantic tests to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run tests with semantic report
  run: pytest --semantic-report --semantic-report-json=semantic-results.json

- name: Upload semantic report
  uses: actions/upload-artifact@v4
  with:
    name: semantic-report
    path: semantic-results.json
```

Set a global threshold for CI:

```bash
pytest --semantic-threshold=0.5 --semantic-report
```

## Threshold resolution

The fixture resolves thresholds in this order:

1. Explicit `threshold` parameter in the `assert_semantic()` call
2. `--semantic-threshold` CLI option
3. Let semantix decide (judge's `recommended_threshold` or intent default)

## Related

- [assert_semantic](assert-semantic.md) -- the underlying assertion function
- [Intents](../concepts/intents.md) -- defining reusable semantic contracts
- [Judges](../judges.md) -- available judge backends
