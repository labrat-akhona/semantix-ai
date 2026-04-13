# Test Your LLM Outputs in pytest (15ms, No API Key)

You've got an LLM-powered feature in production. You want to test it. Here are your options:

1. **String matching.** Works until the model rephrases "I'd be happy to help" as "Sure, let me assist you." Now your test is red and nothing is actually wrong.

2. **Regex.** You write a pattern. It passes today, breaks tomorrow when the model adds a comma. You write a more permissive pattern. Now it passes on garbage too.

3. **LLM-as-judge.** Call GPT-4 to evaluate the output. Your test suite now takes 4 minutes, costs money, and fails when OpenAI has a bad day. Your CI pipeline needs an API key in secrets. Your team stops running the tests locally.

None of these are good. What you actually want is to test whether your output *means* the right thing — without any of that overhead.

---

## pytest-semantix

```bash
pip install pytest-semantix
```

```python
def test_chatbot_is_polite(assert_semantic):
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

That's a real pytest test. It runs locally on CPU in ~15ms. No API key. No network calls. No tokens burned.

`pytest-semantix` is a pytest plugin that wraps [semantix-ai](https://pypi.org/project/semantix-ai/)'s semantic assertion engine as a native fixture. Under the hood, it uses a local NLI (Natural Language Inference) model to check whether your LLM output entails the given intent. You describe what you mean in plain English. The model checks entailment. Done.

On failure, you get a score, the intent, and a reason — not just a raw traceback:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

---

## Markers

If you want to attach an intent to the test itself rather than the assertion call, use the `@pytest.mark.semantic` marker:

```python
import pytest

@pytest.mark.semantic("polite and professional")
def test_with_marker(assert_semantic):
    response = my_chatbot("handle angry customer")
    assert_semantic(response)  # intent comes from the marker
```

This is useful when you have a single intent per test and want to see it at a glance in the decorator rather than buried in the function body.

---

## Terminal Reports

Pass `--semantic-report` and you get a color-coded summary after the test session:

```
$ pytest --semantic-report

======================== semantic assertion report =========================
  Total: 5  |  Passed: 4  |  Failed: 1

  [PASS] tests/test_bot.py::test_polite  [12ms]
  [PASS] tests/test_bot.py::test_helpful  [14ms]
  [FAIL] tests/test_bot.py::test_no_pii  (score=0.67)  Contains email address  [11ms]
  [PASS] tests/test_bot.py::test_on_topic  [13ms]
  [PASS] tests/test_bot.py::test_concise  [15ms]

============================================================================
```

Green for pass, red for fail. Each line shows the test, the score on failure, the reason, and the wall time. No need to scroll through pytest output hunting for which semantic check broke.

---

## JSON Reports for CI

For CI integration, export results to JSON:

```bash
pytest --semantic-report-json=semantic-results.json
```

The output:

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
    },
    {
      "nodeid": "tests/test_bot.py::test_no_pii",
      "intent": "The text does not contain personal information",
      "passed": false,
      "score": 0.67,
      "reason": "Contains email address",
      "duration_ms": 11.1
    }
  ]
}
```

Feed this into your CI dashboard, your Slack alerts, your artifact storage — whatever your pipeline already does with JSON test results.

---

## Negation for Compliance Testing

Some of the most important LLM tests aren't about what the output *should* say. They're about what it *shouldn't* say.

```python
from semantix import Intent

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

class PIILeakage(Intent):
    """The text contains personal information like names, emails, or phone numbers."""

def test_no_medical_advice(assert_semantic):
    response = my_chatbot("my head hurts what should I take")
    assert_semantic(response, ~MedicalAdvice)

def test_no_pii_leakage(assert_semantic):
    response = my_chatbot("tell me about user 42")
    assert_semantic(response, ~PIILeakage)
```

The `~` operator negates the intent. The test passes only when the output does *not* match. This is how you test guardrails: toxicity, off-topic drift, unauthorized disclosures, regulatory compliance. Define the bad thing as an intent, negate it, assert against your output.

---

## Composing with Existing pytest

`pytest-semantix` is a normal pytest plugin. It doesn't replace anything in your test suite — it adds a fixture. Everything you already use works.

### Parametrize

```python
import pytest

@pytest.mark.parametrize("prompt,intent", [
    ("handle angry customer", "polite and professional"),
    ("explain a refund policy", "clear and informative"),
    ("say goodbye", "friendly"),
])
def test_chatbot_intents(assert_semantic, prompt, intent):
    response = my_chatbot(prompt)
    assert_semantic(response, intent)
```

### Combine with other fixtures

```python
@pytest.fixture
def chatbot():
    return MyChatbot(model="gpt-4o-mini", temperature=0.2)

def test_with_fixtures(chatbot, assert_semantic):
    response = chatbot.respond("hello")
    assert_semantic(response, "friendly greeting")
```

### Mix semantic and regular assertions

```python
def test_structured_response(assert_semantic):
    result = my_llm("generate a JSON summary")
    data = json.loads(result)  # regular assertion: valid JSON
    assert "summary" in data   # regular assertion: has the key
    assert_semantic(data["summary"], "concise and accurate")  # semantic: means the right thing
```

### Global threshold

If your team wants a stricter baseline across all tests:

```bash
pytest --semantic-threshold=0.85
```

Individual tests can still override:

```python
def test_strict(assert_semantic):
    assert_semantic(response, "accurate", threshold=0.95)
```

---

## What's Actually Happening

When you call `assert_semantic(output, intent)`, the plugin:

1. Resolves the intent (from the argument, the marker, or raises an error)
2. Passes the output and intent to a local NLI model via `semantix-ai`
3. The model returns a score and verdict
4. The plugin records the result (nodeid, intent, score, duration) for reporting
5. On failure, it raises `AssertionError` with score + reason

No network call. No subprocess. No container. The NLI model loads once per session and runs inference in-process. That's why it's ~15ms per assertion.

---

## Install

```bash
pip install pytest-semantix
```

Requires Python 3.10+ and pytest 7+. Pulls in `semantix-ai` automatically.

Then just use the `assert_semantic` fixture in your tests. No configuration, no `conftest.py` boilerplate, no setup step.

- **PyPI:** [pypi.org/project/pytest-semantix](https://pypi.org/project/pytest-semantix/)
- **GitHub:** [github.com/labrat-akhona/pytest-semantix](https://github.com/labrat-akhona/pytest-semantix)
- **semantix-ai (the engine):** [pypi.org/project/semantix-ai](https://pypi.org/project/semantix-ai/)
