# How to Test LLM Outputs in pytest (Without Calling an LLM)

Every team building with LLMs hits the same wall: how do you actually test that your outputs mean the right thing? String matching breaks the moment your model rephrases anything. Regex is worse. And calling GPT-4 as a judge in CI is slow, flaky, and expensive.

What if you could just write this:

```bash
pip install semantix-ai
```

```python
from semantix.testing import assert_semantic

def test_chatbot_is_polite():
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

That's a real test. It runs locally on CPU in ~15ms, needs no API key, and tells you *why* it failed.

---

## How It Works

`assert_semantic` uses a local NLI (Natural Language Inference) model under the hood. No network calls, no tokens burned, no OpenAI key in your CI secrets. You describe the intent in plain English, and the model checks whether the output entails that intent.

On failure, you get a score, the intent, and a reason — not just a raw `AssertionError`:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

---

## Reusable Contracts with Intent Classes

For intents you use across many tests, define them as classes. The docstring *is* the requirement:

```python
from semantix import Intent
from semantix.testing import assert_semantic

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude."""

def test_decline_is_professional():
    response = decline_invite("the office party")
    assert_semantic(response, ProfessionalDecline)
```

---

## Compliance Testing with Negation

Need to verify your chatbot *doesn't* do something? Use the `~` operator to negate an intent:

```python
from semantix import Intent
from semantix.testing import assert_semantic

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

def test_chatbot_avoids_medical_advice():
    response = chatbot("my head hurts what should I take")
    assert_semantic(response, ~MedicalAdvice)
```

This passes only when the output does *not* match the intent. Useful for PII leakage, toxicity, off-topic drift — anything your model should stay away from.

---

## What You Get

- **~15ms per assertion** on CPU — fast enough for hundreds of tests in CI
- **No API keys or costs** — the NLI model runs locally
- **Score + reason on failure** — not just pass/fail
- **Works with any test runner** — pytest, unittest, or plain `assert`
- **Composable intents** — combine with `&` (all of), `|` (any of), `~` (not)

---

## Get Started

```bash
pip install semantix-ai
```

Write your first semantic test in under a minute. Works out of the box with pytest, no configuration needed.

We also have integrations for [Guardrails AI](https://github.com/guardrails-ai/guardrails), [LangChain](https://github.com/langchain-ai/langchain), and [Instructor](https://github.com/567-labs/instructor) if you're using those in production.

- **PyPI:** [pypi.org/project/semantix-ai](https://pypi.org/project/semantix-ai/)
- **GitHub:** [github.com/labrat-akhona/semantix-ai](https://github.com/labrat-akhona/semantix-ai)
