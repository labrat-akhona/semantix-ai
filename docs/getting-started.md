# Getting Started

Get a semantic test running in under 2 minutes.

## Install

```bash
pip install semantix-ai
```

This installs the core library with the default NLI judge. The first time you run a check, it downloads a small (~85 MB) cross-encoder model from HuggingFace Hub.

For a smaller footprint (~25 MB, no PyTorch dependency):

```bash
pip install "semantix-ai[turbo]"
```

## Your first semantic assertion

```python
from semantix.testing import assert_semantic

def test_chatbot_is_polite():
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

That's it. `assert_semantic` runs a local NLI model to check whether the response *entails* "polite and professional". No API key, no network calls, ~15ms.

On failure:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

## Your first Intent class

For reusable semantic contracts, define an Intent:

```python
from semantix import Intent, validate_intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude."""

@validate_intent
def decline_invite(event: str) -> ProfessionalDecline:
    return call_my_llm(event)

result = decline_invite("the company retreat")
# Returns a ProfessionalDecline instance wrapping the validated text
print(result.text)  # the actual string
```

The `@validate_intent` decorator:

1. Calls your function and captures the raw string output.
2. Evaluates it against the Intent's docstring using a local NLI judge.
3. Returns a `ProfessionalDecline` instance on success, or raises `SemanticIntentError` on failure.

## Add retries for self-healing

```python
from typing import Optional

@validate_intent(retries=2)
def decline(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
    prompt = f"Decline this invite: {event}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return call_llm(prompt)
```

On the first call, `semantix_feedback` is `None`. If validation fails, the next call receives a Markdown report explaining what went wrong, the score, and the rejected output. The LLM uses this to self-correct.

## Block unwanted content

```python
from semantix import Intent, Not

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

Safe = ~MedicalAdvice  # or Not(MedicalAdvice)

@validate_intent
def chatbot(msg: str) -> Safe:
    return call_my_llm(msg)
```

## What's next

- [Intents](concepts/intents.md) -- deep dive into defining semantic contracts
- [Composition](concepts/composition.md) -- combine intents with `&`, `|`, `~`
- [Testing](testing/assert-semantic.md) -- `assert_semantic()` reference
- [Judges](judges.md) -- choose the right speed/accuracy tradeoff
- [Integrations](integrations/guardrails.md) -- drop into Guardrails, Instructor, Pydantic AI, LangChain, DSPy
