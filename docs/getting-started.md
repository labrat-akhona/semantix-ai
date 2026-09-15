# Getting Started

Get a semantic test running in under 2 minutes.

## Install

```bash
pip install "semantix-ai[turbo]"
```

This installs the library and the quantized ONNX judge without PyTorch. On first use,
the judge downloads model and tokenizer files from Hugging Face Hub (about 79 MB for
the INT8 model). Inference runs locally, typically around 15–70 ms per check depending
on CPU and input length; startup and download time are additional.

Try it immediately:

```bash
semantix demo --no-color
semantix check "Thank you for your help." --intent "The text expresses gratitude."
semantix prove --n 20 --no-color
```

For the PyTorch backend, install `"semantix-ai[nli]"`. A bare `pip install semantix-ai`
installs only the core interfaces and needs a user-supplied Judge or an inference extra.

### Offline use

Run a check once while connected to populate the model cache. For subsequent offline
runs, set `HF_HUB_OFFLINE=1` before starting Python (PowerShell:
`$env:HF_HUB_OFFLINE="1"`). This prevents Hub version checks as well as downloads;
missing cached files cause an error. See the [Hugging Face environment variable
reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline).

## Your first semantic assertion

```python
from semantix.testing import assert_semantic

def test_response_expresses_gratitude():
    response = "Thank you for your help."
    assert_semantic(response, "The text expresses gratitude.")
```

`assert_semantic` checks whether the response entails the stated claim. Replace the
canned response with your application's output. Test representative positives and
negatives: abstract qualities such as politeness and compound requirements are
harder for a small NLI model than concrete single claims.

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

You can also use `@validate_intent(Safe)` on a function returning `str`. It returns
an Intent instance after validation. Negation flips the threshold verdict; absence
of entailment can reflect uncertainty and should not be treated as proof of safety.

## What's next

- [Intents](concepts/intents.md) -- deep dive into defining semantic contracts
- [Composition](concepts/composition.md) -- combine intents with `&`, `|`, `~`
- [Testing](testing/assert-semantic.md) -- `assert_semantic()` reference
- [Judges](judges.md) -- choose the right speed/accuracy tradeoff
- [Integrations](integrations/guardrails.md) -- drop into Guardrails, Instructor, Pydantic AI, LangChain, DSPy
