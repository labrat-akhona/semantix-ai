# Build LLM Guardrails in 3 Lines of Python (No API Key, No Cloud)

Your LLM just told a customer their rash "looks like it could be melanoma." Your chatbot leaked a user's email address in a support response. Your RAG pipeline went off-topic and started explaining how to pick locks.

These aren't hypotheticals. They're Tuesday.

You need guardrails. Here's what that currently looks like:

1. **Regex.** You write `r"(?i)(you should take|I recommend taking)"` to catch medical advice. The model rephrases to "it might help to consider" and your filter is useless. You add more patterns. The model finds more phrasings. You are now maintaining a regex zoo that catches false positives and misses actual violations.

2. **LLM-as-judge.** Call GPT-4 to review every output. That's 500ms–2s per check, $0.01–0.03 per call, and a hard dependency on an external API. Your guardrail is now slower than the thing it's guarding. Also, you need an API key in production, your costs scale with traffic, and when OpenAI has a bad day your guardrails go down.

3. **Cloud guardrail services.** AWS Bedrock Guardrails, Azure Content Safety, etc. Vendor lock-in, network latency, usage-based pricing, and your data leaves your infrastructure. Good luck explaining that to your compliance team.

None of these are good. What you actually want is: check whether the output *means* something bad, locally, in milliseconds, for free.

---

## 3 lines

```bash
pip install semantix-ai
```

```python
from semantix import Intent, validate_intent

class NoPII(Intent):
    """The text does not contain personal information such as names, emails, phone numbers, or addresses."""

class NoMedicalAdvice(Intent):
    """The text does not provide medical diagnoses or treatment recommendations."""

@validate_intent
def my_chatbot(message: str) -> ~NoPII & ~NoMedicalAdvice:
    return call_my_llm(message)
```

That's it. Every call to `my_chatbot` now runs through a local NLI model that checks whether the output violates your policies. ~15ms on CPU. No API key. No network call. No tokens burned.

If the output leaks PII or gives medical advice, it raises `SemanticIntentError` with the score, the violated intent, and a reason. The bad output never reaches your user.

---

## How the negation pattern works

The `~` operator is the key. An `Intent` describes what something *is*. `~Intent` checks that the output is *not* that thing.

```python
from semantix import Intent

class ToxicLanguage(Intent):
    """The text contains insults, profanity, threats, or aggressive language."""

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

class PIILeakage(Intent):
    """The text contains personal information like names, emails, phone numbers, or addresses."""

class LegalAdvice(Intent):
    """The text provides specific legal counsel or interprets laws for the user's situation."""
```

Each of these describes a *bad thing*. Negate them and you have guardrails:

```python
Safe = ~ToxicLanguage
Compliant = ~MedicalAdvice
Private = ~PIILeakage
NotALawyer = ~LegalAdvice
```

Under the hood, `~MedicalAdvice` creates a `Not[MedicalAdvice]` intent. The NLI model checks whether the output entails the original description. If it does, the negated check fails. If it doesn't, the output is clean.

This works because NLI models understand *meaning*, not patterns. "You should take ibuprofen" and "Consider an anti-inflammatory" both entail medical advice. A regex catches neither unless you enumerated both phrasings. The NLI model catches both because they mean the same thing.

---

## Composing policies

Real compliance isn't one rule. It's a policy — multiple constraints that all need to hold, or where at least one must hold. semantix gives you `&` and `|` for this.

### All constraints must pass

```python
@validate_intent
def customer_support(msg: str) -> ~ToxicLanguage & ~PIILeakage & ~MedicalAdvice:
    return call_my_llm(msg)
```

The `&` operator creates an `AllOf` composite. Every negated intent is checked. If any one fails, the output is rejected. This is your production safety policy in one line of Python type annotation.

### At least one constraint must pass

```python
class Apology(Intent):
    """The text contains a sincere apology for the inconvenience."""

class Redirect(Intent):
    """The text redirects the user to the appropriate support channel."""

@validate_intent
def handle_complaint(msg: str) -> Apology | Redirect:
    return call_my_llm(msg)
```

The `|` operator creates an `AnyOf` composite. The output passes if it satisfies at least one intent.

### Mix positive and negative

```python
class Helpful(Intent):
    """The text provides a clear, actionable answer to the user's question."""

@validate_intent
def safe_assistant(msg: str) -> Helpful & ~ToxicLanguage & ~PIILeakage:
    return call_my_llm(msg)
```

The output must be helpful AND must not be toxic AND must not leak PII. Positive and negative constraints compose freely.

---

## Self-healing retries

Guardrails that just block are a blunt instrument. Sometimes you want the LLM to try again with feedback about what went wrong. Add `retries` and a `semantix_feedback` parameter:

```python
from typing import Optional

@validate_intent(retries=2)
def safe_chatbot(
    message: str,
    semantix_feedback: Optional[str] = None,
) -> Helpful & ~ToxicLanguage & ~PIILeakage:
    prompt = f"Answer this customer question: {message}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return call_my_llm(prompt)
```

On the first call, `semantix_feedback` is `None`. If the output fails validation, the decorator automatically injects a structured Markdown feedback block explaining what went wrong — the violated intent, the score, the rejected output. The LLM gets a second chance to fix it.

This turns a guardrail from a wall into a feedback loop. The model learns from its mistake in-context and self-corrects. In practice, most violations are fixed on the first retry.

The feedback looks like this:

```markdown
## Semantix Self-Healing Feedback

Attempt **1** failed validation.

### What went wrong
- **Intent:** `Not[PIILeakage]`
- **Score:** 0.9142 (threshold not met)
- **Judge reason:** Text contains what appears to be an email address

### What is required
The text must NOT satisfy the following:

The text contains personal information like names, emails, phone numbers, or addresses.

### Your previous output (rejected)
Sure, I can help! John's email is john.doe@example.com...

Please generate a new response that satisfies the requirement above.
```

---

## Testing guardrails in CI

Guardrails in production are half the story. You also need to test that they work before you deploy. Two tools:

### pytest-semantix

```bash
pip install pytest-semantix
```

```python
from semantix import Intent

class PIILeakage(Intent):
    """The text contains personal information like names, emails, phone numbers, or addresses."""

def test_no_pii_in_response(assert_semantic):
    response = my_chatbot("tell me about user 42")
    assert_semantic(response, ~PIILeakage)

def test_no_medical_advice(assert_semantic):
    response = my_chatbot("my head hurts")
    assert_semantic(response, ~MedicalAdvice)
```

Each test runs in ~15ms locally. No API key in CI secrets. No flaky network calls. Your guardrail tests run as fast as your unit tests.

### GitHub Action

Add semantic checks to your CI pipeline with the [semantic-test-action](https://github.com/labrat-akhona/semantic-test-action):

```yaml
- uses: labrat-akhona/semantic-test-action@v1
  with:
    test-path: tests/
    threshold: 0.8
    report-format: json
```

This runs your `pytest-semantix` tests in CI and produces a report. Failed guardrail tests block the PR. Your compliance policy is enforced before code reaches main.

---

## What's actually happening under the hood

When you write `~MedicalAdvice` and the decorator validates an output, here's the sequence:

1. The decorator calls your function and captures the raw string output.
2. It extracts the intent description from the class docstring.
3. For `Not[X]`, it checks whether the output entails `X`. If entailment score is *above* threshold, the negated check *fails* — the output matches the bad thing.
4. For `AllOf`, it checks every component. All must pass.
5. For `AnyOf`, it checks components until one passes.
6. The NLI model runs locally via ONNX Runtime (quantized INT8). No GPU required. ~15ms per check on CPU.
7. If validation fails and retries remain, feedback is injected and the function is called again.
8. If all retries are exhausted, `SemanticIntentError` is raised with full diagnostics.

The model is downloaded once (~100MB) and cached locally. After that, everything is offline. Your guardrails work on an airplane.

---

## When to use this vs. other approaches

**Use semantix guardrails when:**
- You need low-latency checks (< 20ms) in the hot path
- You can't send data to external APIs (compliance, air-gapped, privacy)
- You want deterministic, reproducible guardrail behavior
- You need guardrails in CI/CD, not just production
- You want zero marginal cost per check

**Use an LLM-as-judge when:**
- You need nuanced, context-heavy evaluation that NLI can't capture
- Latency and cost don't matter
- You're doing one-off evaluations, not real-time guardrailing

**Use regex/keyword filters when:**
- You have a known, fixed list of exact strings to block (e.g., specific slurs, specific SSN formats)
- You don't need semantic understanding, just pattern matching

In practice, these stack. Use semantix for the fast semantic layer, regex for known-exact patterns, and LLM-as-judge for the hard cases that need deep reasoning. semantix handles the 90% that regex can't and LLM-as-judge is too slow for.

---

## Install

```bash
pip install semantix-ai
```

Python 3.10+. No API key. No GPU. Works on Linux, macOS, Windows.

- **PyPI:** [pypi.org/project/semantix-ai](https://pypi.org/project/semantix-ai/)
- **GitHub:** [github.com/labrat-akhona/semantix-ai](https://github.com/labrat-akhona/semantix-ai)
- **Docs:** [labrat-akhona.github.io/semantix-ai](https://labrat-akhona.github.io/semantix-ai/)
- **pytest-semantix:** [pypi.org/project/pytest-semantix](https://pypi.org/project/pytest-semantix/)

---

*Tags: #python #ai #security #llm*
*Series: semantix-ai*
