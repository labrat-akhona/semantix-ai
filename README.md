<p align="center">
  <h1 align="center">semantix-ai</h1>
  <p align="center"><strong>Validate what your LLM outputs mean, not just their shape.</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/v/semantix-ai?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/pyversions/semantix-ai" alt="Python versions"></a>
  <a href="https://github.com/labrat-akhona/semantix-ai/blob/master/LICENSE"><img src="https://img.shields.io/github/license/labrat-akhona/semantix-ai" alt="License"></a>
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/dm/semantix-ai?color=green" alt="Downloads"></a>
</p>

---

```bash
pip install semantix-ai
```

```python
from semantix.testing import assert_semantic

def test_chatbot_is_polite():
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

Runs locally. ~15ms. No API key. Works in pytest, unittest, or any test runner.

On failure:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

---

## What It Does

semantix validates that LLM outputs **mean the right thing** — using a local NLI model, not string matching or another LLM call.

```python
from semantix import Intent, validate_intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude."""

@validate_intent
def decline_invite(event: str) -> ProfessionalDecline:
    return call_my_llm(event)

result = decline_invite("the company retreat")
# Returns a validated ProfessionalDecline — or raises SemanticIntentError
```

**Key properties:**
- **Local inference** — NLI model runs on CPU, no data leaves your machine
- **~15ms per check** — negligible overhead on any LLM call
- **Zero API cost** — no tokens burned for validation
- **212 tests** — well-tested, MIT licensed

---

## Compliance with Negation

Block what your model must NOT say — PII, medical advice, competitor mentions:

```python
from semantix import Intent, Not

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

Safe = ~MedicalAdvice  # or Not(MedicalAdvice)

@validate_intent
def chatbot(msg: str) -> Safe:
    return call_my_llm(msg)
```

Compose with `&` (all must pass) and `|` (any must pass):

```python
SafeAndPolite = Polite & ~MedicalAdvice & ~LegalAdvice
```

---

## Self-Healing Retries

On failure, semantix injects structured feedback so the LLM knows what went wrong:

```python
from typing import Optional

@validate_intent(retries=2)
def decline(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
    prompt = f"Decline this invite: {event}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return call_llm(prompt)
```

First call: `semantix_feedback` is `None`. On retry: it receives a Markdown report with the score, reason, and rejected output. Reliability improves from 21% to 70% across 3 intent categories.

---

## Framework Integrations

Drop into your existing stack — retries are handled natively by each framework.

### Guardrails AI

```python
from guardrails import Guard
from semantix.integrations.guardrails import SemanticIntent

guard = Guard().use(SemanticIntent("must be polite and professional"))
result = guard.validate("Thank you for your patience.")
```

### Instructor

```python
from semantix.integrations.instructor import SemanticStr
from pydantic import BaseModel

class Response(BaseModel):
    reply: SemanticStr["must be polite and professional", 0.85]
```

### Pydantic AI

```python
from pydantic_ai import Agent
from semantix.integrations.pydantic_ai import semantix_validator

agent = Agent("openai:gpt-4o", output_type=str)
agent.output_validator(semantix_validator(Polite))
```

### LangChain

```python
from semantix.integrations.langchain import SemanticValidator

validator = SemanticValidator(Polite)
chain = prompt | llm | StrOutputParser() | validator
```

Install extras: `pip install "semantix-ai[instructor]"`, `"semantix-ai[pydantic-ai]"`, `"semantix-ai[langchain]"`, `"semantix-ai[guardrails]"`

---

## Self-Training Flywheel

Every retry produces labeled training data — rejected output, reason, corrected output:

```python
from semantix.training import TrainingCollector
from semantix.training.exporters import export_openai

collector = TrainingCollector("training_data.jsonl")

@validate_intent(retries=2, collector=collector)
def decline(event: str) -> ProfessionalDecline:
    return call_my_llm(event)

# Export to OpenAI fine-tuning format
export_openai("training_data.jsonl", "finetune.jsonl")
```

Your guardrail becomes your training pipeline:

```
Validate → Fail → Correct → Capture → Fine-tune → Validate (fewer failures)
```

---

## Pluggable Judges

Choose the right speed/accuracy tradeoff:

```python
from semantix import NLIJudge, EmbeddingJudge, LLMJudge, CachingJudge

# Default — local NLI entailment (no API key, ~15ms)
@validate_intent(judge=NLIJudge())

# Fast — local cosine similarity (~5ms)
@validate_intent(judge=EmbeddingJudge())

# Accurate — GPT-4o-mini with 0-1 scoring + reason
@validate_intent(judge=LLMJudge(model="gpt-4o-mini"))

# Cached — wraps any judge with LRU cache
@validate_intent(judge=CachingJudge(NLIJudge(), maxsize=256))
```

Quantized mode for minimal footprint (~25MB vs ~500MB for PyTorch):

```bash
pip install "semantix-ai[turbo]"
# Automatically uses QuantizedNLIJudge (INT8 ONNX, no PyTorch)
```

---

## Advanced Features

**Forensic analysis** — token-level attribution on failure:
```python
from semantix import ForensicJudge, QuantizedNLIJudge
judge = ForensicJudge(QuantizedNLIJudge())
# Verdict.reason: "Suspect Tokens: [indemnify, forfeit, waive]"
```

**Streaming** — validate once the full stream is assembled:
```python
from semantix import StreamCollector
for chunk in StreamCollector(Polite, judge=my_judge).wrap(llm_stream()):
    print(chunk, end="")
```

**Audit trail** — hash-chained JSON-LD certificates:
```python
from semantix.audit.engine import AuditEngine
engine = AuditEngine()
engine.verify_chain()  # True if no tampering
```

**MCP server** — any AI agent can validate intents as a tool:
```bash
pip install "semantix-ai[mcp,nli]"
mcp run semantix/mcp/server.py
```

**Async** — works transparently with `async def`.

---

## Installation

```bash
pip install semantix-ai                    # Core (default NLI judge)
pip install "semantix-ai[turbo]"           # Quantized ONNX (smallest footprint)
pip install "semantix-ai[openai]"          # LLM judge (GPT-4o-mini)
pip install "semantix-ai[instructor]"      # Instructor integration
pip install "semantix-ai[pydantic-ai]"     # Pydantic AI integration
pip install "semantix-ai[langchain]"       # LangChain integration
pip install "semantix-ai[guardrails]"      # Guardrails AI integration
pip install "semantix-ai[all]"             # Everything
```

> The package name on PyPI is `semantix-ai`. The import is `from semantix import ...`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, testing, and submission guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built by <a href="https://github.com/labrat-akhona">Akhona Eland</a> in South Africa</em>
</p>
