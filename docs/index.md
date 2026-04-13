# semantix-ai

**Validate what your LLM outputs mean, not just their shape.**

---

semantix is a semantic type system for AI outputs. It uses local NLI (Natural Language Inference) to check whether text *means* what you require -- no API keys, no network calls, ~15ms per check.

```bash
pip install semantix-ai
```

```python
from semantix.testing import assert_semantic

def test_chatbot_is_polite():
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

On failure you get a clear diagnostic:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

## Key properties

- **Local inference** -- NLI model runs on CPU, no data leaves your machine
- **~15ms per check** -- negligible overhead on any LLM call
- **Zero API cost** -- no tokens burned for validation
- **Pluggable judges** -- NLI, embedding, LLM, quantized ONNX, forensic, caching
- **Framework integrations** -- Guardrails AI, Instructor, Pydantic AI, LangChain, DSPy
- **Self-healing retries** -- structured feedback so the LLM corrects itself
- **Training flywheel** -- every retry produces labeled fine-tuning data

## How it works

1. You define an **Intent** -- a class whose docstring describes what the text should mean.
2. A **Judge** evaluates whether the output entails the intent using NLI, embeddings, or an LLM.
3. If validation fails, semantix can **retry** with structured feedback, and **collect** the correction pair for fine-tuning.

```python
from semantix import Intent, validate_intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude."""

@validate_intent
def decline_invite(event: str) -> ProfessionalDecline:
    return call_my_llm(event)

result = decline_invite("the company retreat")
# Returns a validated ProfessionalDecline -- or raises SemanticIntentError
```

## Quick links

| Topic | Description |
|---|---|
| [Getting Started](getting-started.md) | Install and run your first semantic test in under 2 minutes |
| [Intents](concepts/intents.md) | Define semantic contracts with Intent classes |
| [Composition](concepts/composition.md) | Combine intents with `Not`, `AllOf`, `AnyOf`, `~`, `&`, `|` |
| [assert_semantic](testing/assert-semantic.md) | One-line semantic assertions for any test runner |
| [pytest-semantix](testing/pytest-plugin.md) | Pytest plugin with markers, reports, and CI integration |
| [Judges](judges.md) | NLI, Embedding, LLM, Quantized, Forensic, Caching |
| [Integrations](integrations/guardrails.md) | Guardrails AI, Instructor, Pydantic AI, LangChain, DSPy |
| [Training](training.md) | Collect correction pairs, calibrate thresholds, export for fine-tuning |
| [Advanced](advanced.md) | Streaming, audit trails, MCP server, async |

## Installation options

```bash
pip install semantix-ai                    # Core (default NLI judge)
pip install "semantix-ai[turbo]"           # Quantized ONNX (smallest footprint)
pip install "semantix-ai[openai]"          # LLM judge (GPT-4o-mini)
pip install "semantix-ai[instructor]"      # Instructor integration
pip install "semantix-ai[pydantic-ai]"     # Pydantic AI integration
pip install "semantix-ai[langchain]"       # LangChain integration
pip install "semantix-ai[guardrails]"      # Guardrails AI integration
pip install "semantix-ai[dspy]"            # DSPy integration
pip install "semantix-ai[mcp,nli]"         # MCP server
pip install "semantix-ai[all]"             # Everything
```

!!! note
    The package name on PyPI is `semantix-ai`. The import is `from semantix import ...`.
