<p align="center">
  <h1 align="center">Semantix</h1>
  <p align="center"><strong>A Semantic Type System for AI Outputs</strong></p>
  <p align="center">
    Define <em>what your LLM output should mean</em>, not just what shape it has.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/v/semantix-ai?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/pyversions/semantix-ai" alt="Python versions"></a>
  <a href="https://github.com/labrat-akhona/semantix-ai/blob/master/LICENSE"><img src="https://img.shields.io/github/license/labrat-akhona/semantix-ai" alt="License"></a>
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/dm/semantix-ai?color=green" alt="Downloads"></a>
</p>

---

## The Problem

You validate your LLM outputs with Pydantic — great, the JSON is well-formed. But the model just returned a "polite decline" that says *"I'd rather gouge my eyes out."* It passes your type checks. It fails the vibe check.

**Pydantic validates shape. Semantix validates meaning.**

```python
from semantix import Intent, validate_intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude or aggressive."""

@validate_intent
def decline_invite(event: str) -> ProfessionalDecline:
    return call_my_llm(event)   # returns a plain string

result = decline_invite("the company retreat")
# ✓ result is a ProfessionalDecline instance — validated by a judge
# ✗ raises SemanticIntentError if the output is rude, off-topic, etc.
```

---

## Installation

```bash
# Core (bring your own judge)
pip install semantix-ai

# With OpenAI judge (GPT-4o-mini — accurate, needs API key)
pip install "semantix-ai[openai]"

# With embedding judge (sentence-transformers — fast, runs locally)
pip install "semantix-ai[embeddings]"

# With NLI judge (cross-encoder entailment — accurate, runs locally)
pip install "semantix-ai[nli]"

# Framework integrations
pip install "semantix-ai[instructor]"   # Instructor
pip install "semantix-ai[pydantic-ai]"  # Pydantic AI
pip install "semantix-ai[langchain]"    # LangChain

# Everything
pip install "semantix-ai[all]"
```

> **Note:** The package name on PyPI is `semantix-ai`. The import is `from semantix import ...`.

---

## Quick Start

### 1. Define an Intent

An Intent is a class whose docstring describes a semantic contract:

```python
from semantix import Intent

class PositiveSentiment(Intent):
    """The text must express a clearly positive, optimistic, or encouraging sentiment."""
    threshold = 0.85  # optional — default is 0.8
```

### 2. Decorate your LLM call

```python
from semantix import validate_intent

@validate_intent
def encourage(name: str) -> PositiveSentiment:
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Encourage {name}"}],
    ).choices[0].message.content
```

### 3. Handle failures

```python
from semantix import SemanticIntentError

try:
    result = encourage("Alice")
    print(result.text)  # the validated string
except SemanticIntentError as e:
    print(f"Failed: {e.intent_name} (score={e.score})")
```

That's it. Three steps. Your LLM output is now semantically typed.

---

## Why Not Just Use Guardrails / NeMo / Instructor?

| | **Semantix** | Guardrails AI | NeMo Guardrails | Instructor |
|---|---|---|---|---|
| **Validates meaning** | ✅ Intent docstrings | ❌ Schema-focused | ✅ Dialogue rails | ❌ Schema-focused |
| **Zero required deps** | ✅ Core is dependency-free | ❌ Heavy dependency tree | ❌ Heavy dependency tree | ❌ Requires Pydantic |
| **Works with any LLM** | ✅ Decorator on any function | ⚠️ LLM-specific wrappers | ⚠️ Config-driven | ⚠️ Patched clients |
| **Pluggable judges** | ✅ LLM / Embedding / NLI / Custom | ❌ Fixed validators | ❌ Fixed approach | ❌ Fixed approach |
| **Lines of code to validate** | ~5 | ~20+ | ~30+ (YAML config) | ~10 |
| **Composable** | ✅ `A & B`, `A \| B` | ❌ | ❌ | ❌ |

Semantix is not a replacement for structural validation — use Pydantic for that. Semantix is the **next layer**: after you know the shape is right, verify the *meaning* is right too.

---

## Universal Agent Support (MCP)

Semantix ships with a built-in [MCP](https://modelcontextprotocol.io/) server so any AI agent can run semantic intent checks as a tool — no code changes required.

```bash
pip install "semantix-ai[mcp,nli]"
mcp run semantix/mcp/server.py
```

The `verify_text_intent` tool accepts any text and intent description, returns a confidence score, and provides structured correction suggestions when validation fails — enabling **cross-agent self-healing**.

### Add to Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "semantix-verify": {
      "command": "mcp",
      "args": ["run", "semantix/mcp/server.py"],
      "cwd": "/path/to/your/semantix-ai"
    }
  }
}
```

Claude can then call `verify_text_intent` to validate any text against a semantic requirement before responding.

---

## Framework Integrations (NEW in v0.1.6)

Drop semantix into your existing LLM framework — retries are handled natively by each framework.

### Instructor

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel
from semantix.integrations.instructor import semantic_validator, SemanticStr
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

# Option 1: Explicit Intent class
class Response(BaseModel):
    reply: Annotated[str, AfterValidator(semantic_validator(Polite))]

# Option 2: Inline shorthand
class QuickResponse(BaseModel):
    reply: SemanticStr["must be polite and professional", 0.85]
```

Validation failures raise `ValueError` — Instructor catches this and retries automatically with `max_retries`.

### Pydantic AI

```python
from pydantic_ai import Agent
from semantix.integrations.pydantic_ai import semantix_validator
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

agent = Agent("openai:gpt-4o", output_type=str)
agent.output_validator(semantix_validator(Polite))

result = agent.run_sync("Decline the meeting invitation")
```

Validation failures raise `ModelRetry` — Pydantic AI retries automatically.

### LangChain

```python
from langchain_core.output_parsers import StrOutputParser
from semantix.integrations.langchain import SemanticValidator
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

validator = SemanticValidator(Polite)
chain = prompt | llm | StrOutputParser() | validator
result = chain.invoke({"event": "the company retreat"})
```

Validation failures raise `OutputParserException` — compatible with LangChain's `RetryWithErrorOutputParser`.

---

## Zero-Latency Infrastructure (NEW in v0.1.5)

### Quantized Inference

Semantix ships a quantized NLI judge that runs INT8 ONNX inference — no PyTorch, no GPU, ~50% faster:

```bash
pip install "semantix-ai[turbo]"
```

```python
from semantix import validate_intent

# Automatically uses QuantizedNLIJudge when onnxruntime is installed
@validate_intent
def review(text: str) -> LegalCompliance:
    return call_llm(text)
```

Total dependency footprint: ~25MB (onnxruntime + tokenizers) vs ~500MB+ for PyTorch.

### Forensic Analysis on Failure

When validation fails, the `ForensicJudge` identifies exactly which tokens caused the contradiction:

```python
from semantix import ForensicJudge, QuantizedNLIJudge

judge = ForensicJudge(QuantizedNLIJudge())

@validate_intent(judge=judge)
def review(text: str) -> LegalCompliance:
    return call_llm(text)

# On failure, Verdict.reason contains:
# ## Breach Report
# **Score:** 0.0823
# ### Token Attribution
# **indemnify** (0.72), **forfeit** (0.58), **waive** (0.41)
# ### Summary
# Intent failed. High contradiction detected. Suspect Tokens: [indemnify, forfeit, waive]
```

### Immutable Audit Trail

Every validation is logged as a hash-chained JSON-LD certificate:

```python
from semantix.audit.engine import AuditEngine

engine = AuditEngine()  # singleton
engine.verify_chain()   # True if no tampering
engine.flush(Path("audit.jsonl"))
```

---

## Features

### Swappable Judges

Choose the right speed/accuracy tradeoff for your use case:

```python
from semantix import EmbeddingJudge, LLMJudge, NLIJudge, CachingJudge

# Default — NLI entailment with softmax calibration (no API key, runs locally)
# Uses correct entailment index + softmax for true 0–1 probability scores
@validate_intent(judge=NLIJudge())
def default_fn(x: str) -> MyIntent: ...

# Fast — local cosine similarity (no API key needed)
@validate_intent(judge=EmbeddingJudge())
def fast_fn(x: str) -> MyIntent: ...

# Accurate — asks GPT-4o-mini for a 0–1 confidence score + reason
@validate_intent(judge=LLMJudge(model="gpt-4o-mini"))
def accurate_fn(x: str) -> MyIntent: ...

# Cached — wraps any judge with LRU cache
@validate_intent(judge=CachingJudge(NLIJudge(), maxsize=256))
def cached_fn(x: str) -> MyIntent: ...
```

### Informed Self-Healing Retries

On failure, the decorator injects structured feedback so the LLM knows *exactly* what went wrong — zero boilerplate:

```python
from typing import Optional
from semantix import validate_intent

@validate_intent(retries=2)
def decline(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
    prompt = f"Decline this invite: {event}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return call_llm(prompt)
```

On the first call `semantix_feedback` is `None`. If validation fails, the next retry receives a Markdown report with the score, reason, requirement, and rejected output — so the LLM can self-correct.

**Benchmark result:** Self-healing improves reliability from 21.1% to 70.0% (+48.9%) across 3 intent categories.

The manual `get_last_failure()` API is also still available for custom feedback formatting.

### Composite Intents

Combine multiple intents with `&` (all must pass) or `|` (any must pass):

```python
from semantix import AllOf, AnyOf

# Operator syntax
PoliteAndPositive = ProfessionalDecline & PositiveSentiment

# Function syntax
FlexibleDecline = AnyOf(ProfessionalDecline, CasualDecline)

@validate_intent(judge=EmbeddingJudge())
def respond(msg: str) -> PoliteAndPositive: ...
```

### Async Support

Works transparently with `async def`:

```python
@validate_intent(judge=EmbeddingJudge())
async def encourage(name: str) -> PositiveSentiment:
    response = await async_openai_call(name)
    return response
```

### Streaming

Validate once the full stream is assembled:

```python
from semantix import StreamCollector

# Iterator wrapper — yields chunks through, validates at the end
sc = StreamCollector(ProfessionalDecline, judge=my_judge)
for chunk in sc.wrap(llm_stream()):
    print(chunk, end="")
result = sc.result()  # validated Intent or raises

# Async context manager
async with StreamCollector(ProfessionalDecline, judge=my_judge) as sc:
    async for chunk in llm_stream:
        sc.feed(chunk)
result = sc.result()
```

### Observability

All validation events are emitted via Python's `logging` module:

```python
import logging
logging.getLogger("semantix").setLevel(logging.DEBUG)
```

```
INFO  semantix.validation | intent=ProfessionalDecline passed=True score=0.92 latency_ms=45.23 attempt=1
```

### Custom Judges

Implement the `Judge` interface to plug in any backend:

```python
from semantix import Judge, Verdict

class MyCustomJudge(Judge):
    def evaluate(self, output: str, intent_description: str, threshold: float = 0.8) -> Verdict:
        score = my_scoring_function(output, intent_description)
        return Verdict(passed=score >= threshold, score=score, reason="Custom logic")
```

---

## API Reference

| Symbol | Description |
|---|---|
| `Intent` | Base class — subclass with a docstring to define a semantic type |
| `SemanticIntentError` | Raised when validation fails (`.output`, `.score`, `.intent_name`) |
| `@validate_intent` | Decorator — validates return values against their Intent type hint |
| `get_last_failure()` | Returns the last `SemanticIntentError` in current context (for smart retries) |
| `Judge` | Abstract base — implement `.evaluate()` for custom backends |
| `Verdict` | Dataclass — `.passed`, `.score`, `.reason` |
| `LLMJudge` | OpenAI-based judge (accurate, needs API key) |
| `EmbeddingJudge` | Sentence-transformers cosine similarity judge (fast, local) |
| `NLIJudge` | Cross-encoder NLI entailment judge (softmax-calibrated, local, **default**) |
| `CachingJudge` | LRU cache wrapper for any judge |
| `AllOf(*intents)` | Composite — all intents must be satisfied |
| `AnyOf(*intents)` | Composite — at least one intent must be satisfied |
| `QuantizedNLIJudge` | INT8 ONNX NLI judge — fast, no PyTorch (needs `onnxruntime`) |
| `ForensicJudge` | Wrapper — token-level attribution Breach Report on failure |
| `AuditEngine` | Hash-chained JSON-LD audit trail singleton |
| `StreamCollector` | Validates streamed LLM output once fully assembled |
| `semantic_validator` | Instructor adapter — Pydantic `AfterValidator` for Intent checking |
| `SemanticStr` | Instructor shorthand — `SemanticStr["must be polite", 0.85]` |
| `semantix_validator` | Pydantic AI adapter — `@agent.output_validator` compatible |
| `SemanticValidator` | LangChain adapter — Runnable with `invoke`/`batch`/pipe support |

---

## Project Structure

```
semantix/
├── __init__.py          # Public API
├── intent.py            # Intent base class + metaclass
├── exceptions.py        # SemanticIntentError
├── decorator.py         # @validate_intent (retries, self-healing)
├── composite.py         # AllOf / AnyOf combinators
├── observability.py     # Structured logging
├── streaming.py         # StreamCollector
├── audit/
│   ├── __init__.py      # Package marker
│   └── engine.py        # AuditEngine (JSON-LD + SHA-256 chain)
├── judges/
│   ├── __init__.py      # Judge ABC + Verdict
│   ├── embedding.py     # EmbeddingJudge
│   ├── llm.py           # LLMJudge (granular 0–1 scoring)
│   ├── nli.py           # NLIJudge (softmax + entailment mapping)
│   ├── quantized_nli.py # QuantizedNLIJudge (ONNX INT8)
│   ├── forensic.py      # ForensicJudge (token attribution)
│   └── caching.py       # CachingJudge
├── integrations/
│   ├── __init__.py      # Package marker
│   ├── instructor.py    # Instructor adapter (semantic_validator, SemanticStr)
│   ├── pydantic_ai.py   # Pydantic AI adapter (semantix_validator)
│   └── langchain.py     # LangChain adapter (SemanticValidator)
└── mcp/
    └── server.py        # MCP server (verify_text_intent tool)
```

---

## Development

```bash
git clone https://github.com/labrat-akhona/semantix-ai.git
cd semantix-ai

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Lint
ruff check .
```

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built by <a href="https://github.com/labrat-akhona">Akhona Eland</a> in South Africa 🇿🇦</em>
</p>
