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

## Features

### Swappable Judges

Choose the right speed/accuracy tradeoff for your use case:

```python
from semantix import EmbeddingJudge, LLMJudge, NLIJudge, CachingJudge

# Fast — local cosine similarity (no API key needed)
@validate_intent(judge=EmbeddingJudge())
def fast_fn(x: str) -> MyIntent: ...

# Accurate — asks GPT-4o-mini Yes/No
@validate_intent(judge=LLMJudge(model="gpt-4o-mini"))
def accurate_fn(x: str) -> MyIntent: ...

# Balanced — local NLI entailment (accurate + no API key)
@validate_intent(judge=NLIJudge())
def balanced_fn(x: str) -> MyIntent: ...

# Cached — wraps any judge with LRU cache
@validate_intent(judge=CachingJudge(LLMJudge(), maxsize=256))
def cached_fn(x: str) -> MyIntent: ...
```

### Smart Retries

Re-invoke the LLM on failure — and tell it *why* it failed:

```python
from semantix import validate_intent, get_last_failure

@validate_intent(judge=EmbeddingJudge(), retries=3)
def decline(event: str) -> ProfessionalDecline:
    hint = ""
    if failure := get_last_failure():
        hint = f"\n\nPrevious attempt scored {failure.score:.2f}. Be more polite."
    return call_llm(f"Decline this invite: {event}{hint}")
```

`get_last_failure()` gives your LLM function access to the previous `SemanticIntentError`, so each retry can be smarter than the last.

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
| `NLIJudge` | Cross-encoder NLI entailment judge (accurate, local) |
| `CachingJudge` | LRU cache wrapper for any judge |
| `AllOf(*intents)` | Composite — all intents must be satisfied |
| `AnyOf(*intents)` | Composite — at least one intent must be satisfied |
| `StreamCollector` | Validates streamed LLM output once fully assembled |

---

## Project Structure

```
semantix/
├── __init__.py          # Public API
├── intent.py            # Intent base class + metaclass
├── exceptions.py        # SemanticIntentError
├── decorator.py         # @validate_intent (retries, logging)
├── composite.py         # AllOf / AnyOf combinators
├── observability.py     # Structured logging
├── streaming.py         # StreamCollector
└── judges/
    ├── __init__.py      # Judge ABC + Verdict
    ├── embedding.py     # EmbeddingJudge
    ├── llm.py           # LLMJudge
    ├── nli.py           # NLIJudge
    └── caching.py       # CachingJudge
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
