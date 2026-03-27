
<p align="center">
    <h1 align="center">Semantix</h1>
    <p align="center"><strong>A Semantic Type System for AI Outputs</strong></p>
    <p align="center">
        Define <em>what your LLM output should mean</em>, not just what shape it has.
    </p>
    <p align="center"><em>Created by Akhona Eland, 2026</em></p>
</p>

---

## Why Semantix?

We have **type systems** for data structures (`int`, `str`, Pydantic models), but nothing for **semantic intent**. Semantix fills that gap:

```python
from semantix import Intent, validate_intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude or aggressive."""

@validate_intent
def decline_invite(event: str) -> ProfessionalDecline:
    return call_my_llm(event)   # returns a plain string

result = decline_invite("the company retreat")
# ✓ result is a ProfessionalDecline instance — validated by a judge model
# ✗ raises SemanticIntentError if the output is rude, off-topic, etc.
```

Think of it as **Pydantic for meaning**.

---

## Installation

```bash
# Core (bring your own judge)
pip install semantix

# With OpenAI judge (GPT-4o-mini, accurate)
pip install "semantix[openai]"

# With embedding judge (sentence-transformers, fast, local)
pip install "semantix[embeddings]"

# Everything
pip install "semantix[all]"
```

---

## Quick Start

### 1. Define an Intent

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
    print(result.text)
except SemanticIntentError as e:
    print(f"Failed: {e.intent_name} (score={e.score})")
```

---

## Features

### Swappable Judges

```python
from semantix import EmbeddingJudge, LLMJudge, CachingJudge

# Fast — local cosine similarity (no API key needed)
@validate_intent(judge=EmbeddingJudge())
def fast_fn(x: str) -> MyIntent: ...

# Accurate — asks GPT-4o-mini Yes/No
@validate_intent(judge=LLMJudge(model="gpt-4o-mini"))
def accurate_fn(x: str) -> MyIntent: ...

# Cached — wraps any judge with LRU cache
@validate_intent(judge=CachingJudge(LLMJudge(), maxsize=256))
def cached_fn(x: str) -> MyIntent: ...
```

### Retries

Re-invoke the LLM if the output fails validation:

```python
@validate_intent(judge=EmbeddingJudge(), retries=3)
def decline(event: str) -> ProfessionalDecline:
    return call_llm(event)  # retried up to 3 extra times on failure
```

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

# Iterator wrapper
sc = StreamCollector(ProfessionalDecline, judge=my_judge)
for chunk in sc.wrap(llm_stream()):
    print(chunk, end="")
result = sc.result()  # validated Intent or raises

# Context manager
async with StreamCollector(ProfessionalDecline, judge=my_judge) as sc:
    async for chunk in llm_stream:
        sc.feed(chunk)
result = sc.result()
```

### Observability

All validation events are emitted via Python's `logging` module under the `semantix` logger:

```python
import logging
logging.getLogger("semantix").setLevel(logging.DEBUG)
```

Output:

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
        return Verdict(passed=score >= threshold, score=score)
```

---

## API Reference

| Symbol | Description |
|---|---|
| `Intent` | Base class — subclass with a docstring to define a semantic type |
| `SemanticIntentError` | Raised when validation fails (`.output`, `.score`, `.intent_name`) |
| `@validate_intent` | Decorator — validates return values against their Intent type hint |
| `Judge` | Abstract base — implement `.evaluate()` for custom backends |
| `Verdict` | Dataclass — `.passed`, `.score`, `.reason` |
| `LLMJudge` | OpenAI-based judge (accurate, needs API key) |
| `EmbeddingJudge` | Sentence-transformers judge (fast, local) |
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
├── judges/
│   ├── __init__.py      # Judge ABC + Verdict
│   ├── embedding.py     # EmbeddingJudge
│   ├── llm.py           # LLMJudge
│   └── caching.py       # CachingJudge
└── tests/               # Full test suite (34 tests)
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

---

## License

MIT

---

<p align="center"><em>Semantix was created and is maintained by Akhona Eland (2026).</em></p>
