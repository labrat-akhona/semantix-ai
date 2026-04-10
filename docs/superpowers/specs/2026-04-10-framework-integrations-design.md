# Framework Integrations Design — Instructor, Pydantic AI, LangChain

**Date:** 2026-04-10
**Author:** Akhona Eland + Claude
**Status:** Approved
**Version target:** v0.1.6

---

## Goal

Add lightweight, native-feeling adapters so semantix can validate LLM outputs inside the three most popular structured-output frameworks: **Instructor**, **Pydantic AI**, and **LangChain**. Each adapter translates a semantix `Verdict` into the framework's native retry/error mechanism — semantix judges meaning, the framework orchestrates retries.

## Approach

**A with a touch of B** — thin adapter files (one per framework, ~80 lines each) that feel native to each framework's idioms. No shared base class, no abstraction layer, no framework-specific Intent subclasses.

### Design Principles

- Retries are delegated to the framework, not duplicated by semantix
- Each integration is a single file under `semantix/integrations/`
- Frameworks are optional dependencies — import errors are handled gracefully
- Users still define `Intent` classes and pick judges the same way they always do

---

## Package Structure

```
semantix/
  integrations/
    __init__.py
    instructor.py
    pydantic_ai.py
    langchain.py
```

### New Optional Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
instructor = ["instructor>=1.0"]
pydantic-ai = ["pydantic-ai>=0.1"]
langchain = ["langchain-core>=0.3"]
```

The `all` extra will be updated to include these three.

---

## Integration 1: Instructor

**File:** `semantix/integrations/instructor.py`

### Primitives

#### `semantic_validator(intent, judge=None)`

A factory that returns a callable compatible with Pydantic's `AfterValidator`. Runs a semantix judge against the field value using the Intent's docstring as the semantic requirement.

- On pass: returns the value unchanged
- On fail: raises `ValueError` with score and reason — Instructor catches this and retries automatically

```python
from semantix.integrations.instructor import semantic_validator
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

class Response(BaseModel):
    reply: Annotated[str, AfterValidator(semantic_validator(Polite))]

response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Response,
    max_retries=2,
    messages=[...]
)
```

#### `SemanticStr` — Shorthand type alias factory

Creates an Intent on the fly from a string description and optional threshold, wrapped in `AfterValidator`. Sugar for quick prototyping.

```python
from semantix.integrations.instructor import SemanticStr

class Response(BaseModel):
    reply: SemanticStr["must be polite and professional", 0.85]
```

Internally, `SemanticStr.__class_getitem__` creates a dynamic Intent subclass from the description string, then returns `Annotated[str, AfterValidator(semantic_validator(dynamic_intent))]`.

### Judge Resolution

If no judge is passed to `semantic_validator`, it uses the same default resolution as `@validate_intent`: QuantizedNLIJudge if onnxruntime is available, otherwise NLIJudge, otherwise raises ImportError.

---

## Integration 2: Pydantic AI

**File:** `semantix/integrations/pydantic_ai.py`

### Primitives

#### `semantix_validator(intent, judge=None, judge_from_deps=False)`

A factory that returns a function compatible with Pydantic AI's `@agent.output_validator` decorator. The returned function:

1. Extracts text from the output (handles `str`, `BaseModel` with common text fields)
2. Runs the judge against the Intent description
3. On pass: returns the output unchanged
4. On fail: raises `ModelRetry` with the semantix failure reason and score

```python
from semantix.integrations.pydantic_ai import semantix_validator
from semantix import Intent
from pydantic_ai import Agent

class Polite(Intent):
    """The text must be polite and professional."""

agent = Agent("openai:gpt-4o", output_type=str)
agent.output_validator(semantix_validator(Polite))

result = agent.run_sync("Decline the meeting invitation")
```

#### Runtime judge via dependencies

When `judge_from_deps=True`, the validator reads the judge from `RunContext.deps` instead of using a fixed judge. The deps object must be a `Judge` instance.

```python
agent = Agent("openai:gpt-4o", output_type=str, deps_type=Judge)
agent.output_validator(semantix_validator(Polite, judge_from_deps=True))
result = agent.run_sync("Decline the invite", deps=NLIJudge())
```

---

## Integration 3: LangChain

**File:** `semantix/integrations/langchain.py`

### Primitives

#### `SemanticValidator(intent, judge=None)`

A `Runnable` that validates text against an Intent. Composes with LangChain's `|` pipe syntax.

- On pass: returns the text unchanged
- On fail: raises `OutputParserException` with semantix failure reason

```python
from semantix.integrations.langchain import SemanticValidator
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

validator = SemanticValidator(Polite)
chain = prompt | llm | StrOutputParser() | validator
result = chain.invoke({"event": "the company retreat"})
```

Compatible with LangChain's `RetryWithErrorOutputParser` for self-healing retries.

Implements `Runnable` to support:
- `|` pipe composition
- `.invoke()`, `.ainvoke()`, `.batch()` interfaces
- Input type: `str`
- Output type: `str`

---

## Out of Scope

- No framework-specific Intent subclasses
- No custom judge implementations per framework
- No streaming support in integrations (users use `StreamCollector` directly)
- No audit hooks wired into integrations (users use `AuditEngine` separately)
- No self-training features (planned for future)

---

## Testing Strategy

Each integration gets its own test file under `semantix/tests/`:

- `test_instructor_integration.py`
- `test_pydantic_ai_integration.py`
- `test_langchain_integration.py`

Tests mock the judge to return controlled `Verdict` objects, so no LLM calls or model downloads are needed. Each test file covers:

1. Validation passes — output returned unchanged
2. Validation fails — correct framework-native error raised with semantix reason
3. Default judge resolution works
4. Edge cases: empty strings, non-string outputs, missing dependencies

Framework imports are guarded with `pytest.importorskip()` so tests are skipped if the framework isn't installed.

---

## Future Considerations

- **Self-training pipeline:** Validation data (scores, failures, corrections) collected via AuditEngine could feed into a fine-tuning pipeline. Not in scope for v0.1.6 but the integration design doesn't preclude it.
- **Additional frameworks:** CrewAI, AutoGen, DSPy — same adapter pattern applies.
- **Streaming validation:** Could add framework-specific streaming validators if demand exists.
