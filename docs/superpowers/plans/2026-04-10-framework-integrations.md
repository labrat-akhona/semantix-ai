# Framework Integrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight, native-feeling integration adapters for Instructor, Pydantic AI, and LangChain so users can validate LLM outputs with semantix inside these frameworks.

**Architecture:** Three independent adapter files under `semantix/integrations/`, one per framework. Each translates a semantix `Verdict` into the framework's native retry/error mechanism. No shared base class. Frameworks are optional dependencies — each adapter guards its imports.

**Tech Stack:** Python 3.10+, Pydantic v2, instructor, pydantic-ai, langchain-core

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `semantix/integrations/__init__.py` | Empty package init |
| Create | `semantix/integrations/instructor.py` | Instructor adapter: `semantic_validator`, `SemanticStr` |
| Create | `semantix/integrations/pydantic_ai.py` | Pydantic AI adapter: `semantix_validator` |
| Create | `semantix/integrations/langchain.py` | LangChain adapter: `SemanticValidator` |
| Create | `semantix/tests/test_instructor_integration.py` | Tests for Instructor adapter |
| Create | `semantix/tests/test_pydantic_ai_integration.py` | Tests for Pydantic AI adapter |
| Create | `semantix/tests/test_langchain_integration.py` | Tests for LangChain adapter |
| Modify | `pyproject.toml` | Add optional dependency groups |
| Modify | `semantix/__init__.py` | No changes needed — integrations are imported explicitly |

---

### Task 1: Package scaffold and dependencies

**Files:**
- Create: `semantix/integrations/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the integrations package**

```python
# semantix/integrations/__init__.py
"""Framework integration adapters for semantix."""
```

- [ ] **Step 2: Add optional dependency groups to pyproject.toml**

In `pyproject.toml`, add these entries under `[project.optional-dependencies]`:

```toml
instructor = ["instructor>=1.0"]
pydantic-ai = ["pydantic-ai>=0.1"]
langchain = ["langchain-core>=0.3"]
```

Update the `all` extra to include the three new groups:

```toml
all = [
    "openai>=1.0",
    "sentence-transformers>=2.2",
    "mcp[cli]>=1.0",
    "onnxruntime>=1.16",
    "tokenizers>=0.15",
    "huggingface-hub>=0.20",
    "instructor>=1.0",
    "pydantic-ai>=0.1",
    "langchain-core>=0.3",
]
```

Also update the `dev` extra to include them:

```toml
dev = [
    "openai>=1.0",
    "sentence-transformers>=2.2",
    "mcp[cli]>=1.0",
    "onnxruntime>=1.16",
    "tokenizers>=0.15",
    "huggingface-hub>=0.20",
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.4.0",
    "instructor>=1.0",
    "pydantic-ai>=0.1",
    "langchain-core>=0.3",
]
```

- [ ] **Step 3: Commit**

```bash
git add semantix/integrations/__init__.py pyproject.toml
git commit -m "chore: scaffold integrations package and add framework dependencies"
```

---

### Task 2: Instructor integration — tests

**Files:**
- Create: `semantix/tests/test_instructor_integration.py`

- [ ] **Step 1: Write tests for `semantic_validator`**

```python
"""Tests for the Instructor integration adapter."""

from __future__ import annotations

import pytest

from semantix.intent import Intent
from semantix.judges import Verdict
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


class Helpful(Intent):
    """The text must be helpful and informative."""
    threshold = 0.9


# ── semantic_validator ──────────────────────────────────────────────


def test_semantic_validator_passes():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=True, score=0.95)
    validator = semantic_validator(Polite, judge=judge)

    result = validator("Thank you for reaching out.")
    assert result == "Thank you for reaching out."
    assert judge.call_count == 1
    assert judge.last_description == Polite.description()


def test_semantic_validator_fails_raises_value_error():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=False, score=0.3, reason="Too aggressive")
    validator = semantic_validator(Polite, judge=judge)

    with pytest.raises(ValueError, match="Semantic validation failed"):
        validator("Get lost!")


def test_semantic_validator_includes_score_and_reason_in_error():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=False, score=0.42, reason="Not polite enough")
    validator = semantic_validator(Polite, judge=judge)

    with pytest.raises(ValueError, match="0.42") as exc_info:
        validator("Whatever.")
    assert "Not polite enough" in str(exc_info.value)


def test_semantic_validator_respects_intent_threshold():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=True, score=0.85)
    validator = semantic_validator(Helpful, judge=judge)

    validator("Here is the information you need.")
    # Helpful.threshold is 0.9, so the judge should receive 0.9 as threshold
    # MockJudge doesn't use threshold but we verify it was called
    assert judge.call_count == 1


def test_semantic_validator_converts_non_string():
    from semantix.integrations.instructor import semantic_validator

    judge = MockJudge(passed=True, score=0.95)
    validator = semantic_validator(Polite, judge=judge)

    result = validator(42)
    assert result == 42
    assert judge.last_output == "42"


# ── SemanticStr ─────────────────────────────────────────────────────


def test_semantic_str_basic():
    from semantix.integrations.instructor import SemanticStr

    # SemanticStr["description"] should produce an Annotated type
    annotated_type = SemanticStr["must be polite"]
    # Verify it's an annotated type with metadata
    assert hasattr(annotated_type, "__metadata__")


def test_semantic_str_with_threshold():
    from semantix.integrations.instructor import SemanticStr

    annotated_type = SemanticStr["must be polite", 0.9]
    assert hasattr(annotated_type, "__metadata__")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_instructor_integration.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` because `semantix.integrations.instructor` doesn't exist yet.

- [ ] **Step 3: Commit**

```bash
git add semantix/tests/test_instructor_integration.py
git commit -m "test: add failing tests for Instructor integration"
```

---

### Task 3: Instructor integration — implementation

**Files:**
- Create: `semantix/integrations/instructor.py`

- [ ] **Step 1: Implement `semantic_validator` and `SemanticStr`**

```python
"""Instructor integration — validate fields with semantix Intents.

Usage with semantic_validator::

    from semantix.integrations.instructor import semantic_validator
    from semantix import Intent

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    class Response(BaseModel):
        reply: Annotated[str, AfterValidator(semantic_validator(Polite))]

Usage with SemanticStr shorthand::

    from semantix.integrations.instructor import SemanticStr

    class Response(BaseModel):
        reply: SemanticStr["must be polite", 0.85]
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator

from semantix.intent import Intent
from semantix.judges import Judge


def _default_judge() -> Judge:
    """Resolve the default judge — same logic as @validate_intent."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


def semantic_validator(
    intent: type[Intent],
    judge: Judge | None = None,
) -> callable:
    """Return a Pydantic AfterValidator-compatible callable.

    On success the original value is returned unchanged.
    On failure a ``ValueError`` is raised with the score and reason —
    Instructor catches this and retries automatically.
    """
    _judge = judge or _default_judge()
    description = intent.description()
    threshold = intent.threshold

    def _validate(value: Any) -> Any:
        text = str(value) if not isinstance(value, str) else value
        verdict = _judge.evaluate(text, description, threshold)
        if not verdict.passed:
            score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
            reason_str = f": {verdict.reason}" if verdict.reason else ""
            raise ValueError(
                f"Semantic validation failed (score={score_str}){reason_str}. "
                f"The text must satisfy: {description}"
            )
        return value

    return _validate


class _SemanticStrMeta(type):
    """Metaclass enabling ``SemanticStr["description", threshold]`` syntax."""

    def __getitem__(cls, params: str | tuple) -> Any:
        if isinstance(params, str):
            desc, threshold = params, 0.8
        elif isinstance(params, tuple) and len(params) == 2:
            desc, threshold = params
        else:
            raise TypeError(
                "SemanticStr expects SemanticStr['description'] or "
                "SemanticStr['description', threshold]"
            )

        # Create a dynamic Intent subclass from the description string.
        dynamic_intent = type(
            f"_SemanticStr_{id(desc)}",
            (Intent,),
            {"__doc__": desc, "threshold": threshold},
        )

        return Annotated[str, AfterValidator(semantic_validator(dynamic_intent))]


class SemanticStr(metaclass=_SemanticStrMeta):
    """Shorthand for annotating string fields with semantic validation.

    ``SemanticStr["must be polite"]`` is equivalent to defining an Intent
    subclass with that docstring and wrapping it in an AfterValidator.
    """
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_instructor_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add semantix/integrations/instructor.py
git commit -m "feat: add Instructor integration — semantic_validator and SemanticStr"
```

---

### Task 4: Pydantic AI integration — tests

**Files:**
- Create: `semantix/tests/test_pydantic_ai_integration.py`

- [ ] **Step 1: Write tests for `semantix_validator`**

```python
"""Tests for the Pydantic AI integration adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


# ── semantix_validator ──────────────────────────────────────────────


def test_semantix_validator_passes_string_output():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=True, score=0.95)
    validator_fn = semantix_validator(Polite, judge=judge)

    # Simulate RunContext
    ctx = MagicMock()
    result = validator_fn(ctx, "Thank you for your patience.")
    assert result == "Thank you for your patience."
    assert judge.call_count == 1


def test_semantix_validator_fails_raises_model_retry():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=False, score=0.3, reason="Too rude")
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()

    # ModelRetry might not be installed, so catch the actual exception type
    with pytest.raises(Exception, match="Semantic validation failed"):
        validator_fn(ctx, "Get lost!")


def test_semantix_validator_includes_score_in_retry_message():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=False, score=0.42, reason="Not professional")
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()
    with pytest.raises(Exception, match="0.42"):
        validator_fn(ctx, "Whatever.")


def test_semantix_validator_handles_non_string_output():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=True, score=0.95)
    validator_fn = semantix_validator(Polite, judge=judge)

    ctx = MagicMock()
    result = validator_fn(ctx, 42)
    assert result == 42
    assert judge.last_output == "42"


def test_semantix_validator_with_judge_from_deps():
    from semantix.integrations.pydantic_ai import semantix_validator

    judge = MockJudge(passed=True, score=0.95)
    validator_fn = semantix_validator(Polite, judge_from_deps=True)

    ctx = MagicMock()
    ctx.deps = judge
    result = validator_fn(ctx, "Thank you.")
    assert result == "Thank you."
    assert judge.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_pydantic_ai_integration.py -v`
Expected: FAIL — `ImportError` because `semantix.integrations.pydantic_ai` doesn't exist yet.

- [ ] **Step 3: Commit**

```bash
git add semantix/tests/test_pydantic_ai_integration.py
git commit -m "test: add failing tests for Pydantic AI integration"
```

---

### Task 5: Pydantic AI integration — implementation

**Files:**
- Create: `semantix/integrations/pydantic_ai.py`

- [ ] **Step 1: Implement `semantix_validator`**

```python
"""Pydantic AI integration — validate agent outputs with semantix Intents.

Usage::

    from semantix.integrations.pydantic_ai import semantix_validator
    from semantix import Intent
    from pydantic_ai import Agent

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    agent = Agent("openai:gpt-4o", output_type=str)
    agent.output_validator(semantix_validator(Polite))
"""

from __future__ import annotations

from typing import Any

from semantix.intent import Intent
from semantix.judges import Judge


def _default_judge() -> Judge:
    """Resolve the default judge — same logic as @validate_intent."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


def semantix_validator(
    intent: type[Intent],
    judge: Judge | None = None,
    judge_from_deps: bool = False,
) -> Any:
    """Return a function compatible with ``@agent.output_validator``.

    Parameters
    ----------
    intent:
        The Intent subclass whose docstring defines the semantic requirement.
    judge:
        Judge to use. Defaults to QuantizedNLIJudge or NLIJudge.
        Ignored when ``judge_from_deps=True``.
    judge_from_deps:
        If True, read the judge from ``ctx.deps`` at runtime instead of
        using a fixed judge. The deps object must be a ``Judge`` instance.
    """
    _judge = None if judge_from_deps else (judge or _default_judge())
    description = intent.description()
    threshold = intent.threshold

    def _validate(ctx: Any, output: Any) -> Any:
        active_judge = ctx.deps if judge_from_deps else _judge
        text = str(output) if not isinstance(output, str) else output
        verdict = active_judge.evaluate(text, description, threshold)
        if not verdict.passed:
            score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
            reason_str = f": {verdict.reason}" if verdict.reason else ""
            # Import ModelRetry at call time — pydantic-ai may not be installed.
            try:
                from pydantic_ai import ModelRetry

                raise ModelRetry(
                    f"Semantic validation failed (score={score_str}){reason_str}. "
                    f"The text must satisfy: {description}"
                )
            except ImportError:
                raise ValueError(
                    f"Semantic validation failed (score={score_str}){reason_str}. "
                    f"The text must satisfy: {description}"
                )
        return output

    return _validate
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_pydantic_ai_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add semantix/integrations/pydantic_ai.py
git commit -m "feat: add Pydantic AI integration — semantix_validator"
```

---

### Task 6: LangChain integration — tests

**Files:**
- Create: `semantix/tests/test_langchain_integration.py`

- [ ] **Step 1: Write tests for `SemanticValidator`**

```python
"""Tests for the LangChain integration adapter."""

from __future__ import annotations

import pytest

from semantix.intent import Intent
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite and professional."""


# ── SemanticValidator ───────────────────────────────────────────────


def test_semantic_validator_invoke_passes():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=True, score=0.95)
    validator = SemanticValidator(Polite, judge=judge)

    result = validator.invoke("Thank you for your patience.")
    assert result == "Thank you for your patience."
    assert judge.call_count == 1


def test_semantic_validator_invoke_fails():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=False, score=0.3, reason="Too rude")
    validator = SemanticValidator(Polite, judge=judge)

    with pytest.raises(Exception, match="Semantic validation failed"):
        validator.invoke("Get lost!")


def test_semantic_validator_error_includes_score():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=False, score=0.42, reason="Not polite")
    validator = SemanticValidator(Polite, judge=judge)

    with pytest.raises(Exception, match="0.42"):
        validator.invoke("Whatever.")


def test_semantic_validator_handles_non_string():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=True, score=0.95)
    validator = SemanticValidator(Polite, judge=judge)

    result = validator.invoke(42)
    assert result == 42
    assert judge.last_output == "42"


def test_semantic_validator_batch():
    from semantix.integrations.langchain import SemanticValidator

    judge = MockJudge(passed=True, score=0.95)
    validator = SemanticValidator(Polite, judge=judge)

    results = validator.batch(["Hello.", "Thank you."])
    assert results == ["Hello.", "Thank you."]
    assert judge.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest semantix/tests/test_langchain_integration.py -v`
Expected: FAIL — `ImportError` because `semantix.integrations.langchain` doesn't exist yet.

- [ ] **Step 3: Commit**

```bash
git add semantix/tests/test_langchain_integration.py
git commit -m "test: add failing tests for LangChain integration"
```

---

### Task 7: LangChain integration — implementation

**Files:**
- Create: `semantix/integrations/langchain.py`

- [ ] **Step 1: Implement `SemanticValidator`**

```python
"""LangChain integration — validate chain outputs with semantix Intents.

Usage::

    from semantix.integrations.langchain import SemanticValidator
    from semantix import Intent

    class Polite(Intent):
        \"\"\"The text must be polite and professional.\"\"\"

    validator = SemanticValidator(Polite)
    chain = prompt | llm | StrOutputParser() | validator
"""

from __future__ import annotations

from typing import Any

from semantix.intent import Intent
from semantix.judges import Judge


def _default_judge() -> Judge:
    """Resolve the default judge — same logic as @validate_intent."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


class SemanticValidator:
    """A LangChain-compatible Runnable that validates text against an Intent.

    Composes with LangChain's ``|`` pipe syntax. On failure raises
    ``OutputParserException`` if langchain-core is installed, otherwise
    ``ValueError``.
    """

    def __init__(self, intent: type[Intent], judge: Judge | None = None) -> None:
        self._judge = judge or _default_judge()
        self._description = intent.description()
        self._threshold = intent.threshold

    def _validate(self, value: Any) -> Any:
        text = str(value) if not isinstance(value, str) else value
        verdict = self._judge.evaluate(text, self._description, self._threshold)
        if not verdict.passed:
            score_str = f"{verdict.score:.2f}" if verdict.score is not None else "N/A"
            reason_str = f": {verdict.reason}" if verdict.reason else ""
            msg = (
                f"Semantic validation failed (score={score_str}){reason_str}. "
                f"The text must satisfy: {self._description}"
            )
            try:
                from langchain_core.exceptions import OutputParserException

                raise OutputParserException(msg)
            except ImportError:
                raise ValueError(msg)
        return value

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Validate a single input."""
        return self._validate(input)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Async validate a single input."""
        return self._validate(input)

    def batch(self, inputs: list[Any], config: Any = None, **kwargs: Any) -> list[Any]:
        """Validate a batch of inputs."""
        return [self._validate(inp) for inp in inputs]

    def __or__(self, other: Any) -> Any:
        """Support ``validator | next_step`` pipe syntax."""
        try:
            from langchain_core.runnables import RunnableSequence

            return RunnableSequence(first=self, last=other)
        except ImportError:
            raise ImportError("langchain-core is required for pipe composition")

    def __ror__(self, other: Any) -> Any:
        """Support ``prev_step | validator`` pipe syntax."""
        try:
            from langchain_core.runnables import RunnableSequence

            return RunnableSequence(first=other, last=self)
        except ImportError:
            raise ImportError("langchain-core is required for pipe composition")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest semantix/tests/test_langchain_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add semantix/integrations/langchain.py
git commit -m "feat: add LangChain integration — SemanticValidator"
```

---

### Task 8: Run full test suite and final commit

**Files:**
- None — verification only

- [ ] **Step 1: Run the entire test suite**

Run: `python3 -m pytest semantix/tests/ -v`
Expected: All 126 existing tests + new integration tests PASS. No regressions.

- [ ] **Step 2: Run linter**

Run: `python3 -m ruff check semantix/integrations/ semantix/tests/test_instructor_integration.py semantix/tests/test_pydantic_ai_integration.py semantix/tests/test_langchain_integration.py`
Expected: No errors.

- [ ] **Step 3: Fix any issues found**

If linter or tests report issues, fix them and commit.

- [ ] **Step 4: Bump version to 0.1.6**

In `pyproject.toml`, change `version = "0.1.5.post2"` to `version = "0.1.6"`.
In `semantix/__init__.py`, change `__version__ = "0.1.5.post2"` to `__version__ = "0.1.6"`.

```bash
git add pyproject.toml semantix/__init__.py
git commit -m "chore: bump version to 0.1.6"
```
