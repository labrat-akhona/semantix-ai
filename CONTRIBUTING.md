# Contributing to semantix-ai

Thanks for your interest in contributing! This guide will get you up and running.

## Dev Setup

```bash
git clone https://github.com/labrat-akhona/semantix-ai.git
cd semantix-ai
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest semantix/tests/ -v
```

All tests use mock judges — no model downloads or API keys required.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

CI runs `ruff check` and `ruff format --check` on every push and PR.

## Project Structure

```
semantix/
  __init__.py          # Public API re-exports
  intent.py            # Intent base class
  decorator.py         # @validate_intent decorator
  composite.py         # AllOf / AnyOf combinators
  training.py          # TrainingCollector for fine-tuning data
  judges/
    __init__.py        # Judge ABC + Verdict
    nli.py             # NLI cross-encoder judge
    quantized_nli.py   # ONNX-quantized NLI judge
    embedding.py       # Embedding similarity judge
    llm.py             # LLM-as-judge (OpenAI)
    caching.py         # CachingJudge wrapper
    forensic.py        # ForensicJudge wrapper (audit trail)
  integrations/
    instructor.py      # Instructor / Pydantic field validation
    pydantic_ai.py     # Pydantic AI output validator
    langchain.py       # LangChain Runnable
  tests/
    conftest.py        # MockJudge, FlipFlopJudge
    test_*.py          # Test modules
```

## Test Conventions

- Use `MockJudge` or `FlipFlopJudge` from `conftest.py` — never download real models in tests.
- New features should include tests.
- Test files live in `semantix/tests/` and follow the `test_*.py` naming pattern.

## Pull Requests

1. Fork the repo and create a feature branch.
2. Write tests for new functionality.
3. Make sure all tests pass and linting is clean.
4. Open a PR against `master` with a clear description of what and why.
