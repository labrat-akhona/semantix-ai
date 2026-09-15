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

The default suite uses mock judges and local fixtures; no model downloads or API
keys are required. Integration tests are excluded by default. To exercise the real
CLI demo and POPIA models (about 79 MB per INT8 model, plus tokenizer files):

```bash
pytest tests/integration/ -m integration -v
```

After the required model files are cached, set `HF_HUB_OFFLINE=1` to run without Hub
requests. Test the demo on release hardware as well: unit-test doubles do not establish
real model accuracy.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format --check .
```

CI runs `ruff check` and `ruff format --check` on every push and PR.

## Project Structure

```
semantix/
  __init__.py          # Public API re-exports
  intent.py            # Intent base class
  decorator.py         # @validate_intent decorator
  composite.py         # AllOf / AnyOf combinators
  training/            # Collection, calibration, fine-tuning exports
  audit/               # Hash-chained certificates and chain summaries
  cli.py               # check / demo / prove / verify / eval commands
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
    dspy.py            # DSPy rewards and metrics
    guardrails.py      # Guardrails validator
  tests/
    conftest.py        # MockJudge, FlipFlopJudge
    test_*.py          # Test modules
```

## Test Conventions

- Use `MockJudge` or `FlipFlopJudge` from `conftest.py` in unit tests. Real-model tests belong under `tests/integration/` and must carry `@pytest.mark.integration`.
- New features should include tests.
- Test files live in `semantix/tests/` and follow the `test_*.py` naming pattern.

## Pull Requests

Build the public documentation before changing user-facing examples:

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

Research notes, outreach drafts, and historical plans stay in the repository but
are excluded from the published documentation site. Run `ruff format .` to apply
formatting when needed; `ruff format --check .` verifies it without editing files.

1. Fork the repo and create a feature branch.
2. Write tests for new functionality.
3. Make sure all tests pass and linting is clean.
4. Open a PR against `master` with a clear description of what and why.
