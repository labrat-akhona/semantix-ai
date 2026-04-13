# Pydantic AI

semantix integrates with [Pydantic AI](https://ai.pydantic.dev/) as an output validator for agents. On failure it raises `ModelRetry`, triggering Pydantic AI's built-in retry mechanism.

## Install

```bash
pip install "semantix-ai[pydantic-ai]"
```

## Usage

```python
from pydantic_ai import Agent
from semantix import Intent
from semantix.integrations.pydantic_ai import semantix_validator

class Polite(Intent):
    """The text must be polite and professional."""

agent = Agent("openai:gpt-4o", output_type=str)
agent.output_validator(semantix_validator(Polite))

result = agent.run_sync("Handle this angry customer complaint")
```

On validation failure, `semantix_validator` raises `ModelRetry` with a message containing the score, reason, and the requirement. Pydantic AI automatically retries the agent call.

## Parameters

```python
semantix_validator(
    intent: type[Intent],
    judge: Judge | None = None,
    judge_from_deps: bool = False,
)
```

| Parameter | Description |
|---|---|
| `intent` | An Intent subclass whose docstring defines the requirement |
| `judge` | Judge backend override. Defaults to QuantizedNLIJudge. |
| `judge_from_deps` | If `True`, reads the judge from `ctx.deps` at runtime instead of using a fixed judge. The deps object must be a `Judge` instance. |

## Dynamic judge from deps

Pass the judge at runtime via Pydantic AI's dependency injection:

```python
from semantix import NLIJudge

agent = Agent("openai:gpt-4o", output_type=str, deps_type=Judge)
agent.output_validator(semantix_validator(Polite, judge_from_deps=True))

result = agent.run_sync(
    "Handle this complaint",
    deps=NLIJudge(),
)
```

This is useful when you want to swap judges in tests or per-request.

## Combining with negation

```python
class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

agent.output_validator(semantix_validator(~MedicalAdvice))
```

## Related

- [Instructor](instructor.md) -- Pydantic field-level validation
- [Guardrails AI](guardrails.md) -- guard-level validation
- [LangChain](langchain.md) -- chain-level validation
- [Judges](../judges.md) -- available judge backends
