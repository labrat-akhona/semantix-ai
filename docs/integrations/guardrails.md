# Guardrails AI

semantix integrates with [Guardrails AI](https://www.guardrailsai.com/) as a custom validator. Use it to add semantic validation to any Guardrails guard.

## Install

```bash
pip install "semantix-ai[guardrails]"
```

## Usage

```python
from guardrails import Guard
from semantix.integrations.guardrails import SemanticIntent

guard = Guard().use(SemanticIntent("must be polite and professional"))
result = guard.validate("Thank you for your patience.")
```

On failure, Guardrails receives a `FailResult` with the score and reason, enabling its built-in retry and correction mechanisms.

## Parameters

```python
SemanticIntent(
    intent: str,
    *,
    threshold: float | None = None,
    judge: Judge | None = None,
    on_fail: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `intent` | Plain English description of what the text should mean |
| `threshold` | Minimum score to pass (0-1). Defaults to the judge's recommended threshold. |
| `judge` | Judge backend override. Defaults to QuantizedNLIJudge. |
| `on_fail` | Guardrails on_fail action (e.g. `"reask"`, `"exception"`). |

## With retries

Guardrails handles retries natively. Combine with `on_fail="reask"` for automatic correction:

```python
guard = Guard().use(
    SemanticIntent(
        "must be polite and professional",
        on_fail="reask",
    )
)
```

## Custom judge

```python
from semantix import LLMJudge

guard = Guard().use(
    SemanticIntent(
        "must be polite and professional",
        judge=LLMJudge(model="gpt-4o-mini"),
    )
)
```

## How it works

`SemanticIntent` is registered as a Guardrails validator under the name `semantix/semantic_intent`. Internally, it creates a dynamic Intent from the description string and evaluates it using the configured judge. On failure, it returns a `FailResult` with a message containing the score, reason, and the requirement.

## Related

- [Instructor](instructor.md) -- Pydantic field-level validation
- [Pydantic AI](pydantic-ai.md) -- agent output validation
- [Judges](../judges.md) -- available judge backends
