# Instructor

semantix integrates with [Instructor](https://python.useinstructor.com/) through Pydantic field validators. Validate individual fields in your structured output models with semantic checks.

## Install

```bash
pip install "semantix-ai[instructor]"
```

## SemanticStr shorthand

The quickest way to add semantic validation to a Pydantic model field:

```python
from pydantic import BaseModel
from semantix.integrations.instructor import SemanticStr

class Response(BaseModel):
    reply: SemanticStr["must be polite and professional"]
```

With an explicit threshold:

```python
class Response(BaseModel):
    reply: SemanticStr["must be polite and professional", 0.85]
```

When Instructor parses the LLM response into this model, the `reply` field is validated semantically. On failure, a `ValueError` is raised -- Instructor catches this and retries automatically.

## semantic_validator function

For more control, use `semantic_validator` with an Intent class and Pydantic's `AfterValidator`:

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel
from semantix import Intent
from semantix.integrations.instructor import semantic_validator

class Polite(Intent):
    """The text must be polite and professional."""

class Response(BaseModel):
    reply: Annotated[str, AfterValidator(semantic_validator(Polite))]
```

### Parameters

```python
semantic_validator(
    intent: type[Intent],
    judge: Judge | None = None,
) -> callable
```

| Parameter | Description |
|---|---|
| `intent` | An Intent subclass whose docstring defines the requirement |
| `judge` | Judge backend override. Defaults to QuantizedNLIJudge. |

## Full example with Instructor

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from semantix.integrations.instructor import SemanticStr

class CustomerResponse(BaseModel):
    reply: SemanticStr["must be polite and professional"]
    summary: SemanticStr["must be a concise one-sentence summary"]

client = instructor.from_openai(OpenAI())
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=CustomerResponse,
    messages=[{"role": "user", "content": "Handle this angry customer..."}],
    max_retries=3,  # Instructor retries on validation failure
)
```

## How it works

`SemanticStr["description"]` expands to `Annotated[str, AfterValidator(...)]`. The validator creates a dynamic Intent from the description, evaluates the field value using a local NLI judge, and raises `ValueError` on failure. Instructor intercepts the validation error and retries the LLM call.

## Related

- [Guardrails AI](guardrails.md) -- guard-level validation
- [Pydantic AI](pydantic-ai.md) -- agent output validation
- [Judges](../judges.md) -- available judge backends
