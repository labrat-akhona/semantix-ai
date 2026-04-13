# LangChain

semantix integrates with [LangChain](https://www.langchain.com/) as a composable Runnable that validates chain outputs against an Intent.

## Install

```bash
pip install "semantix-ai[langchain]"
```

## Usage

```python
from semantix import Intent
from semantix.integrations.langchain import SemanticValidator

class Polite(Intent):
    """The text must be polite and professional."""

validator = SemanticValidator(Polite)
chain = prompt | llm | StrOutputParser() | validator
```

`SemanticValidator` implements LangChain's Runnable protocol -- it supports `invoke()`, `ainvoke()`, `batch()`, and the `|` pipe operator.

On failure, it raises `OutputParserException` (if `langchain-core` is installed) or `ValueError`.

## Parameters

```python
SemanticValidator(
    intent: type[Intent],
    judge: Judge | None = None,
)
```

| Parameter | Description |
|---|---|
| `intent` | An Intent subclass whose docstring defines the requirement |
| `judge` | Judge backend override. Defaults to QuantizedNLIJudge. |

## Async support

```python
result = await chain.ainvoke({"input": "handle angry customer"})
```

`SemanticValidator.ainvoke()` works the same as `invoke()` since the local NLI judge is CPU-bound and doesn't benefit from async I/O. However, it integrates correctly with LangChain's async pipeline.

## Batch validation

```python
results = validator.batch(["response 1", "response 2", "response 3"])
```

## Custom judge

```python
from semantix import LLMJudge

validator = SemanticValidator(Polite, judge=LLMJudge(model="gpt-4o-mini"))
chain = prompt | llm | StrOutputParser() | validator
```

## Full example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from semantix import Intent
from semantix.integrations.langchain import SemanticValidator

class Polite(Intent):
    """The text must be polite and professional."""

prompt = ChatPromptTemplate.from_template(
    "You are a customer support agent. Respond to: {input}"
)
llm = ChatOpenAI(model="gpt-4o-mini")
validator = SemanticValidator(Polite)

chain = prompt | llm | StrOutputParser() | validator
result = chain.invoke({"input": "I'm furious about my order!"})
```

## Related

- [DSPy](dspy.md) -- reward functions for DSPy modules
- [Pydantic AI](pydantic-ai.md) -- agent output validation
- [Judges](../judges.md) -- available judge backends
