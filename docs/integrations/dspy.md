# DSPy

semantix integrates with [DSPy](https://dspy.ai/) (v2.6+) by providing reward functions and metric functions compatible with `dspy.Refine`, `dspy.BestOfN`, and `dspy.Evaluate`.

## Install

```bash
pip install "semantix-ai[dspy]"
```

## semantic_reward

Use `semantic_reward` with `dspy.Refine` or `dspy.BestOfN`:

```python
import dspy
from semantix import Intent
from semantix.integrations.dspy import semantic_reward

class Polite(Intent):
    """The text must be polite and professional."""

qa = dspy.ChainOfThought("question -> answer")
refined = dspy.Refine(module=qa, N=3, reward_fn=semantic_reward(Polite))
result = refined(question="Handle an angry customer")
```

With `dspy.BestOfN`:

```python
best = dspy.BestOfN(module=qa, N=5, reward_fn=semantic_reward(Polite))
result = best(question="Handle an angry customer")
```

### Parameters

```python
semantic_reward(
    intent: type[Intent] | str,
    *,
    field: str | None = None,
    judge: Judge | None = None,
    threshold: float | None = None,
) -> Callable[[dict, Any], float]
```

| Parameter | Description |
|---|---|
| `intent` | An Intent subclass or a plain English description string |
| `field` | The prediction field to validate. If `None`, uses the last field. |
| `judge` | Judge backend override. Defaults to QuantizedNLIJudge. |
| `threshold` | Score threshold. Outputs at or above get reward 1.0, below get raw score. |

### Using a plain string

```python
reward = semantic_reward("must be polite and professional")
refined = dspy.Refine(module=qa, N=3, reward_fn=reward)
```

### Specifying a field

```python
reward = semantic_reward(Polite, field="answer")
```

## semantic_metric

Use `semantic_metric` with `dspy.Evaluate` and DSPy optimizers like `dspy.MIPROv2`:

```python
from semantix.integrations.dspy import semantic_metric

metric = semantic_metric(Polite)

# With dspy.Evaluate
evaluator = dspy.Evaluate(devset=dev_data, metric=metric)
score = evaluator(my_module)

# With optimizers
optimized = dspy.MIPROv2(metric=metric, ...)(my_module, trainset=train_data)
```

### Parameters

```python
semantic_metric(
    intent: type[Intent] | str,
    *,
    field: str | None = None,
    judge: Judge | None = None,
    threshold: float | None = None,
) -> Callable[..., float]
```

Same parameters as `semantic_reward`. The returned function has the signature `metric(example, pred) -> float` expected by DSPy evaluators and optimizers.

## How it works

The reward/metric function extracts text from the prediction's specified field (or the last field by default), evaluates it against the intent using a local NLI judge, and returns the entailment score as a float between 0.0 and 1.0.

## Related

- [LangChain](langchain.md) -- chain-level validation
- [Pydantic AI](pydantic-ai.md) -- agent output validation
- [Judges](../judges.md) -- available judge backends
