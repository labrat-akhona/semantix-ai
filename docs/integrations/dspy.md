# DSPy

semantix integrates with [DSPy](https://dspy.ai/) (v2.6+) by providing reward and metric functions that are compatible with `dspy.Refine`, `dspy.BestOfN`, `dspy.Evaluate`, and optimizers like `dspy.MIPROv2` — running locally on a quantized NLI model, with no API calls and ~15 ms per evaluation.

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

## Benchmarks

`semantic_reward` has been benchmarked against a 70B LLM-judge baseline (Groq Llama 3.3 70B) across two DSPy optimization tasks. Summary:

| Dimension | semantix (local NLI) | Groq Llama 3.3 70B |
|---|---|---|
| Avg latency per evaluation | ~15 ms | ~500-900 ms |
| Cost per 1,000 evaluations | $0 | ~$0.13 |
| API key required | No | Yes |
| Reward-agreement with Gemini 2.5 Flash proxy | <!-- paste Pearson r when results land --> | <!-- paste --> |
| `BestOfN(N=5)` paired win-rate | <!-- paste --> | <!-- paste --> |

Full methodology, raw CSVs, run metadata, and notebooks (which render on GitHub): [`benchmarks/dspy/`](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks/dspy).

### Two experiments per task

- **Reward-agreement** — every judge scores every output; measure Pearson r vs Gemini 2.5 Flash (operational proxy-ground-truth).
- **Optimization-impact** — `dspy.BestOfN(N=5)` with semantix as `reward_fn`, vs the same loop with Groq as `reward_fn`. Final output scored by Flash. Report paired win/loss/tie.

### When to prefer semantix over an LLM-judge reward

- You're iterating on a DSPy program and want reward evaluation to not gate your feedback loop.
- You need deterministic, seedable rewards (NLI scores are stable; LLM judges are not).
- You want to run optimization on CI without API costs or rate limits.

### When to prefer an LLM-judge

- The intent requires world knowledge or multi-step reasoning beyond entailment.
- You need freeform rationales, not a single score.
- Evaluation volume is low (<100 calls) and the latency/cost trade-off doesn't matter.

## Related

- [LangChain](langchain.md) -- chain-level validation
- [Pydantic AI](pydantic-ai.md) -- agent output validation
- [Judges](../judges.md) -- available judge backends
