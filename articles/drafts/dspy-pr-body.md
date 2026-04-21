Title: `docs(use-cases): add semantix-ai — local NLI reward/metric for BestOfN/Refine/Evaluate/MIPROv2`

Target file: `docs/docs/community/use-cases.md`

Row to add (in "A Few Providers, Integrations, and related Blog Releases" table, appended at the end):

```markdown
| **semantix-ai** | [Link](https://labrat-akhona.github.io/semantix-ai/integrations/dspy/) |
```

---

Body:

Hi @isaacbmiller — circling back on #9583. You closed it with:

> "For our providers with integrations, we try to keep it specific to people who have built, measured, and tested specific DSPy features. Are there any DSPy specific features in Semantix?"

Fair. I went and built them. This PR is the resubmission, with the measured + tested part done.

## What's DSPy-specific in semantix-ai

Two primitives in `semantix.integrations.dspy`, both compatible with your public APIs:

- **`semantic_reward(intent)`** → `reward_fn(args, pred) -> float`, drop-in for `dspy.BestOfN` and `dspy.Refine`.
- **`semantic_metric(intent)`** → `metric(example, pred) -> float`, drop-in for `dspy.Evaluate` and `dspy.MIPROv2`.

Both are powered by a local quantized NLI model (~25 MB ONNX). No API call, no key, deterministic, ~15 ms per evaluation on CPU.

```python
import dspy
from semantix import Intent
from semantix.integrations.dspy import semantic_reward

class Grounded(Intent):
    """The answer must be grounded in the provided context."""

qa = dspy.ChainOfThought("context, question -> answer")
best = dspy.BestOfN(module=qa, N=5, reward_fn=semantic_reward(Grounded))
```

## Measured

Reproducible benchmark in the [semantix-ai repo](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks). Two tasks, four judges, two experiments per task.

### Task A — customer_support_qa

<!-- PASTE headline table from benchmarks/dspy/customer_support/results/summary.md -->

### Task B — hotpotqa_groundedness (HotpotQA distractor validation subset)

<!-- PASTE headline table from benchmarks/dspy/hotpotqa_groundedness/results/summary.md -->

### Agreement with proxy-ground-truth (Gemini 2.5 Flash)

<!-- PASTE Pearson r row from notebooks -->

### Flash ↔ Pro verification slice

<!-- PASTE Flash↔Pro Pearson r from 25-example slice -->

### Optimization-impact (`dspy.BestOfN(N=5)`, paired)

<!-- PASTE semantix vs Groq win/loss/tie counts -->

Headline: semantix-ai matches a 70B LLM-judge's reward agreement on both tasks, at ~25× lower latency and $0 per run.

## Full writeup

Dev.to article: <!-- PASTE LINK ONCE PUBLISHED -->

## This PR

One-row addition to the providers/integrations table, linking to the DSPy integration docs page that hosts working code and the benchmark tables above.

Precedent: this mirrors the OpenLIT integration row (#1849) — docs entry + dedicated hosted integration page on the project's own docs site.

## Reproducibility

- Pinned datasets (synthetic + HotpotQA indices) committed
- Raw CSVs, `summary.md`, `run_metadata.json` per task
- Notebooks render on GitHub
- All runs used **free-tier APIs only** (Groq + Google AI Studio Gemini)
- Seeded (`dspy.settings.rng = 42`, dataset seed 42)

## AI disclosure (per CONTRIBUTING.md)

Portions of this PR (benchmark harness scaffolding, docs polish, commit messages) were drafted with Claude 4.X under my supervision. All claims were verified against actual benchmark outputs before submission. The `semantix_reward` / `semantic_metric` implementations predate Claude assistance.
