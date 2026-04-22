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

Fair. I went and built them. This PR is the resubmission, with the measured + tested part done on a small but honest slice.

## What's DSPy-specific in semantix-ai

Two primitives in `semantix.integrations.dspy`, both compatible with your public APIs:

- **`semantic_reward(intent)`** → `reward_fn(args, pred) -> float`, drop-in for `dspy.BestOfN` and `dspy.Refine`.
- **`semantic_metric(intent)`** → `metric(example, pred) -> float`, drop-in for `dspy.Evaluate` and `dspy.MIPROv2`.

Both are powered by a local quantized NLI cross-encoder (~79 MB INT8 ONNX per CPU variant, auto-selected). No API call, no key, deterministic, ~70 ms per evaluation on CPU.

```python
import dspy
from semantix import Intent
from semantix.integrations.dspy import semantic_reward

class Grounded(Intent):
    """The answer must be grounded in the provided context."""

qa = dspy.ChainOfThought("context, question -> answer")
best = dspy.BestOfN(module=qa, N=5, reward_fn=semantic_reward(Grounded))
```

## Measured — customer_support_qa agreement vs Groq Llama 3.3 70B

Reproducible benchmark in [semantix-ai/benchmarks](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks). 50 paired examples, customer-support polite-response classification, scored by both **semantix (local NLI)** and **Groq Llama 3.3 70B** (free tier).

| Metric | Value |
|---|---|
| Pearson r (continuous scores) | **0.596** |
| Cohen's kappa @ 0.5 | 0.487 (moderate agreement) |
| Cohen's kappa @ 0.3 | 0.633 (substantial agreement) |
| Binary agreement @ 0.5 | 76% (38/50) |
| Binary agreement @ 0.3 | 84% (42/50) |

At threshold 0.3–0.4, semantix reaches substantial agreement with Groq Llama 3.3 70B on this task.

| Latency | semantix | groq-llama-3.3-70b |
|---|---|---|
| Mean | 70 ms | 799 ms |
| p50 | 64 ms | 777 ms |
| p95 | 121 ms | 992 ms |
| Paid cost / 1k calls | $0.0000 | $0.1312 |

**~11× lower latency and ~$131/day saved at 1M calls vs Groq's paid tier**, with no cross-border data transfer.

### Honest scope

I originally scoped this to four judges × two tasks (customer-support QA + HotpotQA) and an optimization experiment. What actually shipped:

- ✅ customer-support QA, semantix vs Groq: 50/50 clean pairs (above)
- ⚠️ gemini-2.5-flash: 15/50 dropped to free-tier RPD cap (20 req/day/model)
- ⚠️ gemini-2.5-pro: 25/25 hit the same cap before I could finish the slice
- ⚠️ HotpotQA task and optimization experiment deferred — Gemini free-tier
  quota wasn't enough to score a matched set, and I'd rather ship one honest
  paired comparison than a multi-task table with more holes than data

The raw CSVs are committed with error columns intact if anyone wants to audit
what didn't run.

## Headline

**semantix-ai reaches 0.596 Pearson / substantial Cohen's kappa agreement with Groq Llama 3.3 70B on polite-response classification, at ~11× lower latency and zero API cost.** Suitable for use as a `reward_fn` or `metric` inside `BestOfN` / `Refine` / `Evaluate` / `MIPROv2` when the per-call latency of an LLM-as-judge would dominate the optimization loop.

## Full writeup

Dev.to article: https://dev.to/akhona_eland_072dac9e0c2c/a-70ms-local-nli-judge-hits-0596-pearson-r-with-groq-llama-33-70b-on-dspy-reward-scoring-1d76

## This PR

One-row addition to the providers/integrations table, linking to the DSPy integration docs page that hosts working code and the benchmark above.

Precedent: this mirrors the OpenLIT integration row (#1849) — docs entry + dedicated hosted integration page on the project's own docs site.

## Reproducibility

- Pinned dataset (50 customer-support examples, seeded)
- Raw CSV + `summary.md` with Pearson r / kappa / per-threshold agreement
- Judge implementations and adapter committed (`benchmarks/common/judges.py`)
- All runs used **free-tier APIs only** (Groq + Google AI Studio Gemini)
- Seeded (`dspy.settings.rng = 42`, dataset seed 42)
- semantix scores regenerated against v0.2.0 (label-index bug fix shipped
  2026-04-21); both the pre-fix and post-fix score sets are diffable from
  git history.

## AI disclosure (per CONTRIBUTING.md)

Portions of this PR (benchmark harness scaffolding, docs polish, commit messages) were drafted with Claude 4.X under my supervision. All claims were verified against actual benchmark outputs before submission. The `semantic_reward` / `semantic_metric` implementations predate Claude assistance.
