Title: `docs: add semantix-ai to providers/integrations (with benchmark)`

Body:

## Summary

Following up on PR #9583, closed on 2026-04-10 with this feedback:

> "For our providers with integrations, we try to keep it specific to people who have built, measured, and tested specific DSPy features. Are there any DSPy-specific features in Semantix?"

This resubmission brings the measurement. semantix-ai ships DSPy-specific primitives (`semantic_reward`, `semantic_metric`) that plug into `dspy.BestOfN`, `dspy.Refine`, `dspy.Evaluate`, and MIPROv2 — without an LLM-judge API call.

## DSPy-specific features

- `semantix.integrations.dspy.semantic_reward(intent)` returns a `reward_fn(args, pred) -> float` compatible with `BestOfN` / `Refine`.
- `semantic_metric(intent)` returns a `metric(example, pred) -> float` compatible with `Evaluate` and optimizers.

Both are 100% local — no API calls, no keys, ~15 ms per evaluation.

## Measurement

Reproducible benchmark at [github.com/labrat-akhona/semantix-ai/tree/master/benchmarks](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks). Two tasks, four judges, two experiments per task.

### Task A — customer_support_qa (200 examples, 3 intents)

<!-- PASTE headline table from benchmarks/dspy/customer_support/results/summary.md -->

### Task B — hotpotqa_groundedness (200 HotpotQA examples, 1 intent)

<!-- PASTE headline table from benchmarks/dspy/hotpotqa_groundedness/results/summary.md -->

### Verification slice (Gemini 2.5 Flash ↔ Pro)

Flash was used as operational proxy-ground-truth across the full 200 examples; Pro was run on a 25-example slice to validate Flash's rankings. Pearson r (Flash vs. Pro) = <PASTE FROM NOTEBOOK>.

## Full writeup

Dev.to article: <PASTE LINK ONCE PUBLISHED>

## This PR

Adds one row to the providers/integrations table pointing to the above.

## Reproducibility

- Pinned datasets (synthetic + HotpotQA indices) committed to repo
- Raw CSVs, summary markdown, run_metadata.json per task
- Notebooks render on GitHub
- All benchmark runs used **free-tier APIs only** (Groq + Google AI Studio)
