# customer_support_qa

## Headline comparison (50 examples, agreement experiment)

| Judge | Avg latency (ms) | Paid-tier cost / 1k | Error rate |
|---|---|---|---|
| semantix (QuantizedNLIJudge, v0.2.0) | 70 | $0.0000 | 0/50 |
| groq-llama-3.3-70b-versatile | 799 | $0.1312 | 0/50 |
| gemini-2.5-flash | 1652 | $0.0223 | 15/50 |
| gemini-2.5-pro | n/a | n/a | 25/25 |

Gemini 2.5 Pro free-tier exhausted its 25-request/day quota before completing
the run; gemini-2.5-flash dropped 15/50 to the same RPD ceiling. The head-to-head
numbers below use the 50 examples where **semantix** and **groq-llama-3.3-70b**
both returned a valid score.

## Agreement with Groq Llama 3.3 70B (paired n = 50)

| Metric | Value |
|---|---|
| Pearson r (continuous scores) | **0.596** |
| Cohen's kappa @ 0.3 | 0.633 |
| Cohen's kappa @ 0.4 | 0.633 |
| Cohen's kappa @ 0.5 | 0.487 |
| Cohen's kappa @ 0.7 | 0.421 |
| Binary agreement @ 0.5 | 76% (38/50) |
| Binary agreement @ 0.3 | 84% (42/50) |

At the semantix-default threshold (0.5) the two judges agree on 38/50 examples
(76%) with moderate Cohen's kappa of 0.487. Lowering the threshold to 0.3
pushes agreement to 84% and kappa to 0.633 (substantial agreement). That's the
decision point: **if you want your local judge to behave like Groq Llama 3.3
70B on polite-response classification, run semantix with threshold 0.3–0.4.**

## Latency and cost (paired n = 50)

| Judge | Mean latency | p50 | p95 | Paid cost / 1k calls |
|---|---|---|---|---|
| semantix | 70 ms | 64 ms | 121 ms | $0.0000 |
| groq-llama-3.3-70b | 799 ms | 777 ms | 992 ms | $0.1312 |

**semantix is ~11× faster and $0.13 cheaper per 1k calls.** At 1M calls per
day that's ~$131/day saved vs Groq's paid tier, with no cross-border data
transfer and no rate limit.

## Reproducibility

- Raw per-call data: `results/raw.csv` (1,028 rows, 261 valid rows, 50
  semantix+Groq paired rows in the agreement experiment).
- semantix scores regenerated on 2026-04-22 against v0.2.0 using the same 50
  cached (intent, text) pairs. The v0.1.5–v0.1.13 label-index bug
  (`probs[2]` read as entailment instead of `probs[1]`) made the original
  2026-04-21 semantix scores meaningless; v0.2.0 fixes the bug and the rerun
  is recorded in `raw.csv`.
- Reward-model optimization experiment (gemini-2.5-flash as final judge,
  semantix vs Groq as reward model) was planned but not completed — Gemini
  free-tier RPD cap blocked it. Rows present in `raw.csv` with `error`
  column populated.
