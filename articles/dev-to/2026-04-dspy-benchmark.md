---
title: "A 70ms Local NLI Judge Hits 0.596 Pearson r With Groq Llama 3.3 70B on DSPy Reward Scoring"
description: "One paired comparison, 50 customer-support examples, semantix-ai vs Groq Llama 3.3 70B as a DSPy reward_fn. No cherry-picking, no extra tasks, no filled-in holes."
tags: dspy, llm, python, benchmarking
published: false
---

## TL;DR

- **`semantic_reward` is a drop-in DSPy reward function powered by a local quantized NLI cross-encoder** — no API call, no key, deterministic, ~70ms per evaluation on CPU.
- On **50 paired customer-support examples**, semantix reaches **Pearson r = 0.596** with Groq Llama 3.3 70B, and **Cohen's kappa 0.633 at threshold 0.3** (substantial agreement), at **~11× lower latency and $0.13 cheaper per 1k calls**.
- Full reproducibility: code, dataset, raw CSVs at [github.com/labrat-akhona/semantix-ai/benchmarks](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks).

## Why another reward function?

DSPy's `BestOfN` and `Refine` lean on a `reward_fn` that scores each candidate from 0 to 1. In practice most users wire up another LLM call — cheap per-request but adds 300–1000 ms and a few cents per optimization run. If you're iterating, that adds up fast.

semantix-ai ships a ~79 MB INT8 quantized NLI cross-encoder (one of four CPU-specific variants, auto-selected based on your hardware) that scores "does text X entail intent Y?" in ~70ms on CPU. Plugging it into DSPy takes one line:

```python
import dspy
from semantix import Intent
from semantix.integrations.dspy import semantic_reward

class Grounded(Intent):
    """The answer must be grounded in the provided context."""

qa = dspy.ChainOfThought("context, question -> answer")
refined = dspy.BestOfN(module=qa, N=5, reward_fn=semantic_reward(Grounded))
```

## The honest scope

I originally set out to benchmark four judges across two tasks with an optimization experiment. Reality:

- ✅ **customer_support_qa, semantix vs Groq Llama 3.3 70B: 50/50 paired scores, clean.** That's this post.
- ⚠️ Gemini 2.5 Flash: 15/50 hit the free-tier 20-requests-per-day-per-model cap mid-run.
- ⚠️ Gemini 2.5 Pro: 25/25 hit the same cap.
- ⚠️ HotpotQA task and `BestOfN` optimization experiment deferred — without Gemini as the final judge I couldn't close the loop, and I'd rather ship one clean pair than a multi-task table with holes.

The raw CSV is committed with error columns intact. Everything you're about to see is reproducible from the 50 rows both judges agreed to complete.

## Setup

- **Dataset:** 50 customer-support response candidates paired with one of ~10 intents ("The response must be polite and professional", "The response must stay on topic", "The agent must decline without being rude", etc.). Seeded generation.
- **semantix**: `QuantizedNLIJudge` from v0.2.0. Auto-detected CPU variant, INT8 ONNX, `onnxruntime` only.
- **Groq**: `groq-llama-3.3-70b-versatile`, free-tier API, temperature 0.
- **Scoring protocol**: Both judges return a continuous 0–1 score. `passed` is derived at threshold.

## Agreement results (paired n = 50)

| Metric | Value |
|---|---|
| Pearson r (continuous scores) | **0.596** |
| Cohen's kappa @ 0.3 | 0.633 |
| Cohen's kappa @ 0.4 | 0.633 |
| Cohen's kappa @ 0.5 | 0.487 |
| Cohen's kappa @ 0.7 | 0.421 |
| Binary agreement @ 0.5 | 76% (38/50) |
| Binary agreement @ 0.3 | 84% (42/50) |

Pearson r = 0.596 is a **moderate positive correlation** between the two judges on raw scores. The binary pass/fail story is more interesting: at the semantix-default threshold 0.5 the two agree on 76% of calls (moderate kappa of 0.487). Drop the threshold to 0.3 and they agree on 84% of calls at **substantial kappa 0.633**.

The actionable knob: **if you want semantix to track Groq Llama 3.3 70B's polite-response classification, run it with threshold 0.3–0.4.** The default 0.5 is tuned against strict NLI datasets; for pragmatic customer-support scoring, a slightly looser threshold is closer to what a 70B LLM-judge would mark as "polite enough".

## Latency and cost

| | semantix | groq-llama-3.3-70b |
|---|---|---|
| Mean latency | 70 ms | 799 ms |
| p50 | 64 ms | 777 ms |
| p95 | 121 ms | 992 ms |
| Paid cost / 1k calls | $0.0000 | $0.1312 |

**~11× lower latency.** On a paid Groq plan, 1M calls per day would cost ~$131/day in Groq API fees alone; semantix adds $0 and never leaves your machine. For a DSPy optimization loop calling the reward function hundreds of times per trial, the difference compounds into hours saved.

## What this means in practice

- **Use semantix as your `reward_fn` in `BestOfN` and `Refine`** when per-call latency of an LLM-as-judge would dominate your optimization loop. At substantial kappa with Groq on polite classification, it's a reasonable signal with two orders-of-magnitude better cost structure.
- **Tune the threshold** against your own held-out examples. The default 0.5 is too strict for conversational-tone tasks; 0.3–0.4 tracks a 70B LLM-judge more faithfully on this task.
- **Don't use it as a reasoner.** It's a narrow entailment classifier. If your task needs "why is this wrong?", pair it with `ForensicJudge` (mask-perturbation saliency) or keep the LLM for final scoring.

## A footnote on the bug that almost killed this post

The original benchmark run on 2026-04-21 showed **Pearson r = -0.594** — a strongly *negative* correlation. I almost shipped that as "semantix disagrees with Groq, caveat emptor". Digging in, I found a label-ordering bug in `QuantizedNLIJudge` (shipped in v0.1.5, fixed in v0.2.0): the code was reading `probs[2]` (neutral) as the entailment score instead of `probs[1]`. Fixing the bug and re-running the 50 cached texts against v0.2.0 flipped the correlation sign and shifted the kappa from near-zero to substantial.

The raw CSV preserves both runs' scores through git history if anyone wants to see the before/after. I'm noting this here because (a) it's a useful cautionary tale about trusting your benchmark when the numbers look too surprising, and (b) it's the exact kind of thing a release gate (like v0.2.0's POPIA macro-F1 gate) is supposed to catch, which it now does.

## Reproducing

```bash
git clone https://github.com/labrat-akhona/semantix-ai
cd semantix-ai
pip install -e ".[turbo]"  # zero-PyTorch install
pip install -r benchmarks/requirements.txt
cp .env.example .env  # add GROQ_API_KEY
python -m benchmarks.dspy.customer_support.run
```

Results land in `benchmarks/dspy/customer_support/results/` (`raw.csv`, `summary.md`).

## What's next

Same minimal-first methodology will be applied to [outlines](https://github.com/dottxt-ai/outlines), [marvin](https://github.com/PrefectHQ/marvin), and [llama_index](https://github.com/run-llama/llama_index) — one paired comparison, no holes, real numbers. Open PR at stanfordnlp/dspy referencing this work: [link TBD once PR is open].

---

*semantix-ai is MIT-licensed. PyPI: [pypi.org/project/semantix-ai](https://pypi.org/project/semantix-ai/). v0.2.0 also ships a POPIA-compliance fine-tune reaching 0.813 macro-F1 on a pinned holdout.*
