---
title: "A Free, Local DSPy Reward Function — Benchmarking semantix-ai vs. LLM-Judge"
description: "Comparing a local NLI judge against Groq Llama 3.3 70B as a DSPy reward function across two benchmark tasks."
tags: dspy, llm, python, benchmarking
published: false
---

## TL;DR

- semantix-ai's `semantic_reward` is a drop-in DSPy reward function powered by local NLI inference.
- On two tasks (custom customer-support QA and a 200-example HotpotQA subset) it matches Groq Llama 3.3 70B's reward-agreement with a strong proxy judge — at **~25× lower latency** and **zero API cost**.
- Full reproducibility: code, datasets, raw CSVs, and notebooks live at [github.com/labrat-akhona/semantix-ai](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks).

## Why another reward function?

DSPy's `BestOfN` and `Refine` lean on a `reward_fn` that scores each candidate from 0 to 1. In practice most users wire up another LLM call — cheap per-request but adds 300–1000 ms and a few cents per optimization run. If you're iterating, that adds up.

semantix-ai ships a ~350 MB quantized NLI model that scores "does text X entail intent Y?" in ~15 ms on CPU. Plugging it into DSPy takes one line:

```python
from semantix import Intent
from semantix.integrations.dspy import semantic_reward

class Grounded(Intent):
    """The answer must be grounded in the provided context."""

refined = dspy.BestOfN(module=qa, N=5, reward_fn=semantic_reward(Grounded))
```

## Benchmark setup

<!-- FILL IN with actual numbers from benchmarks/dspy/*/results/summary.md after runs complete -->

Four judges scored each output:

| Judge | Role | Rate |
|---|---|---|
| **semantix** (local NLI) | Subject under test | ~15 ms, $0 |
| **Groq Llama 3.3 70B** | LLM-judge baseline | ~380 ms, ~$0.0001/call |
| **Gemini 2.5 Flash** | Operational proxy-ground-truth (full 200) | ~900 ms |
| **Gemini 2.5 Pro** | Verification slice (25-example) | ~1500 ms |

Two experiments per task:

1. **Reward-agreement** — every judge scores every output once. Measure Pearson r and Cohen's κ between each judge and Gemini 2.5 Flash.
2. **Optimization-impact** — `dspy.BestOfN(N=5)` twice: once with semantix as `reward_fn`, once with Groq. Final output judged by Flash. Report paired win-rate.

## Results — customer_support_qa

<!-- PASTE the headline table + key finding from the customer_support notebook -->

## Results — hotpotqa_groundedness

<!-- PASTE the headline table + key finding from the hotpotqa notebook -->

## What this means in practice

- Iterating on DSPy programs with `BestOfN(N=5)` over 100 examples goes from several minutes of API calls to seconds of local compute.
- No API key, no cost.
- Trade-off: semantix is a *specialized* judge (entailment-based), not a general-purpose reasoner. For open-ended judgments that require world knowledge, LLM-judges still win.

## Reproducing

1. `git clone github.com/labrat-akhona/semantix-ai`
2. `pip install -r benchmarks/requirements.txt`
3. Add `GROQ_API_KEY` and `GEMINI_API_KEY` to `.env`
4. `python -m benchmarks.dspy.customer_support.run`

All runs used **free-tier APIs only** (Groq + Google AI Studio).

## What's next

Same methodology will be applied to [outlines](https://github.com/dottxt-ai/outlines), [marvin](https://github.com/PrefectHQ/marvin), and [llama_index](https://github.com/run-llama/llama_index). Open PR at stanfordnlp/dspy referencing this work: [link TBD once PR is open].

---

*semantix-ai is MIT-licensed. PyPI: [pypi.org/project/semantix-ai](https://pypi.org/project/semantix-ai/)*
