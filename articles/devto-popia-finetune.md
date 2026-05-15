<!--
Dev.to frontmatter (for API publish via POST /api/articles — optional, ignored by paste-into-editor flow):

---
title: "I Fine-Tuned a Compliance Judge and Beat the Stock Model by +29.6pp F1"
published: false
description: "An open-source cross-encoder NLI judge for POPIA clauses, quantized to INT8 ONNX, with a hash-pinned eval set and a CI release gate. 0.517 → 0.813 macro F1 on a 150-pair holdout. Reproducible recipe in ~6 minutes of CPU."
tags: machinelearning, python, opensource, nlp
canonical_url:
cover_image:
---
-->

# I Fine-Tuned a Compliance Judge and Beat the Stock Model by +29.6pp F1

**The problem:** if your LLM-powered product touches personal information in South Africa, POPIA sits over it. The regulator doesn't ask "is your model good?" — they ask "can you demonstrate the output was validated against the clause, and can you show me the validation?"

**The uncomfortable answer most teams give today:** "we call GPT-4 as a judge with a prompt that mentions POPIA." That's not a defence. It's non-deterministic, sends personal information cross-border, and produces no receipt.

**What I built instead:** a local NLI cross-encoder fine-tuned on 7 POPIA clauses, released under Apache 2.0, shipped as a quantized ONNX model, scored and gated on every CI run.

The result, on a pinned 150-pair holdout:

| | Stock `cross-encoder/nli-MiniLM2-L6-H768` | Fine-tuned `nli-popia-v1` |
|---|---|---|
| Macro F1 | 0.517 | **0.813** |
| Accuracy | 0.707 | 0.833 |
| Worst clause | 0.400 (general processing / data subject rights) | 0.727 (cross-border transfers) |
| Best per-clause lift | — | **+0.493** (general processing) |
| Regressions | — | zero |

**+29.6 percentage points macro F1, every clause improved, nothing got worse.** 79MB per CPU-variant INT8 ONNX on disk, ~15ms per inference on CPU, zero API calls.

Here's how it went.

---

## Why NLI, not a prompt-based judge

Natural Language Inference is an old, narrow, boring task: given a premise and a hypothesis, return the probability the premise entails the hypothesis. Cross-encoders have been doing this deterministically for a decade.

If you reframe "does this text satisfy POPIA's consent clause?" as an NLI problem:

- **Premise:** the LLM's output
- **Hypothesis:** "The text collects personal information only after obtaining explicit, informed, opt-in consent."

…you get a deterministic score in 0.0–1.0, in one tiny ONNX model, without shipping customer data to a third-party API.

The catch: stock NLI models are trained on SNLI/MNLI. They're great at "a dog is playing in the park / an animal is outside" and terrible at "This message confirms your purchase; we'll process your data per our privacy policy / The text obtains explicit opt-in consent before collecting personal information."

Stock macro F1 on POPIA clauses: **0.517.** Two of the seven clauses — general processing and data subject rights — came in at **0.400 F1**. Coin-flip territory.

So I fine-tuned.

---

## The data: 180 hand-authored pairs, no scraping

This is the part nobody wants to hear: I wrote the training data by hand.

Seven clauses — consent, minimality, security safeguards, breach notification, cross-border transfers, general processing, data subject rights — × a handful of positive examples (text that satisfies the clause) + a handful of negatives (text that violates it) + paraphrases. About 180 pairs.

Why hand-authored:

- **Scraped legal text is the wrong distribution.** My users aren't writing statutes; they're writing support replies, KYC confirmations, breach emails. I needed LLM-shaped text, not Act-shaped text.
- **Synthetic generation would poison the eval.** If GPT-4 writes my training data and GPT-4 writes the outputs being validated in production, I'm measuring GPT-4's self-consistency, not POPIA compliance.
- **180 pairs is enough for 7-clause cross-encoder fine-tuning.** The base model already speaks English; I'm teaching it a narrow decision boundary, not a new language.

The 150-pair holdout was hand-authored separately, pinned by hash, and never leaks into training. If the hash of the eval file changes, the release gate fails.

---

## The fine-tune: 5 epochs, six minutes on CPU

The whole training recipe:

```bash
pip install "semantix-ai[train]"
python scripts/train_popia.py
```

Under the hood it's unremarkable:

- Base: `cross-encoder/nli-MiniLM2-L6-H768` (~22M params, tiny)
- 5 epochs, batch 16, lr 2e-5, warmup 10%, weight decay 0.01
- Cross-entropy loss, early stopping on `eval_loss` against a 10% dev split
- CPU training on 180 rows: **~6 minutes**
- ONNX export with four CPU-variant INT8 quantizations (AVX2 / AVX512 / AVX512-VNNI / ARM64), auto-selected at load time based on CPU detection

Each quantized variant is ~79MB; consumers only download the one their CPU needs. Inference is zero-PyTorch — `onnxruntime` + `tokenizers`, nothing else.

---

## The release gate: CI fails if the next fine-tune regresses

This is the part I think more ML projects should steal.

```yaml
# .github/workflows/popia-eval.yml (abridged)
- name: Run release gate
  run: |
    python -m semantix.cli eval popia --json | tee report.json
    python -c "import json; r=json.load(open('report.json'));
               import sys; sys.exit(0 if r['release_gate_passed'] else 1)"
```

The gate logic is boring and strict:

```python
release_gate_passed = (
    (finetune_macro_f1 - stock_macro_f1) >= 0.10
    and no_per_clause_regression_vs_stock
)
```

Any future fine-tune that drops below a +10pp macro-F1 lift, OR regresses a single clause vs stock, fails CI and blocks the release. The model artifact has the same quality gate as the code that loads it.

---

## Using it

```bash
pip install "semantix-ai[popia]"
```

```python
from semantix import validate_intent
from semantix.presets.popia import POPIA_CONSENT, POPIA_SECURITY

@validate_intent(POPIA_CONSENT)
def compose_signup_confirmation(name: str) -> str:
    # Your LLM call here. If the output doesn't satisfy POPIA_CONSENT,
    # the decorator retries with structured feedback. If it still fails,
    # it raises — with a Semantic Certificate in the audit trail.
    ...
```

On first import, the quantized ONNX model downloads once from HuggingFace and caches locally. No HF token required — the model is public.

Seven presets ship with the library, one per clause. Each has a pre-tuned threshold based on the per-clause F1 on the holdout. You can override any threshold; the defaults are the F1-optimal operating points.

---

## What I will and won't claim

**I will claim:** on a pinned 150-pair hand-authored POPIA holdout, the fine-tune beats the stock MiniLM2 NLI cross-encoder by +29.6pp macro F1, every clause improves, no regressions. That result is reproducible — the eval set is hashed, the CI gate enforces it.

**I won't claim:** this model replaces a POPIA specialist, a DPIA, or the Information Regulator's guidance. It's a deterministic, local, auditable primitive you can wire into your validation pipeline. It tells you whether a specific output is consistent with a specific POPIA clause at a specific threshold. That's a narrower claim than "POPIA-compliant" and it's the only claim I can actually defend with a holdout F1 number.

**I especially won't claim:** 180 pairs is enough training data for every production use case. If your domain has dialect, local legal phrasing, or adversarial customers trying to slip past the guard, you should fine-tune on *your* failures. The repo includes the training recipe for exactly that reason.

---

## The reusable part

The thing I'm most interested in is that the entire recipe — hand-authored seeds + paraphrases + cross-encoder fine-tune + ONNX export + release gate — is regulation-agnostic. Swap POPIA for GDPR and you get `nli-gdpr-v1`. Swap for HIPAA and you get `nli-hipaa-v1`. Swap for EU AI Act clause libraries and you get a judge per article.

v0.2.0 already ships a **GDPR sibling-model scaffold** — same `Judge` interface, 7 EU-clause presets, expansion seeds, training script, and a documented runtime fallback to POPIA weights until the GDPR artifact trains. It is deliberately a scaffold: same API surface, same CI gate pattern, no weights pretending to exist. That is the contract. The second regulator costs less than the first.

`nli-popia-v1` is the first trained artifact. It's 0.813 macro F1 and it's live:

- **PyPI:** https://pypi.org/project/semantix-ai/0.2.0/
- **GitHub release:** https://github.com/labrat-akhona/semantix-ai/releases/tag/v0.2.0
- **Model card:** https://huggingface.co/labrat-aiko/nli-popia-v1

---

## If you're building on this

The failure modes to watch for:

1. **Threshold tuning matters more than you'd think.** The per-clause F1-optimal thresholds in the presets are tuned on my holdout, not yours. If your domain's distribution is different, re-tune.
2. **False negatives on ambiguous consent language.** "By continuing, you agree to…" is legally grey, and the model reflects that. Tighten the threshold if you want the library to err on the side of rejecting.
3. **This is a classifier, not a reasoner.** It doesn't explain *why* a clause failed. Pair it with `semantix.judges.ForensicJudge` (ships with `[turbo]`) if you need a mask-perturbation saliency breach report.

If you ship something interesting with it, or fine-tune a sibling (GDPR, HIPAA, UK DPA), I'd love to see it. Issues and PRs welcome on the repo.

---

*semantix-ai is an MIT-licensed semantic type system for AI outputs. `v0.2.0` is the first release with compliance-specific fine-tunes and ships both the trained POPIA artifact and a GDPR sibling-model scaffold. The POPIA model weights are Apache 2.0. Everything here was built by one person; numbers are reproducible, judgement calls are mine.*
