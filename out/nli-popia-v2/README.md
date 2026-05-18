---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: zero-shot-classification
tags:
- nli
- zero-shot-classification
- compliance
- popia
- south-africa
- data-protection
- ai-act
base_model: cross-encoder/nli-MiniLM2-L6-H768
datasets:
- labrat-aiko/popia-training
- labrat-aiko/popia-eval
metrics:
- f1
model-index:
- name: nli-popia-v2
  results:
  - task:
      type: natural-language-inference
      name: NLI on POPIA clauses (v1 holdout, 7 clauses)
    metrics:
    - type: f1
      value: 0.7465
      name: Macro F1 (v1 holdout)
  - task:
      type: natural-language-inference
      name: NLI on POPIA clauses (v2 holdout, 3 new clauses)
    metrics:
    - type: f1
      value: 0.8621
      name: Macro F1 (v2 holdout)
---

# nli-popia-v2

A cross-encoder NLI judge fine-tuned for **South African POPIA compliance reasoning**. Given a premise (a description of a real-world processing scenario) and a hypothesis (a clause-level legal claim), the model predicts `contradiction`, `entailment`, or `neutral`.

Successor to [`labrat-aiko/nli-popia-v1`](https://huggingface.co/labrat-aiko/nli-popia-v1). Broader clause coverage (10 clauses vs 7) at a small in-domain F1 cost on the original holdout.

## What v2 adds over v1

Three new clauses that v1 did not cover, chosen because they are the highest-leverage POPIA sections for **AI / ML workloads** specifically:

| New clause | POPIA section | Why it matters for AI |
|---|---|---|
| **Children's information** | §34-35 | Special protections for under-18s — relevant for EdTech, gaming, paediatric health AI |
| **Special personal information** | §26-33 | Race, religion, health, biometric — every vision model and biometric system touches this |
| **Automated decision-making** | §71 | *The* POPIA clause for AI: requires explanation + right to object on solely-automated decisions |

Combined with v1's seven clauses (consent, minimality, security safeguards, general processing, breach notification, cross-border transfers, data subject rights), v2 covers **10 POPIA clauses** spanning the operative provisions of the Act.

## Evaluation

Two pinned holdouts, neither overlapping with training data.

### v1 holdout — 150 pairs, 7 clauses (original POPIA-Judge v1 territory)

| | Stock cross-encoder | **POPIA-Judge v2** | Delta |
|---|---|---|---|
| Macro F1 | 0.4960 | **0.7465** | **+25.05pp** |

Per-clause F1 on v1 holdout:

| Clause | Stock | v2 | Delta |
|---|---|---|---|
| consent | 0.573 | 0.766 | +0.193 |
| minimality / purpose limitation | 0.529 | 0.611 | +0.083 |
| security safeguards | 0.237 | 0.570 | +0.333 |
| general processing | 0.437 | 0.857 | +0.420 |
| breach notification | 0.335 | 0.712 | +0.376 |
| cross-border transfers | 0.614 | 0.903 | +0.289 |
| data subject rights | 0.476 | 0.809 | +0.333 |

### v2 holdout — 48 pairs, 3 new clauses

| | Stock cross-encoder | **POPIA-Judge v2** | Delta |
|---|---|---|---|
| Macro F1 | 0.3285 | **0.8621** | **+53.36pp** |

Per-clause F1 on v2 holdout:

| Clause | Stock | v2 | Delta |
|---|---|---|---|
| children's information | 0.339 | 0.874 | +0.536 |
| special personal information | 0.365 | 0.717 | +0.352 |
| automated decision-making | 0.259 | 0.850 | +0.591 |

### Honest comparison vs v1

v1 model (`nli-popia-v1`) reported macro F1 0.813 on its 7-clause holdout. v2 model scores **0.7465 on the same holdout** — a ~7pp regression on v1 territory, with the same 22M-parameter base spread across 3 more clauses. If you only need the original 7 clauses, v1 is still the stronger model on that narrow scope. v2 is the right choice when you need the 3 new AI-critical clauses or want a single judge across the full set.

A future v3 with a larger base model (e.g. `nli-deberta-v3-base`) is expected to close this gap.

## Usage

> **Bundled artifacts:** ONNX (fp32 + 4 quantized variants). PyTorch weights will be added in a follow-up release — for now, load via `optimum.onnxruntime` as shown below.

Drop-in via [`semantix-ai`](https://pypi.org/project/semantix-ai/) (0.2.1+):

```python
from semantix.judges import POPIAJudge

judge = POPIAJudge(version="v2")
verdict = judge.evaluate(
    "Our lending AI rejects applicants with a single SMS and no human review.",
    "The responsible party is complying with §71 by offering data subjects "
    "the opportunity to make representations.",
)
# Verdict(passed=False, score=...)
```

Or raw ONNX runtime via `optimum`:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("labrat-aiko/nli-popia-v2")
model = ORTModelForSequenceClassification.from_pretrained(
    "labrat-aiko/nli-popia-v2", file_name="onnx/model.onnx"
)

inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
logits = model(**inputs).logits
# label order: 0=contradiction, 1=entailment, 2=neutral
```

ONNX quantized variants (~25 MB each) are bundled in `onnx/`:

- `model_quint8_avx2.onnx` — broad CPU compatibility
- `model_qint8_avx512.onnx` — modern x86 servers
- `model_qint8_avx512_vnni.onnx` — Intel CPUs with VNNI
- `model_qint8_arm64.onnx` — ARM CPUs (Apple Silicon, AWS Graviton, Ampere)

## Training

- **Base:** `cross-encoder/nli-MiniLM2-L6-H768` (22M params, label order: contradiction=0, entailment=1, neutral=2)
- **Training rows:** 261 (180 from v1 + 81 from v2 — seeds + paraphrases for the new clauses)
- **Epochs:** 6, learning rate 2e-5, batch 16, warmup ratio 0.1, weight decay 0.01
- **Best model:** lowest eval_loss across 6 epochs (load_best_model_at_end)
- **Compute:** Single NVIDIA GTX 1650 (4 GB), CUDA 12.1, ~2 min training time
- **Reproducibility:** training script at [`scripts/train_popia_v2.py`](https://github.com/labrat-akhona/semantix-ai/blob/master/scripts/train_popia_v2.py), eval hashes pinned at `scripts/_popia_eval_v2_hash.txt` and `scripts/_popia_eval_hash.txt`

## Intended use

- **Primary:** verifying that LLM outputs and automated-processing pipelines comply with named POPIA clauses, as part of an audit-grade compliance pipeline (e.g. `semantix-ai`'s `@validate_intent` decorator).
- **Secondary:** standalone clause-level NLI for compliance review tools, internal-audit checklists, and ML systems where regulatory clause text is too long to fit a prompt.

## Limitations

- **English only.** The model is trained on English POPIA-relevant scenarios. South Africa has 11 official languages — multilingual coverage is future work.
- **Single-clause focus.** Composite clauses (e.g., consent AND cross-border) should be evaluated per-leaf — the `semantix` decorator handles this automatically as of v0.2.1.
- **POPIA-specific.** Training scenarios reference South African institutions and statutes. For GDPR, see `GDPRJudge` (sibling model, currently in v0 scaffold).
- **Not legal advice.** Verdicts are statistical entailment estimates, not legal determinations. Treat as one input among many in a compliance review.
- **22M-param base.** A larger base would likely improve in-domain F1. v2 retained the small base for ONNX deployability (~25 MB quantized).

## Bias and fair use

POPIA itself was drafted to *protect* against discriminatory processing of special personal information (§26). The training data deliberately includes scenarios where AI systems would be flagged for inferring race, religion, or health status without lawful basis. The model may therefore *correctly* flag systems that engage in such processing — this is intentional, not a bias to correct.

## License

Apache-2.0 — both code and model weights. Free for commercial use.

## Citation

```bibtex
@misc{eland2026popiajudge_v2,
  author = {Eland, Akhona},
  title = {nli-popia-v2: A POPIA Clause-Level NLI Judge with AI-Focused Clause Coverage},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/labrat-aiko/nli-popia-v2}
}
```

## Sibling artefacts

- [`labrat-aiko/nli-popia-v1`](https://huggingface.co/labrat-aiko/nli-popia-v1) — predecessor, 7-clause coverage, higher F1 on v1 holdout
- [`semantix-ai`](https://pypi.org/project/semantix-ai/) — Python library that uses this judge
- [`labrat-aiko/popia-training`](https://huggingface.co/datasets/labrat-aiko/popia-training) / [`labrat-aiko/popia-eval`](https://huggingface.co/datasets/labrat-aiko/popia-eval) — training & eval datasets
