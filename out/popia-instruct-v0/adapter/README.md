---
license: mit
language:
- en
library_name: peft
pipeline_tag: text-generation
tags:
- popia
- compliance
- south-africa
- data-protection
- qlora
- lora
- peft
- base_model:adapter:microsoft/Phi-3-mini-4k-instruct
base_model: microsoft/Phi-3-mini-4k-instruct
datasets:
- labrat-aiko/popia-training
---

# popia-instruct-v0

A **LoRA adapter** for [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), fine-tuned for **South African POPIA compliance Q&A**. Given a compliance scenario or a question about POPIA, the adapted model answers in a consistent template grounded in actual POPIA Act section text.

This is the third leg of the **SA AI Compliance Stack**, alongside [`nli-popia-v2`](https://huggingface.co/labrat-aiko/nli-popia-v2) (clause-level NLI judge) and [`sa-compliance-embeddings-v1`](https://huggingface.co/labrat-aiko/sa-compliance-embeddings-v1) (POPIA retrieval).

## What this is (and isn't)

**v0. Narrow but real.** Trained on 805 instruction examples derived from the POPIA Act text and labelled scenarios. It pattern-matches scenarios → relevant POPIA clauses, recites section text faithfully, and answers compliance-review prompts in a consistent template.

**Not a compliance lawyer.** Section-number recall is hit-or-miss for clauses with sparse training coverage. It does not do multi-hop reasoning across statutes, does not know case law, and does not have an opinion on novel edge cases outside training distribution.

Treat as an MVP for grounded POPIA generation. The right benchmark is *faithfulness to the literal Act text*, not free-form legal reasoning — where it should comfortably beat frontier models that have no POPIA-specific weight updates.

## Training

- **Base:** `microsoft/Phi-3-mini-4k-instruct` (3.8B params, MIT license)
- **Adapter:** LoRA, r=16, α=32, dropout 0.05, targets `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` — 25.2M trainable params (0.65% of base)
- **Quantization:** 4-bit NF4 (bitsandbytes) with `bf16` compute dtype
- **Optimizer:** `paged_adamw_8bit`, lr 2e-4, cosine schedule, 5% warmup, weight decay 0
- **Schedule:** 2 epochs, batch 1, gradient accumulation 8, effective batch 8
- **Data:** 805 (system + user + assistant) instruction examples derived from:
  - 114 POPIA Act sections (canonical "Explain POPIA §X" and "What does Section X govern?" pairs)
  - 261 hand-authored compliance scenarios (entailment / contradiction / neutral)
  - 192 routing examples ("What POPIA rule is implicated by this scenario?")
  - 10 clause → section listings
- **Compute:** Single NVIDIA GTX 1650 (4 GB VRAM, Turing, CUDA 12.1)
- **Wall time:** ~85 min training + ~5 min model download + setup
- **Loss curve:** train 1.57 → 0.26 over 190 effective steps; eval 0.38 → 0.33 between epochs (clean descent, no overfitting visible)

Training script: [`scripts/train_popia_instruct.py`](https://github.com/labrat-akhona/semantix-ai/blob/master/scripts/train_popia_instruct.py).
Dataset builder: [`scripts/build_popia_instruct_dataset.py`](https://github.com/labrat-akhona/semantix-ai/blob/master/scripts/build_popia_instruct_dataset.py). Every instruction-response pair traces deterministically to a POPIA section text or a labelled scenario — no LLM-generated content in the training set.

## Usage

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_id = "labrat-aiko/popia-instruct-v0"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": "You are a careful South African data-protection assistant. Answer in clear, professional English with reference to the Protection of Personal Information Act, 2013 (POPIA). When citing sections, use the form 'POPIA §X'."},
    {"role": "user", "content": "What POPIA rule is implicated by this scenario? Our after-school maths app lets any 9-year-old sign up with just an email address."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Qualitative samples (from training-time eval)

Three held-out scenarios, deterministic greedy decoding:

**1. Clause routing — exact match with gold:**
> *User:* What POPIA rule is implicated by this scenario? Sign-up for our homework helper happens on the child's own phone with no parent in the loop.
> *Gold:* This scenario implicates POPIA §34 and POPIA §35 — the children's information provisions.
> *Pred:* This scenario implicates POPIA §34 and POPIA §35 — the children's information provisions.

**2. Section listing — partial:**
> *User:* Which POPIA sections govern data subject rights?
> *Gold:* The data subject rights concern is anchored in POPIA §5, §23, §24, §25.
> *Pred:* The data subject rights concern is anchored in POPIA §5 and POPIA §23.

**3. Section recall failure (clause label still correct):**
> *User:* What POPIA rule is implicated by this scenario? NUMSA's member-services team uses the union's own membership records to administer benefits.
> *Gold:* This scenario implicates POPIA §26 and POPIA §27 — the special personal information provisions.
> *Pred:* This scenario implicates POPIA §10 and POPIA §13 — the special personal information provisions.

This is the v0 reality: clause-name routing and template adherence are solid; section-number recall is partial. v1 would benefit from 10× more training data, a 7B-class base, and case-law / Information Regulator decisions in the corpus.

## Limitations

- **English only.** Multilingual coverage for SA's other 10 official languages is future work.
- **POPIA-specific.** No GDPR, no HIPAA, no EU AI Act knowledge except by coincidental base-model overlap.
- **No case law / no IR decisions.** Trained on the Act text + synthetic scenarios. Real-world disputes hinge on enforcement patterns that aren't in the training data.
- **Section recall partial.** As shown in sample 3 above. Confidence in section numbers should be treated as a hint, not a citation.
- **Not legal advice.** As with [`nli-popia-v2`](https://huggingface.co/labrat-aiko/nli-popia-v2), this model is a research / decision-support tool, not a substitute for a compliance review.
- **Small base.** 3.8B params is small. Quantitative reasoning, multi-paragraph drafting, and long-context tasks will be base-Phi-3 quality at best.

## License

LoRA adapter weights: **MIT** (inherits the base model's licence).
Training data and recipe: released under Apache-2.0 in the `semantix-ai` repository.

## Citation

```bibtex
@misc{eland2026popiainstruct,
  author = {Eland, Akhona},
  title = {popia-instruct-v0: A {QLoRA}-Fine-Tuned Phi-3-mini Adapter for South African Data Protection Q\&A},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/labrat-aiko/popia-instruct-v0}
}
```

## Sibling artefacts

- [`labrat-aiko/nli-popia-v2`](https://huggingface.co/labrat-aiko/nli-popia-v2) — clause-level NLI judge over the same 10 POPIA clauses (verification, not generation)
- [`labrat-aiko/sa-compliance-embeddings-v1`](https://huggingface.co/labrat-aiko/sa-compliance-embeddings-v1) — POPIA retrieval embedding model
- [`semantix-ai`](https://pypi.org/project/semantix-ai/) — Python library that uses the judge in production
- [POPIA-Bench v1](https://github.com/labrat-akhona/semantix-ai/tree/master/bench/popia-v1) — public benchmark for clause-level POPIA NLI
