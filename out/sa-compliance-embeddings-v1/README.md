---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: feature-extraction
tags:
- sentence-similarity
- compliance
- popia
- south-africa
- data-protection
- retrieval
- embeddings
base_model: BAAI/bge-small-en-v1.5
datasets:
- labrat-aiko/popia-training
metrics:
- recall
model-index:
- name: sa-compliance-embeddings-v1
  results:
  - task:
      type: information-retrieval
      name: POPIA clause retrieval (scenario -> POPIA section)
    metrics:
    - type: recall_at_1
      value: 0.4766
      name: Recall@1
    - type: recall_at_5
      value: 0.6563
      name: Recall@5
---

# sa-compliance-embeddings-v1

A 384-dimensional sentence embedding model fine-tuned for **South African compliance retrieval**, especially POPIA (Protection of Personal Information Act, 2013). Given a compliance scenario or query, this model retrieves the most relevant POPIA section text — a task where general-purpose embeddings under-perform because they are not trained on South African regulatory language or POPIA's specific structure.

**This is, to our knowledge, the first publicly distributed embedding model fine-tuned on POPIA-grounded data.**

## Why this exists

General embedding models (`intfloat/e5-small-v2`, `BAAI/bge-small-en-v1.5`) are trained on web-scale English. They have no special grounding in South African regulatory language, no exposure to POPIA's section structure, and no concept of which scenarios trigger which clauses. The base model only retrieves the correct POPIA section on the first try for ~21% of compliance queries; this fine-tune raises that to ~48%.

For practical use — building "show me the POPIA clauses relevant to this scenario" tools, retrieval-augmented compliance reviews, or audit-pipeline section lookups — that gap is the difference between useful and noise.

## Evaluation

Retrieval task: given a labelled compliance scenario (entailment or contradiction relative to a known POPIA clause), retrieve the canonical POPIA section text for that clause from a corpus of 114 POPIA sections.

| Metric | Stock `bge-small-en-v1.5` | **sa-compliance-embeddings-v1** | Delta |
|---|---|---|---|
| Recall@1 | 0.211 | **0.477** | **+26.6pp** |
| Recall@3 | 0.445 | **0.594** | **+14.9pp** |
| Recall@5 | 0.508 | **0.656** | **+14.8pp** |
| Recall@10 | 0.680 | **0.766** | **+8.6pp** |

Eval set: 128 entailment + contradiction scenarios from `data/popia_eval.jsonl` and `data/popia_eval_v2.jsonl` — held out from training.

## Usage

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("labrat-aiko/sa-compliance-embeddings-v1")
model = AutoModel.from_pretrained("labrat-aiko/sa-compliance-embeddings-v1")

def embed(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)

queries = ["Our app collects under-13 learner data without parental consent."]
docs = [
    "POPIA §34. Prohibition on processing personal information of children...",
    "POPIA §22. Notification of security compromises...",
]
q_emb = embed(queries)
d_emb = embed(docs)
scores = q_emb @ d_emb.T  # higher = more relevant
```

## Training

- **Base:** `BAAI/bge-small-en-v1.5` (33M params, 384-dim, mean pooling)
- **Loss:** MultipleNegativesRankingLoss (InfoNCE with in-batch negatives, temperature 1/20)
- **Training data:** 308 (anchor, positive) pairs built from:
  - The full POPIA Act text (114 sections, extracted from the official PDF)
  - The labelled scenarios in `data/popia_seeds*.jsonl` and `data/popia_paraphrases*.jsonl`
  - The clause hypotheses in those same files
- **Schedule:** 6 epochs, AdamW lr 2e-5, linear warmup over 10% of steps, batch 16
- **Compute:** NVIDIA GTX 1650 (4 GB), ~45 seconds total

## Intended use

- **Primary:** retrieving the POPIA section text relevant to a given compliance scenario or query, e.g. for RAG pipelines that need to reason about POPIA.
- **Secondary:** clustering / similarity over SA-compliance documents, dataset deduplication, weak-supervision labelling for downstream classifiers.

## Limitations

- **English only** — POPIA materials are predominantly English. Multilingual coverage of South Africa's other 10 official languages is future work.
- **POPIA-focused corpus** — training data is concentrated on POPIA Act text. Coverage of FSCA AI guidance, SARB circulars, Treasury directives, and Information Regulator media statements is planned for v2.
- **Small base model** — 33M parameters. A larger base (`bge-base-en-v1.5` ~110M) would likely give a few more points of recall but is slower and less deployable.
- **Not a legal index** — verdict bias still requires human review. Use this for routing and recall, not as the final word on which clause applies.

## Roadmap

- **v2 corpus expansion:** add FSCA AI report (Nov 2025), SARB circulars, IR media statements, and SA Treasury procurement guidance to the training pairs.
- **GDPR sibling:** the same recipe applied to GDPR articles + ECJ decisions, scheduled before EU AI Act Art. 50 binds (2 Aug 2026).
- **Bench:** the SA Compliance Retrieval Bench, released alongside POPIA-Bench v1.

## License

Apache-2.0 — both code and model weights. Free for commercial use.

## Citation

```bibtex
@misc{eland2026sacompliance,
  author = {Eland, Akhona},
  title = {sa-compliance-embeddings-v1: A POPIA-Grounded Sentence Embedding Model for South African Compliance Retrieval},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/labrat-aiko/sa-compliance-embeddings-v1}
}
```

## Sibling artefacts

- [`labrat-aiko/nli-popia-v2`](https://huggingface.co/labrat-aiko/nli-popia-v2) — clause-level NLI judge over the same 10 POPIA clauses
- [`labrat-aiko/nli-popia-v1`](https://huggingface.co/labrat-aiko/nli-popia-v1) — original 7-clause NLI judge
- [`semantix-ai`](https://pypi.org/project/semantix-ai/) — Python library for output-side compliance validation
