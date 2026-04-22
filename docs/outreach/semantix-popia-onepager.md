# nli-popia-v1 — One Page

**A locally-hosted, open-source model for validating text against POPIA clauses.**

---

## In one paragraph

`nli-popia-v1` is a small Natural Language Inference (NLI) model fine-tuned to score whether an English text (a signup flow, a breach notification, a customer-support reply) is semantically consistent with seven canonical clauses of South Africa's Protection of Personal Information Act. It runs on an ordinary laptop in ~70 ms per check, makes no network calls, costs nothing per use, and ships under Apache 2.0. It does **not** determine POPIA compliance — that remains a determination only the Information Regulator and a qualified Privacy Officer can make.

## The seven clauses the model scores against

| # | Clause | POPIA anchor |
|---|---|---|
| 1 | Consent | s.11(1)(a) |
| 2 | Minimality / purpose limitation | s.10, s.13 |
| 3 | Security safeguards | s.19 |
| 4 | Breach notification | s.22 |
| 5 | Cross-border transfers | s.72 |
| 6 | General lawful processing | s.9 |
| 7 | Data subject rights | s.23–s.25 |

Each clause is encoded as a single-sentence NLI hypothesis. The model returns a probability that a given text *entails* the hypothesis. A team deploying the model picks a decision threshold (default 0.70).

## The claim the model defends

> *Is this text semantically consistent with clause X at threshold T?*

That is a narrower, technical determination than compliance.

## What the model does not claim

- It does not certify compliance with POPIA.
- It does not replace a Data Protection Impact Assessment.
- It does not replace a Privacy Officer or legal counsel.
- It does not replace the Information Regulator's guidance.

## How it was built

- **Base model:** `cross-encoder/nli-MiniLM2-L6-H768` (a public stock NLI cross-encoder).
- **Training data:** 60 hand-authored seed pairs expanded via LLM paraphrase to 600+ training examples, spanning SaaS, retail, healthcare, fintech, public-sector, and NGO scenarios across South African provinces.
- **Evaluation:** a hash-pinned 150-pair holdout set. The SHA-256 of the eval file is committed to the repo; any change to the eval set must be reviewed as a standalone commit.
- **Measured lift:** macro-F1 of **0.813** on the pinned holdout versus **0.517** for the unmodified stock NLI baseline — a **+29.6 percentage-point** improvement.
- **Release gate:** a GitHub Actions workflow blocks any future release from regressing on any of the seven clauses.

## How it is distributed

- **Model weights:** <https://huggingface.co/labrat-aiko/nli-popia-v1> (Apache 2.0)
- **Training + holdout dataset:** <https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli> (Apache 2.0)
- **Interactive demo:** <https://huggingface.co/spaces/labrat-aiko/popia-judge-demo>
- **Library:** <https://github.com/labrat-akhona/semantix-ai> (MIT)

## Who authored it

Akhona Eland, South African software engineer.
`akhona@automationarchitects.ai` · <https://github.com/labrat-akhona>

## What Akhona is asking the Information Regulator

1. A review of how the intended use is framed in the model card, README, and demo.
2. Feedback on the seven clause hypotheses.
3. Whether the Regulator's office is open to a short working meeting (virtual or at the Pretoria office).

He is not asking for endorsement, certification, or any claim that the software is "POPIA-compliant".
