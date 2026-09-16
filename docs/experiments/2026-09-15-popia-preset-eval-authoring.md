# Authoring `data/popia_preset_eval.jsonl` — 70 preset premises

This file decides whether each POPIA preset can be called validated (roadmap item E3). For each
slot you write two fields, `scenario` and `premise`. `preset`, `clause`, `expected`, and `style` are
already set and balanced; don't change them.

## The loop

```bash
python scripts/validate_popia_preset_eval.py          # progress bars and checks
python scripts/validate_popia_preset_eval.py --pin    # only when 70/70 and clean
```

Commit `scripts/_popia_preset_eval_hash.txt` on its own once it is pinned.

## What the labels mean

Labels describe the premise under the POPIA clause, never whether the preset's wording "matches".
Presets point in different directions (`POPIA_CROSS_BORDER` describes a violation;
`POPIA_BREACH` is negated), and the gate handles each preset's direction itself.

| `expected` | Meaning | Cross-border example |
|---|---|---|
| `complies` | The premise shows the clause being met | Records go to an EU processor under a contract with POPIA-equivalent safeguards, and customers consented |
| `violates` | The premise shows the clause being broken | Nightly replication to a US cluster with no agreement, consent, or other lawful basis |
| `insufficient` | On topic, but not enough to decide | Records are backed up to a US data centre; nothing is said about safeguards or consent |

## Styles

- `memo`: an internal policy, process, or incident note.
- `news`: a third-person report about an organisation.
- `first-person`: what a support agent or an LLM would write to a customer.

## One finished row

```json
{"preset": "POPIA_CROSS_BORDER", "clause": "POPIA cross-border transfers", "expected": "insufficient", "style": "news", "scenario": "retailer US backups", "premise": "A Johannesburg clothing retailer confirmed that customer loyalty records are now backed up nightly to a data centre in Virginia, USA. The company did not say what agreements govern the transfer."}
```

## Traps the validator catches

- A premise copied from any `data/popia_*.jsonl` file or POPIA-Bench (leak).
- A duplicate premise or scenario tag.
- A changed label or style (the per-preset split breaks).
- A premise under 60 characters (warning only).

## Traps it can't catch

- One-liners. The three preset examples that failed in the 2026-09-15 review were single
  sentences; give who, what, and when.
- Restating the preset's description as the premise.
- `insufficient` premises that are merely short rather than genuinely undecidable.
- Unrealistic settings: use plausible South African organisations and situations, as
  `data/popia_eval.jsonl` does.
