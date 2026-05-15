# POPIA-Bench v1

A 197-pair held-out benchmark for **clause-level POPIA compliance NLI**. Each example consists of a real-world processing scenario (premise), a clause-grounded hypothesis, and a gold label (`contradiction` / `entailment` / `neutral`) — across the ten operative clause groups of South Africa's Protection of Personal Information Act, 2013.

This is the eval set used as the release gate for `nli-popia-v1` and `nli-popia-v2`. We release it publicly so that other compliance-AI artifacts (open-source or commercial) can be measured against the same yardstick.

## Why this exists

There is no shared, public, English-language benchmark for clause-level compliance reasoning over a national data-protection statute. Hazard-taxonomy benchmarks (HarmBench, etc.) measure different things. RAG-faithfulness benchmarks measure grounding, not statutory entailment. Contract-NLI is a related corpus but covers commercial contracts, not regulator clauses.

If you ship a guardrail product that claims POPIA coverage, you should be able to publish a number on POPIA-Bench. If you can't, that's information for buyers.

## Files

| File | Purpose |
|---|---|
| `popia_bench.jsonl` | The benchmark — 197 examples, one JSON object per line |
| `popia_bench.sha256` | Pinned SHA-256 of the bench file (immutable) |
| `score.py` | Stand-alone scorer (no external dependencies beyond stdlib) |

## Schema

Each line in `popia_bench.jsonl`:

```json
{
  "premise":   "Our after-school maths app lets any 9-year-old sign up with just an email address.",
  "hypothesis":"The responsible party has obtained consent from a competent person before processing the child's personal information.",
  "label":     "contradiction",
  "clause":    "POPIA children's information",
  "scenario":  "edtech-no-parental-consent"
}
```

`label` is the gold class. `clause` is the POPIA clause group the example targets — used for the per-clause F1 breakdown.

## Clauses covered

| Clause | Examples |
|---|---|
| POPIA consent | 22 |
| POPIA minimality / purpose limitation | 21 |
| POPIA security safeguards | 21 |
| POPIA general processing | 21 |
| POPIA breach notification | 22 |
| POPIA cross-border transfers | 22 |
| POPIA data subject rights | 21 |
| POPIA children's information ($\S34$-$35$) | 16 |
| POPIA special personal information ($\S26$-$33$) | 16 |
| POPIA automated decision-making ($\S71$) | 15 |
| **Total** | **197** |

## Submission format

A JSONL file with one object per line:

```json
{"id": 0, "prediction": "contradiction"}
{"id": 1, "prediction": "entailment"}
...
```

- `id` is the 0-based line index into `popia_bench.jsonl`.
- `prediction` must be one of `contradiction`, `entailment`, `neutral`.

## Scoring

```bash
python bench/popia-v1/score.py submissions/your_submission.jsonl
```

Outputs JSON with `macro_f1` (overall) and `per_clause_f1`. We recommend submitters report both the overall macro F1 and the per-clause breakdown, since composite scores can hide major regressions on one clause.

## Reference scores

| Model | Macro F1 | Notes |
|---|---|---|
| `cross-encoder/nli-MiniLM2-L6-H768` (stock) | ~0.48 | Baseline — no POPIA-specific fine-tuning |
| `labrat-aiko/nli-popia-v1` | ~0.78 (7-clause subset only) | v1 covers only the first 7 clauses; not directly comparable on full bench |
| `labrat-aiko/nli-popia-v2` | ~0.78 | Trained on the 10-clause coverage matching the bench |

(Reference scores were computed at training time; the canonical leaderboard is the GitHub `bench/popia-v1/RESULTS.md` file, which we update on accepted submissions.)

## How to submit

1. Run your model over `popia_bench.jsonl` and produce a submission JSONL as above.
2. Run `python bench/popia-v1/score.py your_submission.jsonl` and confirm the score.
3. Open a pull request on `labrat-akhona/semantix-ai` adding an entry to `bench/popia-v1/RESULTS.md` with:
   - Your model identifier (URL preferred)
   - Macro F1
   - Per-clause F1 table
   - Latency on a stated reference machine
   - Any caveats about your eval recipe (zero-shot? few-shot? fine-tuned on POPIA training data?)

We do not require submitters to release weights, but we do require the submission to be reproducible from the model's public artifacts.

## Versioning and immutability

`popia_bench.jsonl` is immutable for the v1 release. Its SHA-256 is pinned in `popia_bench.sha256`. Any changes (new examples, corrections) ship as `popia-bench-v2`, `popia-bench-v3`, etc. — never overwrite v1 in place.

## License

The bench data is released under **CC BY 4.0** — free to use commercially, no attribution required for derived analyses, but attribution appreciated for direct distribution. The scoring script is **Apache-2.0**.

## Citation

```bibtex
@misc{eland2026popiabench,
  author = {Eland, Akhona},
  title = {POPIA-Bench v1: A Clause-Level Compliance {NLI} Benchmark for South African Data Protection},
  year = {2026},
  url = {https://github.com/labrat-akhona/semantix-ai/tree/master/bench/popia-v1}
}
```

## Notes on quality

- All 197 examples are hand-authored, anchored in plausible South African scenarios (Capitec, Pick n Pay, Discovery, TymeBank, SASSA, DBE, NUMSA, the Diocese of Cape Town, MRC, StatsSA, etc.).
- Authored under a deliberate policy of avoiding "trick" examples — these are realistic scenarios a compliance officer might actually see.
- We do not claim the bench is bias-free. POPIA itself was drafted to *protect against* certain categories of processing — scenarios that flag race or religion inference for "culture-fit" hiring will correctly fail. That is intentional and matches the statute.
- The `neutral` examples are deliberately written to be off-topic for the named clause (e.g., adult banking transactions in a `children's information` test). They check whether the model has learned the *boundaries* of each clause, not just the positive cases.
