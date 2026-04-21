# POPIA Fine-Tune Design

**Status:** Draft (awaiting user review)
**Date:** 2026-04-21
**Author:** Akhona Eland

## Summary

Ship a POPIA-fine-tuned NLI model as an opt-in addition to `semantix-ai`. The fine-tune is an external HuggingFace artifact plus a thin judge class and a handful of pre-built POPIA-aligned `Intent` presets. The main library is untouched; end users install with `pip install semantix-ai[popia]`. The training pipeline lives in the main repo as `scripts/` for narrative transparency, but is not part of the runtime library.

The public claim at release: "POPIA-fine-tuned NLI beats the stock cross-encoder by at least 10 percentage points on macro-F1, measured on a hand-authored, publicly reproducible held-out eval set."

## Goals

1. Ship one HuggingFace model (`labrat-akhona/nli-popia-v1`) and a ~200-LOC runtime surface in the main repo that loads it.
2. Produce a defensible, reproducible performance delta claim against the stock NLI model.
3. Keep the main product unaffected: no new runtime dependencies, no new public API surface beyond `POPIAJudge` and `semantix.presets.popia.*`, no new CLI surface beyond `semantix eval popia`.
4. Preserve the narrative moat: the seed dataset and training scripts are visible in the main repo so reviewers can audit the craft.

## Non-goals

- FSCA, NCA, CPA, or PAIA coverage. POPIA only in v1.
- A generic fine-tuning toolkit. Training scripts are one-shots, not a supported API.
- Real-time training or online learning. The model is a static artifact.
- A separate `semantix-popia` package. Everything lives in the main repo behind extras.
- Adoption gating (no waitlist). The model ships when the release gate passes.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ main repo (labrat-akhona/semantix-ai)                         │
│                                                              │
│ RUNTIME (ships to pip):                                      │
│   semantix/judges/popia.py         POPIAJudge (subclass)     │
│   semantix/presets/popia.py        5-7 Intent presets        │
│   semantix/eval/popia.py           evaluate_popia() helper   │
│   semantix/cli.py                  + `semantix eval popia`   │
│                                                              │
│ DEV-ONLY (not shipped, behind [train] extra):                │
│   data/popia_seeds.jsonl           ~60 hand-authored seeds   │
│   data/popia_eval.jsonl            ~150 hand-labeled pairs   │
│   scripts/expand_popia_seeds.py    LLM synthetic expansion   │
│   scripts/train_popia.py           transformers fine-tune    │
│   scripts/eval_popia.py            reproducibility entry     │
│   scripts/_popia_eval_hash.txt     pinned eval set hash      │
│                                                              │
│ CI:                                                          │
│   .github/workflows/popia-eval.yml release gate enforcement  │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ artifact upload (manual)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ HuggingFace (labrat-akhona/nli-popia-v1)                     │
│   model_q*.onnx            quantized variants (per CPU arch)│
│   tokenizer.json           inherited from base model         │
│   eval.jsonl               held-out set (public)             │
│   README.md                model card                        │
│   License: Apache 2.0                                        │
└──────────────────────────────────────────────────────────────┘
```

`POPIAJudge` subclasses the existing `QuantizedNLIJudge`, overriding only `_REPO_ID`, `recommended_threshold`, and adding a `clauses()` classmethod. All ONNX loading, CPU variant detection, softmax, and caching logic is inherited. The base class is ~60 LOC; the subclass is ~40 LOC.

The runtime and dev-only sides are firewalled by the `pyproject.toml` extras: `[popia]` pulls the same deps as `[turbo]` (no new runtime deps), while `[train]` adds `transformers`, `torch`, `datasets`, `accelerate` only for contributors running the training scripts.

## Components

### New runtime files

#### `semantix/judges/popia.py` (~40 LOC)

```python
class POPIAJudge(QuantizedNLIJudge):
    _REPO_ID: ClassVar[str] = "labrat-akhona/nli-popia-v1"
    recommended_threshold: ClassVar[float | None] = 0.75  # pinned from held-out calibration

    @classmethod
    def clauses(cls) -> list[str]:
        # Exact section numbers verified against POPIA Act 4 of 2013
        # during seed authoring (Data flow step 1). Labels below are
        # the canonical values the model ships with.
        return [
            "POPIA consent",
            "POPIA minimality / purpose limitation",
            "POPIA security safeguards",
            "POPIA breach notification",
            "POPIA cross-border transfers",
            "POPIA general processing",
            "POPIA data subject rights",
        ]
```

#### `semantix/presets/__init__.py`

Empty (package marker only). Establishes `semantix.presets` as a namespace for future preset families. Users import directly from submodules: `from semantix.presets.popia import POPIA_CROSS_BORDER`.

#### `semantix/presets/popia.py` (~80 LOC)

5-7 module-level `Intent` constants, one per POPIA section plus 1-2 broader ones. Each preset carries:
- A description in POPIA language
- A `clause` attribute (e.g. `"POPIA cross-border transfers"`; section numbers appended post-verification)
- A `negate` flag where the clause is negation-shaped (e.g. §22 breach notification)
- An optional `threshold` override where the clause warrants stricter matching (e.g. §19 security at 0.85)

Presets (7 total, aligned 1:1 with `POPIAJudge.clauses()`). Constant names use POPIA section numbers verified against POPIA Act 4 of 2013 during seed authoring; placeholder names `POPIA_CONSENT`, `POPIA_MINIMALITY` etc. are used during development until verified, then renamed to include the correct section number (e.g. `POPIA_11_CONSENT`):

- `POPIA_CONSENT` → clause `"POPIA consent"`
- `POPIA_MINIMALITY` → clause `"POPIA minimality / purpose limitation"`
- `POPIA_SECURITY` → clause `"POPIA security safeguards"` (threshold 0.85)
- `POPIA_BREACH` → clause `"POPIA breach notification"` (negate=True)
- `POPIA_CROSS_BORDER` → clause `"POPIA cross-border transfers"`
- `POPIA_PROCESSING` → clause `"POPIA general processing"`
- `POPIA_DATA_SUBJECT_RIGHTS` → clause `"POPIA data subject rights"`

#### `semantix/eval/popia.py` (~60 LOC)

Pure data-processing module, no side effects. Exposes:

```python
@dataclass(frozen=True)
class EvalReport:
    n_pairs: int
    stock_accuracy: float
    stock_f1_macro: float
    popia_accuracy: float
    popia_f1_macro: float
    per_clause: dict[str, tuple[float, float]]  # clause -> (stock_f1, popia_f1)
    delta_f1: float                              # popia_f1_macro - stock_f1_macro
    release_gate_passed: bool                    # delta_f1 >= 0.10 AND no per-clause regression

def evaluate_popia(eval_path: str | Path, popia_judge: Judge, base_judge: Judge) -> EvalReport: ...
```

Release gate rules:
1. `delta_f1 >= 0.10` on macro-F1
2. `popia_f1 >= stock_f1` for every clause (no per-clause regression)

### Modified runtime files

#### `semantix/__init__.py`

One new conditional re-export mirroring the existing pattern for `QuantizedNLIJudge`:

```python
try:
    from semantix.judges.popia import POPIAJudge
    __all__ += ["POPIAJudge"]
except ImportError:
    pass
```

#### `semantix/cli.py`

New `eval` subcommand with `popia` handler (~60 LOC). Downloads `eval.jsonl` from HF, loads both judges, runs `evaluate_popia()`, prints a delta table. Supports `--json` for machine-readable output. Exit codes: 0 (gate passed), 1 (gate failed), 2 (I/O or missing file).

Example output:

```
Downloading eval set from HF...       ok (150 pairs)
Loading stock QuantizedNLIJudge...    ok
Loading POPIAJudge...                 ok

                   stock    POPIA    Δ
Accuracy           0.62     0.78     +0.16
F1 (macro)         0.59     0.76     +0.17
POPIA cross-border 0.55     0.82     +0.27
POPIA minimality   0.64     0.74     +0.10
...

Release gate (≥0.10 F1 delta, no per-clause regression): PASS
```

#### `pyproject.toml`

Two new extras:

```toml
popia = ["semantix-ai[turbo]"]
train = [
    "transformers>=4.40",
    "torch>=2.0",
    "datasets>=2.14",
    "accelerate>=0.30",
    "optimum[onnxruntime]>=1.20",
]
```

`[popia]` is a re-pointer for discoverability — installs the same deps as `[turbo]`. No runtime cost. `[train]` is contributor-only.

### New dev-only files

#### `data/popia_seeds.jsonl`

~60 hand-authored NLI pairs, ~10 per clause, balanced across entailment/neutral/contradiction. Every row has:
```json
{"clause":"POPIA cross-border transfers","premise":"...","hypothesis":"...","label":"entailment","scenario":"cross-border-saas"}
```
This file is the load-bearing artifact for the "SA dev hand-authored the seeds" narrative. Reviewers can read it in 10 minutes.

#### `data/popia_eval.jsonl`

~150 hand-labeled held-out pairs, same schema. **Never fed to training.** Uploaded to HF alongside the model so external parties can reproduce the delta claim. SHA-256 pinned in `scripts/_popia_eval_hash.txt`.

#### `scripts/expand_popia_seeds.py`

Calls Claude or GPT-4 with a fixed prompt: *"Rewrite this NLI pair 30 ways while preserving the label. Vary: business domain, company size, SA province, phrasing register."* Writes `data/popia_train.jsonl`. Post-processing:
- Deduplicate on hash of `(premise, hypothesis)`.
- Sanity filter: re-classify each generated pair with the stock NLI model; drop rows where the stock model's top prediction confidence for the intended label is below 0.5 (catches obvious LLM paraphrase failures).
- Target post-filter size: ~1500 rows.

Cost: ~$5-15 in LLM API calls.

#### `scripts/train_popia.py`

Main training entry point. Steps:
1. Compute SHA-256 of `data/popia_eval.jsonl`, compare against `scripts/_popia_eval_hash.txt`, abort if mismatch.
2. Load `cross-encoder/nli-MiniLM2-L6-H768` via `transformers.AutoModelForSequenceClassification`.
3. Load `data/popia_train.jsonl` via `datasets.Dataset.from_json`.
4. Train: 3 epochs, batch size 16, learning rate 2e-5, warmup 10%, weight decay 0.01. Best checkpoint by F1 on a small slice carved from the training data (*not* the release eval set).
5. Export to ONNX via `optimum.onnxruntime.ORTModelForSequenceClassification`.
6. Quantize to INT8 dynamic, produce CPU variants (AVX2, AVX512, AVX512-VNNI, ARM64) matching the `QuantizedNLIJudge` file layout.
7. Output to `./out/nli-popia-v1/` for manual upload.

Expected runtime: ~30 min on a single GPU, ~3 hours on CPU.

#### `scripts/eval_popia.py`

Wraps `semantix.eval.popia.evaluate_popia()` for local reproducibility. Same code path as `semantix eval popia` — no drift.

#### `scripts/_popia_eval_hash.txt`

One-line file: the SHA-256 of `data/popia_eval.jsonl` at authoring time. Committed once, never updated. Any attempt to evolve the eval set requires a visible commit changing this file — auditable.

### New test files

#### `semantix/tests/test_popia_judge.py` (~6 tests)

Patches `huggingface_hub.hf_hub_download` and `onnxruntime.InferenceSession` to return fakes. Verifies:
- `POPIAJudge._REPO_ID == "labrat-akhona/nli-popia-v1"`
- `POPIAJudge.recommended_threshold` returns the pinned value
- `POPIAJudge.clauses()` returns the expected list
- Subclass identity: `isinstance(POPIAJudge(), QuantizedNLIJudge)` and `Judge`
- `evaluate()` delegates to parent correctly (fake logits → expected `Verdict`)
- HF download failure raises `RuntimeError` with manual-download guidance (no silent fallback to base NLI)

#### `semantix/tests/test_popia_presets.py` (~8 tests)

Pure data validation, no mocking:
- All presets are `Intent` instances
- Every preset has a non-empty description
- Every preset has a `clause` attribute matching a known POPIA section
- Negation-shaped presets have `negate=True`; others have `negate=False`
- Thresholds, when set, are in `[0.5, 0.95]`
- Module `__all__` matches the actual preset count
- `POPIA_CLAUSES` canonical list aligns 1:1 with presets' `.clause` attributes
- Round-trip: each preset's description → `_to_hypothesis()` → non-empty hypothesis

#### `semantix/tests/test_cli_eval.py` (~4 tests)

Patches HF download + both judge classes with fakes returning scripted verdicts:
- `semantix eval popia` exits 0 when gate passes, 1 when gate fails
- `--json` output contains all `EvalReport` fields
- Human-readable output includes per-clause breakdown
- Missing eval file produces exit code 2 with a clear message

#### `semantix/tests/test_eval_popia.py` (~5 tests)

Covers `semantix/eval/popia.py` with hand-crafted fixture predictions. Asserts the exact `EvalReport` output including the per-clause regression gate rule.

#### `tests/integration/test_popia_end_to_end.py`

Marked `@pytest.mark.integration`, **excluded from default pytest run**. Actually downloads the HF model, loads `POPIAJudge`, runs 3-4 preset intents against hand-chosen outputs, verifies verdicts match expected semantics. Invoked by developer before release and by the `popia-eval.yml` CI workflow.

## Data flow — training

1. **Author seeds** (~1-2 days, manual). Write `data/popia_seeds.jsonl` with POPIA act text open alongside. 60 pairs, 8-12 per clause, balanced labels.
2. **Synthetic expansion** (~30 min LLM calls). `scripts/expand_popia_seeds.py` produces `data/popia_train.jsonl` (~1500 rows after filtering).
3. **Author held-out eval** (~1 day, manual). Write `data/popia_eval.jsonl` by hand. 150 pairs. Commit once. Pin hash to `scripts/_popia_eval_hash.txt`.
4. **Fine-tune** (~30 min GPU / ~3 hr CPU). `scripts/train_popia.py` loads train set, fine-tunes base NLI, exports ONNX, quantizes.
5. **Manual HF upload**. Create `labrat-akhona/nli-popia-v1`, upload ONNX variants + tokenizer + `eval.jsonl` + model card. License: **Apache 2.0 on the model**, MIT on the code.

## Data flow — runtime

Cold path (first call):
1. `POPIAJudge()` → inherited `QuantizedNLIJudge.__init__` detects CPU variant.
2. `hf_hub_download("labrat-akhona/nli-popia-v1", <variant>.onnx)` → one-time ~25 MB download, cached in `~/.cache/huggingface/hub/`.
3. `onnxruntime.InferenceSession` + tokenizer loaded and cached on the judge instance.

Warm path: identical to `QuantizedNLIJudge`. ~15 ms per `evaluate()` call.

Threshold resolution preserves the existing order: explicit param > intent.threshold > judge.recommended_threshold > default 0.8. Negation handling reuses the existing `negate=True` machinery on `Intent`. Audit trail integration is zero-new-code — the certificate's `intent.description` contains POPIA preset text.

**Fallback policy: none.** If `[popia]` is installed and the HF repo is unreachable, `POPIAJudge()` raises `RuntimeError` with a "download the model manually to X" message. No silent fallback to base NLI — that would violate the contract.

## Release gate

Enforced in CI via `.github/workflows/popia-eval.yml`. Triggers on:
- Tags matching `v*-popia`
- Any commit touching `semantix/judges/popia.py` or `semantix/presets/popia.py`

Workflow steps:
1. Install `semantix-ai[popia]`.
2. Run `semantix eval popia --json > report.json`.
3. Fail the build if `report.release_gate_passed` is false.
4. Upload `report.json` as a workflow artifact.

Expected runtime: ~90 seconds on `ubuntu-latest` (no GPU).

**Public-claim discipline:** no performance claim in any announcement, README, or PR body may exceed the numbers in the most recent release `report.json`. This is enforceable by PR review.

**Gate-failure policy:** if the 10 pp gate fails at release time, we do **not** ship publicly and we do **not** relax the gate to match the achieved delta. We iterate on training data (more seeds, better synthetic prompts, more expansion volume) or base-model choice until the gate is met. This preserves the "beats base by double digits" narrative and prevents drift toward "we shipped what we got."

## Eval-set integrity

`scripts/train_popia.py` computes SHA-256 of `data/popia_eval.jsonl` at training time and compares against `scripts/_popia_eval_hash.txt`. Mismatch aborts training with a loud error.

Evolving the eval set requires a visible commit updating the pinned hash. A reviewer can audit that single commit to verify the evolution was intentional, not inadvertent leakage from training iteration.

## License split

- **Code** (main repo, all scripts, all tests, all runtime files): MIT
- **Model artifact** (HF repo): Apache 2.0
- **Seed dataset** (`data/popia_seeds.jsonl`, `data/popia_eval.jsonl`): CC-BY-4.0

Rationale: MIT matches the rest of the semantix-ai ecosystem. Apache 2.0 on the model matches ecosystem norms for NLI weights on HF and allows commercial use with attribution. CC-BY-4.0 on the dataset is standard for open NLI datasets (SNLI, MultiNLI precedent) and keeps the "SA dev hand-authored these" attribution visible downstream.

## Narrative and announcement artifacts

Not part of the implementation plan but noted here so the spec is complete:
- A blog post draft under `articles/drafts/` titled "Fine-tuning NLI for POPIA: how and why."
- A 3-line quickstart in the main README under a new "POPIA support" section.
- A model card on HF with: training data sizes, clause coverage, eval results, reproducibility instructions (`pip install semantix-ai[popia] && semantix eval popia`), license.

## Risks and mitigations

1. **The fine-tune produces no measurable delta.** Mitigation: release gate fails, we don't ship. Falling back to either more training data or a different base model becomes the next iteration — not a public event.
2. **Eval set is too small to be statistically meaningful.** 150 pairs is on the low end. Mitigation: bootstrap confidence intervals in `EvalReport` (future extension, not v1). For v1, the delta needs to be ≥10 pp, which is well outside noise at n=150 if real.
3. **HF rate limits block model downloads in CI.** Mitigation: CI caches the HF cache directory between runs. Documented as a known failure mode in the workflow.
4. **POPIA presets encode the author's reading of the act, which may differ from legal opinion.** Mitigation: the model card and presets module docstring explicitly state that presets are engineering heuristics, not legal advice. Users are directed to their own DPIA / legal counsel for compliance determinations.
5. **An overseas team ships a POPIA-tuned NLI first while we're building.** Mitigation: public-facing timeline discipline — no waitlist, no pre-announcement, ship directly when the gate passes.

## Out-of-scope extensions (future v2+)

- FSCA / NCA / CPA / PAIA coverage
- Multi-model ensembles (POPIA + general NLI with confidence weighting)
- Fine-tuning toolkit as a public API
- Afrikaans and isiZulu input handling
- Bootstrap confidence intervals in `EvalReport`
- Active learning loop from production `TrainingCollector` logs

Each of these is a separate spec, not a feature creep target for v1.
