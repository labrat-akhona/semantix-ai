# POPIA Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a POPIA-fine-tuned NLI model as an opt-in addition to `semantix-ai`, with a thin `POPIAJudge` class, 7 pre-built `Intent` presets, a CLI release gate, and a CI workflow that enforces a ≥10 pp F1 delta against stock NLI on a publicly reproducible held-out eval set.

**Architecture:** The training pipeline (`scripts/`, `data/`) is dev-only behind `[train]` extras and not imported at runtime. The runtime surface (`POPIAJudge`, presets, `evaluate_popia()`, `semantix eval popia` CLI) is ~200 LOC across 4 new files + small edits to `semantix/__init__.py`, `semantix/cli.py`, and `pyproject.toml`. The ONNX model ships as a HuggingFace artifact (`labrat-akhona/nli-popia-v1`); `POPIAJudge` subclasses the existing `QuantizedNLIJudge` and only overrides the repo ID, threshold, and a `clauses()` classmethod.

**Tech Stack:** Python 3.10+, `transformers`, `torch`, `datasets`, `accelerate`, `optimum[onnxruntime]` (dev-only for training); `onnxruntime`, `tokenizers`, `huggingface-hub` (runtime, already in `[turbo]` extras); `pytest` for tests; GitHub Actions for CI; Apache 2.0 licensed model on HuggingFace; MIT licensed code; CC-BY-4.0 licensed seed/eval datasets.

**Spec reference:** `docs/superpowers/specs/2026-04-21-popia-finetune-design.md`

---

## Phase 0: Foundations

### Task 1: Create `semantix/presets/` package marker

**Files:**
- Create: `semantix/presets/__init__.py`

- [ ] **Step 1: Create the package marker**

Create `semantix/presets/__init__.py` with this exact content:

```python
"""Pre-built Intent presets for specific regulatory or domain contexts.

Users import presets directly from submodules:

    from semantix.presets.popia import POPIA_CROSS_BORDER

This package is namespace-only; no symbols are re-exported at the top level.
"""
```

- [ ] **Step 2: Verify the package imports**

Run: `python -c "import semantix.presets; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add semantix/presets/__init__.py
git commit -m "feat: add semantix.presets package namespace"
```

---

### Task 2: Create `semantix/eval/` package marker

**Files:**
- Create: `semantix/eval/__init__.py`

- [ ] **Step 1: Create the package marker**

Create `semantix/eval/__init__.py` with this exact content:

```python
"""Evaluation harnesses for semantix judges.

Submodules expose pure-data evaluation routines that compare a candidate
judge against a baseline on a labeled dataset and produce release-gate
reports. Not imported at module load; users import from submodules.
"""
```

- [ ] **Step 2: Verify the package imports**

Run: `python -c "import semantix.eval; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add semantix/eval/__init__.py
git commit -m "feat: add semantix.eval package namespace"
```

---

## Phase 1: Dev-only training infrastructure

### Task 3: Hand-author POPIA seed dataset

**Files:**
- Create: `data/popia_seeds.jsonl`

**This is a MANUAL human-authoring task. No code. No test. Estimated 1-2 days of focused work with POPIA Act 4 of 2013 open alongside (free at https://popia.co.za).**

- [ ] **Step 1: Read POPIA sections relevant to the 7 clauses**

Reference reading (use these exact section anchors in the POPIA text):
- Consent — the condition around data-subject consent
- Minimality / purpose limitation — processing not excessive relative to purpose
- Security safeguards — technical and organisational measures
- Breach notification — notification to regulator and data subjects
- Cross-border transfers — the §71-equivalent condition
- General processing — the overall conditions for lawful processing
- Data subject rights — access, correction, objection

Pin the exact section numbers as you read. Update `POPIAJudge.clauses()` and preset names later (Task 10, 11) to include the verified numbers.

- [ ] **Step 2: Author 60 seed pairs — 8-12 per clause, balanced labels**

Create `data/popia_seeds.jsonl`, one JSON object per line, using this exact schema:

```json
{"clause":"POPIA cross-border transfers","premise":"We store user records exclusively in eu-central-1.","hypothesis":"Personal information is transferred outside South Africa.","label":"entailment","scenario":"cross-border-saas"}
```

Required keys on every row:
- `clause`: one of the 7 canonical strings (exact match to `POPIAJudge.clauses()`)
- `premise`: a realistic output an LLM might produce in an SA business context
- `hypothesis`: the NLI-style claim being tested
- `label`: exactly one of `"entailment"`, `"neutral"`, `"contradiction"`
- `scenario`: a short tag (e.g. `"cross-border-saas"`, `"clinic-consent"`, `"bank-breach"`) so reviewers can see domain coverage

Target distribution per clause: 3-4 entailment, 3-4 neutral, 3-4 contradiction.

Draw scenarios from real SA contexts: SaaS hosting in eu-central-1, banking KYC flows, private healthcare consent forms, retail loyalty programmes, government service portals, NGO donor data handling.

- [ ] **Step 3: Validate the file is well-formed JSONL**

Run:
```bash
python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('data/popia_seeds.jsonl')]
print(f'{len(rows)} rows')
print('per clause:', Counter(r['clause'] for r in rows))
print('per label:', Counter(r['label'] for r in rows))
assert len(rows) >= 55
assert len(set(r['clause'] for r in rows)) == 7
assert all(r['label'] in ('entailment','neutral','contradiction') for r in rows)
print('ok')
"
```
Expected: ~60 rows, 7 clauses, each label represented, final `ok`.

- [ ] **Step 4: Commit**

```bash
git add data/popia_seeds.jsonl
git commit -m "data: add hand-authored POPIA seed NLI pairs"
```

---

### Task 4: Hand-author POPIA held-out eval dataset

**Files:**
- Create: `data/popia_eval.jsonl`

**MANUAL human-authoring task. ~1 day. These pairs must NEVER be shown to the training loop.**

- [ ] **Step 1: Author 150 held-out pairs using the same schema as seeds**

Same JSONL schema as Task 3. Use **different premises and scenarios** than the seeds — do not paraphrase seed rows, author fresh examples. Target: 150 rows, ~20-25 per clause, balanced labels.

Reviewers should be able to load this file and judge each row's label in isolation without seeing the seed file.

- [ ] **Step 2: Validate the file is well-formed and disjoint from seeds**

Run:
```bash
python -c "
import json
from collections import Counter
seeds = [json.loads(l) for l in open('data/popia_seeds.jsonl')]
evals = [json.loads(l) for l in open('data/popia_eval.jsonl')]
seed_keys = {(r['premise'], r['hypothesis']) for r in seeds}
eval_keys = {(r['premise'], r['hypothesis']) for r in evals}
overlap = seed_keys & eval_keys
print(f'{len(evals)} eval rows')
print('per clause:', Counter(r['clause'] for r in evals))
print('per label:', Counter(r['label'] for r in evals))
print(f'overlap with seeds: {len(overlap)}')
assert len(evals) >= 140
assert len(overlap) == 0, f'LEAK: {overlap}'
print('ok')
"
```
Expected: ~150 rows, balanced clauses, `overlap with seeds: 0`, final `ok`.

- [ ] **Step 3: Commit**

```bash
git add data/popia_eval.jsonl
git commit -m "data: add hand-labeled POPIA held-out eval set"
```

---

### Task 5: Pin the held-out eval set hash

**Files:**
- Create: `scripts/_popia_eval_hash.txt`

- [ ] **Step 1: Compute and write the hash**

Run:
```bash
mkdir -p scripts
python -c "import hashlib; print(hashlib.sha256(open('data/popia_eval.jsonl','rb').read()).hexdigest())" > scripts/_popia_eval_hash.txt
cat scripts/_popia_eval_hash.txt
```
Expected: a 64-character hex hash printed. File contains that hash + newline.

- [ ] **Step 2: Commit**

```bash
git add scripts/_popia_eval_hash.txt
git commit -m "data: pin POPIA eval set hash for training integrity check"
```

---

### Task 6: Synthetic seed expansion script

**Files:**
- Create: `scripts/expand_popia_seeds.py`

- [ ] **Step 1: Write the expansion script**

Create `scripts/expand_popia_seeds.py`:

```python
"""Expand POPIA seed NLI pairs into a training set via LLM paraphrase.

Reads data/popia_seeds.jsonl, calls an LLM to produce ~30 variants per seed
preserving the label, deduplicates, and writes data/popia_train.jsonl.

Usage:
    OPENAI_API_KEY=... python scripts/expand_popia_seeds.py
    # or
    ANTHROPIC_API_KEY=... python scripts/expand_popia_seeds.py --provider anthropic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

SEEDS_PATH = Path("data/popia_seeds.jsonl")
TRAIN_PATH = Path("data/popia_train.jsonl")
VARIANTS_PER_SEED = 30

EXPAND_PROMPT = """You are expanding a natural-language-inference training example.

Given one (premise, hypothesis, label) triple, produce {n} NEW variants that
preserve the label exactly. Each variant MUST:
- vary business domain (SaaS, banking, healthcare, retail, government, NGO)
- vary company size (startup, SME, enterprise)
- vary SA province context where plausible (Western Cape, Gauteng, KZN, etc.)
- vary phrasing register (casual, formal, legalese)
- keep the NLI label identical to the original

The clause (POPIA concept) is: {clause}
The original label is: {label}

Original premise: {premise}
Original hypothesis: {hypothesis}

Output ONLY a JSON array of {n} objects, each with keys "premise" and
"hypothesis". No commentary, no markdown fencing, no extra fields.
"""


def call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def expand_seed(seed: dict, provider: str) -> list[dict]:
    prompt = EXPAND_PROMPT.format(
        n=VARIANTS_PER_SEED,
        clause=seed["clause"],
        label=seed["label"],
        premise=seed["premise"],
        hypothesis=seed["hypothesis"],
    )
    raw = call_anthropic(prompt) if provider == "anthropic" else call_openai(prompt)
    # Strip optional markdown fencing
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # OpenAI json_object mode wraps in {"variants": [...]}; Anthropic returns bare array
    data = json.loads(raw)
    if isinstance(data, dict):
        # heuristic: take the first list-valued field
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    rows = []
    for v in data:
        if not isinstance(v, dict) or "premise" not in v or "hypothesis" not in v:
            continue
        rows.append({
            "clause": seed["clause"],
            "premise": v["premise"],
            "hypothesis": v["hypothesis"],
            "label": seed["label"],
            "scenario": seed.get("scenario", "synthetic"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    args = ap.parse_args()

    if not SEEDS_PATH.exists():
        print(f"missing {SEEDS_PATH}", file=sys.stderr)
        return 1

    seeds = [json.loads(l) for l in SEEDS_PATH.read_text().splitlines() if l.strip()]
    print(f"expanding {len(seeds)} seeds -> target ~{len(seeds) * VARIANTS_PER_SEED} rows")

    seen: set[str] = set()
    out: list[dict] = []
    # Include seeds themselves so the training set contains the original anchors
    for s in seeds:
        key = hashlib.sha256(f"{s['premise']}|{s['hypothesis']}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(s)

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] expanding {seed['clause']}/{seed['label']}...", flush=True)
        try:
            variants = expand_seed(seed, args.provider)
        except Exception as e:
            print(f"  failed: {e}; skipping", file=sys.stderr)
            continue
        kept = 0
        for v in variants:
            key = hashlib.sha256(f"{v['premise']}|{v['hypothesis']}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
            kept += 1
        print(f"  kept {kept}/{len(variants)} (running total: {len(out)})")
        time.sleep(0.5)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(out)} rows to {TRAIN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the expansion (requires API key, ~30 min, ~$5-15)**

Run:
```bash
ANTHROPIC_API_KEY=... python scripts/expand_popia_seeds.py
```
Expected: prints `expanding 60 seeds -> target ~1800 rows`, iterates through each seed, final line `wrote ~1500-1900 rows to data/popia_train.jsonl`.

- [ ] **Step 3: Sanity-filter the training set against stock NLI**

Create a small inline filter (one-shot, not committed):
```bash
python -c "
import json
from semantix.judges.quantized_nli import QuantizedNLIJudge
from semantix.judges.nli import _to_hypothesis

judge = QuantizedNLIJudge()
rows = [json.loads(l) for l in open('data/popia_train.jsonl')]
kept = []
for i, r in enumerate(rows):
    if i % 100 == 0: print(f'{i}/{len(rows)}')
    v = judge.evaluate(r['premise'], r['hypothesis'], threshold=0.5)
    stock_says_entailed = v.passed
    intended_entailed = r['label'] == 'entailment'
    # Keep rows where stock predicts the intended label with reasonable confidence,
    # OR where the stock disagreement is the whole point of fine-tuning (low-confidence edge cases)
    if v.score is None or abs((v.score or 0.5) - 0.5) < 0.1:
        kept.append(r)  # uncertain: fine-tune can help
    elif stock_says_entailed == intended_entailed:
        kept.append(r)  # stock agrees: safe training signal
    # else: stock confidently disagrees with intended label -> likely paraphrase failure
with open('data/popia_train.jsonl', 'w') as f:
    for r in kept:
        f.write(json.dumps(r) + '\n')
print(f'filtered: kept {len(kept)}/{len(rows)}')
"
```
Expected: reports final kept count (~1200-1700 rows).

- [ ] **Step 4: Commit the script (NOT the generated data)**

The train set is a derived artifact; add it to `.gitignore` to avoid committing megabytes of synthetic text.

```bash
echo "data/popia_train.jsonl" >> .gitignore
git add scripts/expand_popia_seeds.py .gitignore
git commit -m "feat(train): add POPIA seed expansion script"
```

---

### Task 7: Fine-tuning + ONNX export script

**Files:**
- Create: `scripts/train_popia.py`

- [ ] **Step 1: Write the training script**

Create `scripts/train_popia.py`:

```python
"""Fine-tune cross-encoder/nli-MiniLM2-L6-H768 on POPIA data, export to ONNX.

Usage:
    python scripts/train_popia.py           # GPU if available, else CPU
    python scripts/train_popia.py --epochs 5

Requires: pip install semantix-ai[train]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
TRAIN_PATH = Path("data/popia_train.jsonl")
EVAL_PATH = Path("data/popia_eval.jsonl")
EVAL_HASH_PATH = Path("scripts/_popia_eval_hash.txt")
OUT_DIR = Path("out/nli-popia-v1")


def verify_eval_integrity() -> None:
    """Abort if the eval set has been modified since the pinned hash was set."""
    if not EVAL_HASH_PATH.exists():
        sys.exit(f"missing {EVAL_HASH_PATH} -- run Task 5 first")
    pinned = EVAL_HASH_PATH.read_text().strip()
    current = hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest()
    if pinned != current:
        sys.exit(
            f"EVAL SET INTEGRITY FAILURE\n"
            f"  pinned hash: {pinned}\n"
            f"  current hash: {current}\n"
            f"If this change is intentional, update {EVAL_HASH_PATH} in a "
            f"standalone commit so reviewers can audit the change."
        )


def label_to_id(label: str) -> int:
    # NLI convention: 0=entailment, 1=neutral, 2=contradiction
    return {"entailment": 0, "neutral": 1, "contradiction": 2}[label]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    verify_eval_integrity()

    # Deferred imports -- keep [train] deps out of the import path until runtime
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    if not TRAIN_PATH.exists():
        sys.exit(f"missing {TRAIN_PATH} -- run Task 6 first")

    rows = [json.loads(l) for l in TRAIN_PATH.read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} training rows")

    # Hold out 10% for in-training best-checkpoint selection
    # (the release eval set is separate and untouched)
    split_idx = int(len(rows) * 0.9)
    train_rows, dev_rows = rows[:split_idx], rows[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

    def tokenize(batch):
        return tokenizer(
            batch["premise"], batch["hypothesis"],
            truncation=True, padding="max_length", max_length=256,
        )

    def to_ds(rows):
        ds = Dataset.from_list([
            {"premise": r["premise"], "hypothesis": r["hypothesis"], "labels": label_to_id(r["label"])}
            for r in rows
        ])
        return ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])

    train_ds = to_ds(train_rows)
    dev_ds = to_ds(dev_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pytorch_out = OUT_DIR / "pytorch"

    targs = TrainingArguments(
        output_dir=str(pytorch_out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(pytorch_out))
    tokenizer.save_pretrained(str(pytorch_out))
    print(f"pytorch checkpoint saved to {pytorch_out}")

    # Export to ONNX
    onnx_dir = OUT_DIR / "onnx"
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(pytorch_out), export=True,
    )
    ort_model.save_pretrained(str(onnx_dir))
    print(f"onnx model saved to {onnx_dir}")

    # Quantize to INT8 dynamic -- produce the variants the existing
    # QuantizedNLIJudge detects (AVX2, AVX512, AVX512-VNNI, ARM64)
    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    variants = {
        "model_quint8_avx2.onnx": AutoQuantizationConfig.avx2(is_static=False, per_channel=False),
        "model_qint8_avx512.onnx": AutoQuantizationConfig.avx512(is_static=False, per_channel=False),
        "model_qint8_avx512_vnni.onnx": AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False),
        "model_qint8_arm64.onnx": AutoQuantizationConfig.arm64(is_static=False, per_channel=False),
    }
    for filename, qconfig in variants.items():
        target = onnx_dir / filename
        quantizer.quantize(save_dir=str(onnx_dir / filename.replace(".onnx", "_tmp")), quantization_config=qconfig)
        # optimum writes to a dir; move the produced file out and clean up
        produced = list((onnx_dir / filename.replace(".onnx", "_tmp")).glob("*.onnx"))[0]
        shutil.move(str(produced), str(target))
        shutil.rmtree(onnx_dir / filename.replace(".onnx", "_tmp"))
        print(f"quantized -> {target}")

    # Copy the eval set alongside the artifact
    shutil.copy(str(EVAL_PATH), str(OUT_DIR / "eval.jsonl"))
    print(f"bundled eval set -> {OUT_DIR / 'eval.jsonl'}")

    print(f"\nDONE. Upload {OUT_DIR} to HuggingFace as labrat-akhona/nli-popia-v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the script fails cleanly without [train] extras**

Run:
```bash
python scripts/train_popia.py --epochs 1 2>&1 | head -5
```
Expected: either integrity-check passes and it tries to import `torch` (error if `[train]` not installed), OR the file-missing error from `TRAIN_PATH`. Not a crash at parse time.

- [ ] **Step 3: Run the actual training (requires [train] installed, ~30min GPU / ~3hr CPU)**

Run:
```bash
pip install -e ".[train]"
python scripts/train_popia.py
```
Expected: trains for 3 epochs, saves pytorch checkpoint, exports ONNX, produces 4 quantized variants under `out/nli-popia-v1/onnx/`, bundles `eval.jsonl`. Final line: `DONE. Upload out/nli-popia-v1 to HuggingFace as labrat-akhona/nli-popia-v1`.

- [ ] **Step 4: Commit the script (NOT the out/ directory)**

```bash
echo "out/" >> .gitignore
git add scripts/train_popia.py .gitignore
git commit -m "feat(train): add POPIA fine-tune + ONNX export script"
```

---

## Phase 2: Runtime library

### Task 8: `evaluate_popia()` eval harness — test first

**Files:**
- Create: `semantix/eval/popia.py`
- Test: `semantix/tests/test_eval_popia.py`

- [ ] **Step 1: Write the failing tests**

Create `semantix/tests/test_eval_popia.py`:

```python
"""Unit tests for semantix.eval.popia."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantix.eval.popia import EvalReport, evaluate_popia
from semantix.judges import Judge, Verdict


class ScriptedJudge(Judge):
    """Judge that returns pre-scripted verdicts keyed by (premise, hypothesis)."""

    def __init__(self, script: dict[tuple[str, str], bool]):
        self._script = script

    def evaluate(self, output: str, intent_description: str, threshold: float = 0.8) -> Verdict:
        passed = self._script.get((output, intent_description), False)
        return Verdict(passed=passed, score=0.9 if passed else 0.1, reason=None)


def _write_eval(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_perfect_popia_beats_random_stock(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "contradiction"},
        {"clause": "POPIA security safeguards", "premise": "p3", "hypothesis": "h3", "label": "entailment"},
        {"clause": "POPIA security safeguards", "premise": "p4", "hypothesis": "h4", "label": "neutral"},
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): False, ("p3", "h3"): True, ("p4", "h4"): False})
    stock = ScriptedJudge({("p1", "h1"): False, ("p2", "h2"): True, ("p3", "h3"): False, ("p4", "h4"): True})

    report = evaluate_popia(eval_path, popia, stock)
    assert report.n_pairs == 4
    assert report.popia_accuracy == 1.0
    assert report.stock_accuracy == 0.0
    assert report.delta_f1 > 0.5


def test_release_gate_requires_both_delta_and_no_per_clause_regression(tmp_path):
    # POPIA beats stock on consent but regresses on security
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "entailment"},
        {"clause": "POPIA security safeguards", "premise": "p3", "hypothesis": "h3", "label": "entailment"},
        {"clause": "POPIA security safeguards", "premise": "p4", "hypothesis": "h4", "label": "entailment"},
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): True, ("p3", "h3"): False, ("p4", "h4"): False})
    stock = ScriptedJudge({("p1", "h1"): False, ("p2", "h2"): False, ("p3", "h3"): True, ("p4", "h4"): True})

    report = evaluate_popia(eval_path, popia, stock)
    assert report.per_clause["POPIA consent"][1] > report.per_clause["POPIA consent"][0]
    assert report.per_clause["POPIA security safeguards"][1] < report.per_clause["POPIA security safeguards"][0]
    assert report.release_gate_passed is False


def test_gate_passes_when_delta_ge_10pp_and_no_regression(tmp_path):
    # 8 rows. POPIA: 7 right, stock: 5 right. Delta accuracy = 25 pp.
    rows = [
        {"clause": "POPIA consent", "premise": f"p{i}", "hypothesis": f"h{i}", "label": "entailment"}
        for i in range(4)
    ] + [
        {"clause": "POPIA security safeguards", "premise": f"s{i}", "hypothesis": f"sh{i}", "label": "entailment"}
        for i in range(4)
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia_script = {(f"p{i}", f"h{i}"): True for i in range(4)} | {(f"s{i}", f"sh{i}"): True for i in range(3)}
    stock_script = {(f"p{i}", f"h{i}"): i < 3 for i in range(4)} | {(f"s{i}", f"sh{i}"): i < 2 for i in range(4)}
    popia = ScriptedJudge(popia_script)
    stock = ScriptedJudge(stock_script)

    report = evaluate_popia(eval_path, popia, stock)
    assert report.delta_f1 >= 0.10
    assert all(popia_f >= stock_f for stock_f, popia_f in report.per_clause.values())
    assert report.release_gate_passed is True


def test_gate_fails_when_delta_below_10pp(tmp_path):
    rows = [
        {"clause": "POPIA consent", "premise": "p1", "hypothesis": "h1", "label": "entailment"},
        {"clause": "POPIA consent", "premise": "p2", "hypothesis": "h2", "label": "entailment"},
    ]
    eval_path = _write_eval(tmp_path, rows)

    popia = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): False})
    stock = ScriptedJudge({("p1", "h1"): True, ("p2", "h2"): False})

    report = evaluate_popia(eval_path, popia, stock)
    assert report.delta_f1 == 0.0
    assert report.release_gate_passed is False


def test_missing_eval_file_raises_filenotfound(tmp_path):
    popia = ScriptedJudge({})
    stock = ScriptedJudge({})
    with pytest.raises(FileNotFoundError):
        evaluate_popia(tmp_path / "nope.jsonl", popia, stock)
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest semantix/tests/test_eval_popia.py -v`
Expected: 5 failures with `ModuleNotFoundError: semantix.eval.popia` or `ImportError`.

- [ ] **Step 3: Implement `semantix/eval/popia.py`**

Create `semantix/eval/popia.py`:

```python
"""POPIA eval harness -- compare candidate judge against baseline, produce release-gate report."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from semantix.judges import Judge

DELTA_F1_GATE = 0.10


@dataclass(frozen=True)
class EvalReport:
    n_pairs: int
    stock_accuracy: float
    stock_f1_macro: float
    popia_accuracy: float
    popia_f1_macro: float
    per_clause: dict[str, tuple[float, float]]  # clause -> (stock_f1, popia_f1)
    delta_f1: float
    release_gate_passed: bool


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _macro_f1(predictions: list[tuple[bool, bool]]) -> float:
    # Binary: entailment (True) vs not (False). Macro-F1 = mean(F1 per class).
    tp_pos = sum(1 for p, y in predictions if p and y)
    fp_pos = sum(1 for p, y in predictions if p and not y)
    fn_pos = sum(1 for p, y in predictions if not p and y)
    tp_neg = sum(1 for p, y in predictions if not p and not y)
    fp_neg = sum(1 for p, y in predictions if not p and y)
    fn_neg = sum(1 for p, y in predictions if p and not y)
    return (_f1(tp_pos, fp_pos, fn_pos) + _f1(tp_neg, fp_neg, fn_neg)) / 2


def evaluate_popia(
    eval_path: str | Path,
    popia_judge: Judge,
    base_judge: Judge,
) -> EvalReport:
    """Run both judges against a POPIA eval JSONL file; compute report and gate."""
    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(str(eval_path))

    rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]

    # Collect (pred, truth) pairs globally and per clause
    all_popia: list[tuple[bool, bool]] = []
    all_stock: list[tuple[bool, bool]] = []
    per_clause_popia: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    per_clause_stock: dict[str, list[tuple[bool, bool]]] = defaultdict(list)

    for r in rows:
        truth = r["label"] == "entailment"
        popia_v = popia_judge.evaluate(r["premise"], r["hypothesis"])
        stock_v = base_judge.evaluate(r["premise"], r["hypothesis"])
        all_popia.append((popia_v.passed, truth))
        all_stock.append((stock_v.passed, truth))
        per_clause_popia[r["clause"]].append((popia_v.passed, truth))
        per_clause_stock[r["clause"]].append((stock_v.passed, truth))

    n = len(rows)
    popia_acc = sum(1 for p, y in all_popia if p == y) / n if n else 0.0
    stock_acc = sum(1 for p, y in all_stock if p == y) / n if n else 0.0
    popia_f1 = _macro_f1(all_popia)
    stock_f1 = _macro_f1(all_stock)

    per_clause: dict[str, tuple[float, float]] = {}
    for clause in per_clause_popia:
        per_clause[clause] = (
            _macro_f1(per_clause_stock[clause]),
            _macro_f1(per_clause_popia[clause]),
        )

    delta_f1 = popia_f1 - stock_f1
    no_regression = all(popia_f >= stock_f for stock_f, popia_f in per_clause.values())
    gate_passed = (delta_f1 >= DELTA_F1_GATE) and no_regression

    return EvalReport(
        n_pairs=n,
        stock_accuracy=stock_acc,
        stock_f1_macro=stock_f1,
        popia_accuracy=popia_acc,
        popia_f1_macro=popia_f1,
        per_clause=per_clause,
        delta_f1=delta_f1,
        release_gate_passed=gate_passed,
    )
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest semantix/tests/test_eval_popia.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add semantix/eval/popia.py semantix/tests/test_eval_popia.py
git commit -m "feat(eval): add evaluate_popia() harness with release gate"
```

---

### Task 9: `POPIAJudge` class — test first

**Files:**
- Create: `semantix/judges/popia.py`
- Test: `semantix/tests/test_popia_judge.py`

- [ ] **Step 1: Write the failing tests**

Create `semantix/tests/test_popia_judge.py`:

```python
"""Unit tests for POPIAJudge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from semantix.judges import Judge, Verdict


def _fake_logits():
    # (entailment, neutral, contradiction) -> softmax picks entailment
    import numpy as np
    return np.array([[3.0, 0.5, 0.2]], dtype=np.float32)


@pytest.fixture
def mocked_onnx(monkeypatch):
    """Patch HF download + ONNX runtime so no network or model file is needed."""
    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]
    fake_session.get_inputs.return_value[0].name = "input_ids"
    fake_session.get_inputs.return_value[1].name = "attention_mask"
    fake_session.run.return_value = [_fake_logits()]

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value.ids = [1, 2, 3]
    fake_tokenizer.encode.return_value.attention_mask = [1, 1, 1]

    monkeypatch.setattr(
        "semantix.judges.quantized_nli.hf_hub_download",
        lambda repo_id, filename, **kw: f"/tmp/fake-{repo_id.replace('/','_')}-{filename.replace('/','_')}",
        raising=False,
    )
    monkeypatch.setattr(
        "semantix.judges.quantized_nli._load_session",
        lambda variant, repo_id=None: (fake_session, fake_tokenizer),
    )
    yield fake_session


def test_repo_id_is_popia(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    assert POPIAJudge._REPO_ID == "labrat-akhona/nli-popia-v1"


def test_recommended_threshold_is_pinned(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    assert POPIAJudge.recommended_threshold == 0.75


def test_clauses_returns_seven_canonical_strings(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    clauses = POPIAJudge.clauses()
    assert len(clauses) == 7
    assert all(isinstance(c, str) and c.startswith("POPIA") for c in clauses)
    assert len(set(clauses)) == 7  # no duplicates


def test_popia_judge_is_subclass_of_quantized_and_base(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    from semantix.judges.quantized_nli import QuantizedNLIJudge
    j = POPIAJudge()
    assert isinstance(j, QuantizedNLIJudge)
    assert isinstance(j, Judge)


def test_evaluate_delegates_and_returns_verdict(mocked_onnx):
    from semantix.judges.popia import POPIAJudge
    j = POPIAJudge()
    v = j.evaluate("some output", "some intent", threshold=0.5)
    assert isinstance(v, Verdict)
    assert v.passed is True  # fake logits favor entailment


def test_download_failure_raises_runtime_error_no_silent_fallback(monkeypatch):
    def boom(variant, repo_id=None):
        raise RuntimeError("HF unreachable (simulated)")
    monkeypatch.setattr("semantix.judges.quantized_nli._load_session", boom)

    from semantix.judges.popia import POPIAJudge
    with pytest.raises(RuntimeError, match="HF unreachable"):
        POPIAJudge()
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest semantix/tests/test_popia_judge.py -v`
Expected: 6 failures, all `ModuleNotFoundError: semantix.judges.popia`.

- [ ] **Step 3: Implement `semantix/judges/popia.py`**

Create `semantix/judges/popia.py`:

```python
"""POPIA-fine-tuned NLI judge.

Loads the `labrat-akhona/nli-popia-v1` quantized ONNX model from HuggingFace.
Inherits all inference, CPU variant detection, and caching logic from
QuantizedNLIJudge -- this subclass only overrides the model identity.

Requires: pip install semantix-ai[popia]  (installs the same deps as [turbo])
"""

from __future__ import annotations

from typing import ClassVar

from semantix.judges.quantized_nli import QuantizedNLIJudge


class POPIAJudge(QuantizedNLIJudge):
    """Semantic judge fine-tuned on POPIA (Protection of Personal Information Act).

    Use this in place of :class:`QuantizedNLIJudge` when validating outputs
    against POPIA-specific intents. Paired with presets from
    :mod:`semantix.presets.popia`.

    The threshold 0.75 is pinned from held-out calibration. Override via the
    ``threshold`` parameter on :meth:`evaluate` if stricter or looser matching
    is needed for a specific intent.
    """

    _REPO_ID: ClassVar[str] = "labrat-akhona/nli-popia-v1"
    recommended_threshold: ClassVar[float | None] = 0.75

    @classmethod
    def clauses(cls) -> list[str]:
        """Return the POPIA concept labels the model was trained on.

        Canonical values -- preset .clause attributes in
        :mod:`semantix.presets.popia` match these 1:1.
        """
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

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest semantix/tests/test_popia_judge.py -v`
Expected: 6 pass.

- [ ] **Step 5: Commit**

```bash
git add semantix/judges/popia.py semantix/tests/test_popia_judge.py
git commit -m "feat(judges): add POPIAJudge subclass loading nli-popia-v1"
```

---

### Task 10: POPIA presets — test first

**Files:**
- Create: `semantix/presets/popia.py`
- Test: `semantix/tests/test_popia_presets.py`

- [ ] **Step 1: Write the failing tests**

Create `semantix/tests/test_popia_presets.py`:

```python
"""Unit tests for semantix.presets.popia."""

from __future__ import annotations

import pytest

from semantix.intent import Intent
from semantix.judges.popia import POPIAJudge


def _all_presets():
    from semantix.presets import popia as m
    return [getattr(m, n) for n in m.__all__]


def test_all_presets_are_intents():
    for preset in _all_presets():
        assert isinstance(preset, Intent)


def test_all_presets_have_nonempty_description():
    for preset in _all_presets():
        assert preset.description
        assert len(preset.description) > 10


def test_all_presets_have_clause_attribute_matching_judge():
    canonical = set(POPIAJudge.clauses())
    for preset in _all_presets():
        assert hasattr(preset, "clause")
        assert preset.clause in canonical, f"{preset.clause!r} not in canonical"


def test_breach_preset_is_negated():
    from semantix.presets.popia import POPIA_BREACH
    assert POPIA_BREACH.negate is True


def test_non_breach_presets_are_not_negated():
    from semantix.presets import popia as m
    for name in m.__all__:
        if name == "POPIA_BREACH":
            continue
        preset = getattr(m, name)
        assert preset.negate is False, f"{name} should not be negated"


def test_security_preset_has_stricter_threshold():
    from semantix.presets.popia import POPIA_SECURITY
    assert POPIA_SECURITY.threshold is not None
    assert 0.8 < POPIA_SECURITY.threshold <= 0.95


def test_all_thresholds_in_valid_range():
    for preset in _all_presets():
        if preset.threshold is not None:
            assert 0.5 <= preset.threshold <= 0.95


def test_preset_count_matches_clause_count():
    from semantix.presets import popia as m
    assert len(m.__all__) == len(POPIAJudge.clauses())


def test_preset_clauses_cover_all_judge_clauses():
    covered = {getattr(__import__("semantix.presets.popia", fromlist=[n]), n).clause for n in __import__("semantix.presets.popia", fromlist=["__all__"]).__all__}
    assert covered == set(POPIAJudge.clauses())
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest semantix/tests/test_popia_presets.py -v`
Expected: 9 failures, all `ModuleNotFoundError: semantix.presets.popia`.

- [ ] **Step 3: Implement `semantix/presets/popia.py`**

Create `semantix/presets/popia.py`:

```python
"""Pre-built Intent presets for POPIA (Protection of Personal Information Act).

Each preset is a module-level :class:`~semantix.intent.Intent` instance anchored
to one of the POPIA concept labels in :meth:`POPIAJudge.clauses`. Use with
:class:`~semantix.judges.popia.POPIAJudge`.

These presets encode the author's engineering reading of POPIA for the
purpose of automated output validation. They are not legal advice. Consult
your DPIA and legal counsel for compliance determinations.
"""

from __future__ import annotations

from semantix.intent import Intent

__all__ = [
    "POPIA_CONSENT",
    "POPIA_MINIMALITY",
    "POPIA_SECURITY",
    "POPIA_BREACH",
    "POPIA_CROSS_BORDER",
    "POPIA_PROCESSING",
    "POPIA_DATA_SUBJECT_RIGHTS",
]


POPIA_CONSENT = Intent(
    description=(
        "The output obtains or confirms explicit, informed, voluntary consent "
        "from the data subject before processing their personal information."
    ),
    clause="POPIA consent",
    negate=False,
)

POPIA_MINIMALITY = Intent(
    description=(
        "The output processes personal information only to the extent necessary "
        "for a specific, explicit, lawful purpose and does not use it for "
        "incompatible secondary purposes."
    ),
    clause="POPIA minimality / purpose limitation",
    negate=False,
)

POPIA_SECURITY = Intent(
    description=(
        "The output describes appropriate, reasonable technical and "
        "organisational security safeguards protecting personal information "
        "against loss, damage, unauthorised access, or unlawful processing."
    ),
    clause="POPIA security safeguards",
    negate=False,
    threshold=0.85,
)

POPIA_BREACH = Intent(
    description=(
        "The output delays, minimises, or fails to notify affected data subjects "
        "and the Information Regulator after a security compromise involving "
        "personal information."
    ),
    clause="POPIA breach notification",
    negate=True,
)

POPIA_CROSS_BORDER = Intent(
    description=(
        "The output transfers personal information to a recipient in a foreign "
        "country without ensuring adequate protection equivalent to POPIA, "
        "data subject consent, or another lawful basis for the transfer."
    ),
    clause="POPIA cross-border transfers",
    negate=False,
)

POPIA_PROCESSING = Intent(
    description=(
        "The output processes personal information lawfully, in a reasonable "
        "manner that does not infringe the privacy of the data subject."
    ),
    clause="POPIA general processing",
    negate=False,
)

POPIA_DATA_SUBJECT_RIGHTS = Intent(
    description=(
        "The output respects the data subject's rights to access, correct, "
        "delete, or object to the processing of their personal information "
        "and responds to such requests within a reasonable time."
    ),
    clause="POPIA data subject rights",
    negate=False,
)
```

- [ ] **Step 4: Check `Intent` accepts the `clause` kwarg**

If `Intent.__init__` does not already accept a `clause` kwarg, this step blocks and we must add it. Run:
```bash
python -c "from semantix.intent import Intent; i = Intent(description='x', clause='y'); print(i.clause)"
```
Expected: prints `y`. If it raises `TypeError: got unexpected kwarg 'clause'`, proceed to Step 5.

- [ ] **Step 5: Extend `Intent` to accept `clause` (only if Step 4 failed)**

Edit `semantix/intent.py` to add `clause: str | None = None` to the `Intent` dataclass/constructor. One new optional field, no behavioural change — existing intents still work. After editing, re-run Step 4 and confirm it prints `y`.

```bash
git add semantix/intent.py
git commit -m "feat(intent): add optional clause attribute for preset categorisation"
```

- [ ] **Step 6: Run tests — expect pass**

Run: `pytest semantix/tests/test_popia_presets.py -v`
Expected: 9 pass.

- [ ] **Step 7: Commit**

```bash
git add semantix/presets/popia.py semantix/tests/test_popia_presets.py
git commit -m "feat(presets): add POPIA preset intents"
```

---

### Task 11: Conditional re-export of `POPIAJudge` from top-level

**Files:**
- Modify: `semantix/__init__.py`

- [ ] **Step 1: Locate existing conditional re-exports**

Run: `grep -n "try:" semantix/__init__.py | head -10`
Expected: at least one `try:` block for a conditional import (e.g. `QuantizedNLIJudge`).

- [ ] **Step 2: Add POPIAJudge re-export after existing pattern**

Find the existing QuantizedNLIJudge re-export block. Immediately after it, add:

```python
try:
    from semantix.judges.popia import POPIAJudge  # noqa: F401
    __all__ = list(__all__) + ["POPIAJudge"]
except ImportError:
    pass
```

- [ ] **Step 3: Verify the import works**

Run: `python -c "from semantix import POPIAJudge; print(POPIAJudge._REPO_ID)"`
Expected: `labrat-akhona/nli-popia-v1`.

- [ ] **Step 4: Commit**

```bash
git add semantix/__init__.py
git commit -m "feat: re-export POPIAJudge at top level when [popia] extras installed"
```

---

### Task 12: Update `pyproject.toml` with `[popia]` and `[train]` extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add extras**

In `[project.optional-dependencies]`, immediately after the existing `turbo` line, add:

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

Then update the `all` extra to include `popia` transitively (no-op since `popia` just re-points to `turbo`, which is already in `all`; no change needed).

- [ ] **Step 2: Verify extras resolve**

Run: `pip install -e ".[popia]" --dry-run 2>&1 | tail -5`
Expected: no resolution errors; lists `onnxruntime`, `tokenizers`, `huggingface-hub` as deps.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(pyproject): add [popia] and [train] extras groups"
```

---

## Phase 3: CLI integration

### Task 13: `semantix eval popia` CLI subcommand — test first

**Files:**
- Modify: `semantix/cli.py`
- Test: `semantix/tests/test_cli_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `semantix/tests/test_cli_eval.py`:

```python
"""Unit tests for `semantix eval popia`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from semantix.cli import main as cli_main
from semantix.eval.popia import EvalReport


def _fake_report(gate: bool, delta: float = 0.15) -> EvalReport:
    return EvalReport(
        n_pairs=150,
        stock_accuracy=0.62,
        stock_f1_macro=0.59,
        popia_accuracy=0.78,
        popia_f1_macro=0.59 + delta,
        per_clause={
            "POPIA consent": (0.60, 0.75),
            "POPIA cross-border transfers": (0.55, 0.82),
        },
        delta_f1=delta,
        release_gate_passed=gate,
    )


def test_eval_popia_exits_zero_when_gate_passes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(True))

    rc = cli_main(["eval", "popia"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_eval_popia_exits_one_when_gate_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(False, delta=0.05))

    rc = cli_main(["eval", "popia"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_eval_popia_json_flag_emits_valid_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("semantix.cli._download_popia_eval", lambda: tmp_path / "eval.jsonl")
    monkeypatch.setattr("semantix.cli._load_popia_judges", lambda: (object(), object()))
    monkeypatch.setattr("semantix.cli.evaluate_popia", lambda *a, **k: _fake_report(True))

    rc = cli_main(["eval", "popia", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["release_gate_passed"] is True
    assert data["n_pairs"] == 150
    assert data["delta_f1"] == 0.15


def test_eval_popia_download_failure_exits_two(capsys, monkeypatch):
    def boom():
        raise FileNotFoundError("HF unreachable")
    monkeypatch.setattr("semantix.cli._download_popia_eval", boom)

    rc = cli_main(["eval", "popia"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unreachable" in err.lower() or "not found" in err.lower()
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest semantix/tests/test_cli_eval.py -v`
Expected: 4 failures, most likely `argparse: invalid choice: 'eval'` or similar.

- [ ] **Step 3: Add `eval popia` to `semantix/cli.py`**

Read the current structure of `semantix/cli.py` to find the subparser setup (look for `add_subparsers`, `subparsers.add_parser("verify", ...)` etc.). Add immediately after the existing subparser definitions:

```python
# --- eval subcommand ---
def _download_popia_eval():
    """Download eval.jsonl from HF and return the local cached path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="labrat-akhona/nli-popia-v1",
        filename="eval.jsonl",
    )


def _load_popia_judges():
    """Instantiate POPIAJudge and stock QuantizedNLIJudge."""
    from semantix.judges.popia import POPIAJudge
    from semantix.judges.quantized_nli import QuantizedNLIJudge
    return POPIAJudge(), QuantizedNLIJudge()


def _run_eval_popia(args) -> int:
    from dataclasses import asdict
    from semantix.eval.popia import evaluate_popia

    try:
        eval_path = _download_popia_eval()
    except Exception as e:
        print(f"failed to download eval set: {e}", file=sys.stderr)
        return 2

    popia, stock = _load_popia_judges()
    report = evaluate_popia(eval_path, popia, stock)

    if args.json:
        # asdict handles nested tuples in per_clause as lists -> JSON-safe
        out = asdict(report)
        out["per_clause"] = {k: list(v) for k, v in out["per_clause"].items()}
        print(json.dumps(out, indent=2))
    else:
        print(f"\n                    stock    POPIA    Delta")
        print(f"Accuracy           {report.stock_accuracy:.2f}     {report.popia_accuracy:.2f}     {report.popia_accuracy - report.stock_accuracy:+.2f}")
        print(f"F1 (macro)         {report.stock_f1_macro:.2f}     {report.popia_f1_macro:.2f}     {report.delta_f1:+.2f}")
        for clause, (stock_f, popia_f) in report.per_clause.items():
            print(f"  {clause:<30} {stock_f:.2f}     {popia_f:.2f}     {popia_f - stock_f:+.2f}")
        verdict = "PASS" if report.release_gate_passed else "FAIL"
        print(f"\nRelease gate (>= 0.10 F1 delta, no per-clause regression): {verdict}")

    return 0 if report.release_gate_passed else 1
```

Then add the subparser registration where other subparsers are registered:

```python
eval_parser = subparsers.add_parser("eval", help="Run release-gate evaluations.")
eval_sub = eval_parser.add_subparsers(dest="eval_target", required=True)
popia_eval = eval_sub.add_parser("popia", help="Evaluate POPIAJudge vs stock NLI.")
popia_eval.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
popia_eval.set_defaults(func=_run_eval_popia)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest semantix/tests/test_cli_eval.py -v`
Expected: 4 pass.

- [ ] **Step 5: Smoke-test the CLI help**

Run: `python -m semantix.cli eval popia --help`
Expected: help text showing `--json` flag. No error.

- [ ] **Step 6: Commit**

```bash
git add semantix/cli.py semantix/tests/test_cli_eval.py
git commit -m "feat(cli): add `semantix eval popia` release-gate subcommand"
```

---

### Task 14: Dev-only wrapper script `scripts/eval_popia.py`

**Files:**
- Create: `scripts/eval_popia.py`

- [ ] **Step 1: Write the wrapper script**

Create `scripts/eval_popia.py`:

```python
"""Reproducibility wrapper: run POPIA eval from the local data/popia_eval.jsonl.

Unlike `semantix eval popia` which downloads eval.jsonl from HF, this script
uses the exact file in the repo -- useful for developers validating a
freshly-trained model before uploading.

Usage:
    python scripts/eval_popia.py                        # uses local out/ model if present
    python scripts/eval_popia.py --use-hf               # downloads from HF instead
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from semantix.eval.popia import evaluate_popia
from semantix.judges.popia import POPIAJudge
from semantix.judges.quantized_nli import QuantizedNLIJudge

LOCAL_EVAL = Path("data/popia_eval.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hf", action="store_true", help="Use HF eval.jsonl instead of local.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.use_hf:
        from huggingface_hub import hf_hub_download
        eval_path = hf_hub_download(repo_id="labrat-akhona/nli-popia-v1", filename="eval.jsonl")
    else:
        if not LOCAL_EVAL.exists():
            print(f"missing {LOCAL_EVAL}", file=sys.stderr)
            return 2
        eval_path = LOCAL_EVAL

    report = evaluate_popia(eval_path, POPIAJudge(), QuantizedNLIJudge())

    if args.json:
        import json
        out = asdict(report)
        out["per_clause"] = {k: list(v) for k, v in out["per_clause"].items()}
        print(json.dumps(out, indent=2))
    else:
        print(f"n_pairs={report.n_pairs}")
        print(f"stock F1={report.stock_f1_macro:.3f}  POPIA F1={report.popia_f1_macro:.3f}  delta={report.delta_f1:+.3f}")
        print(f"gate: {'PASS' if report.release_gate_passed else 'FAIL'}")

    return 0 if report.release_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the script (will fail without a real model -- that's ok)**

Run: `python scripts/eval_popia.py 2>&1 | head -3`
Expected: either a report (if model is accessible) or an HF download error (if not). Not a syntax error.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_popia.py
git commit -m "feat(train): add POPIA eval reproducibility wrapper"
```

---

## Phase 4: Integration tests + CI

### Task 15: Integration test infrastructure

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_popia_end_to_end.py`
- Modify: `pyproject.toml` (add `integration` marker)

- [ ] **Step 1: Register the `integration` pytest marker**

In `pyproject.toml`, find or add the `[tool.pytest.ini_options]` section and ensure it contains:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: tests that require network access and real HF downloads (deselect with '-m \"not integration\"')",
]
addopts = "-m 'not integration'"
```

If a `[tool.pytest.ini_options]` section already exists, merge these entries into it.

- [ ] **Step 2: Create the integration test directory**

Run: `mkdir -p tests/integration && touch tests/integration/__init__.py`

- [ ] **Step 3: Write the integration test**

Create `tests/integration/test_popia_end_to_end.py`:

```python
"""End-to-end integration test: download real POPIAJudge and validate semantics.

This test:
  - downloads the real labrat-akhona/nli-popia-v1 model from HuggingFace
  - runs 3 POPIA presets against hand-chosen outputs
  - verifies verdicts match the expected POPIA semantics

Runs only when pytest is invoked with `-m integration`. Excluded from the
default suite because it requires network access and ~25MB of downloads.
"""

from __future__ import annotations

import pytest

from semantix import validate
from semantix.judges.popia import POPIAJudge
from semantix.presets.popia import (
    POPIA_CONSENT,
    POPIA_CROSS_BORDER,
    POPIA_BREACH,
)


@pytest.fixture(scope="module")
def popia_judge():
    return POPIAJudge()


@pytest.mark.integration
def test_consent_positive(popia_judge):
    output = "I confirm I have read and agree to the privacy terms."
    v = validate(output=output, intent=POPIA_CONSENT, judge=popia_judge)
    assert v.passed is True


@pytest.mark.integration
def test_cross_border_negative_detected(popia_judge):
    output = "Data is stored exclusively in eu-central-1 and streamed via us-east-1."
    v = validate(output=output, intent=POPIA_CROSS_BORDER, judge=popia_judge)
    # cross-border transfer is asserted by the output -> intent matches -> passed=True
    assert v.passed is True


@pytest.mark.integration
def test_breach_negated_intent_fires_on_delayed_notification(popia_judge):
    output = "We'll notify affected users in the next quarterly newsletter."
    v = validate(output=output, intent=POPIA_BREACH, judge=popia_judge)
    # Intent is negated ("must NOT delay notification"). Output DOES delay,
    # so the negated intent matches -> passed=False (violation).
    assert v.passed is False
```

- [ ] **Step 4: Verify integration tests are excluded by default**

Run: `pytest --collect-only semantix/tests/ tests/ 2>&1 | tail -5`
Expected: integration tests appear as deselected. The default suite count does not include them.

- [ ] **Step 5: Verify integration tests can be opted into**

Run (only if a real model is uploaded to HF): `pytest -m integration tests/integration/ -v`
Expected: 3 tests pass (or marked `xfail` if model not yet uploaded).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_popia_end_to_end.py pyproject.toml
git commit -m "test(integration): add POPIA end-to-end smoke test (opt-in marker)"
```

---

### Task 16: GitHub Actions release-gate workflow

**Files:**
- Create: `.github/workflows/popia-eval.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/popia-eval.yml`:

```yaml
name: POPIA release gate

on:
  push:
    tags:
      - 'v*-popia'
    paths:
      - 'semantix/judges/popia.py'
      - 'semantix/presets/popia.py'
      - 'semantix/eval/popia.py'
  pull_request:
    paths:
      - 'semantix/judges/popia.py'
      - 'semantix/presets/popia.py'
      - 'semantix/eval/popia.py'
  workflow_dispatch:

jobs:
  eval:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache HuggingFace hub
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface/hub
          key: hf-${{ runner.os }}-nli-popia-v1

      - name: Install
        run: pip install -e ".[popia]"

      - name: Run release gate
        run: |
          python -m semantix.cli eval popia --json | tee report.json
          python -c "import json; r=json.load(open('report.json')); import sys; sys.exit(0 if r['release_gate_passed'] else 1)"

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: popia-eval-report
          path: report.json
```

- [ ] **Step 2: Verify workflow syntax locally (no execution)**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/popia-eval.yml'))"`
Expected: no output, no error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/popia-eval.yml
git commit -m "ci: add POPIA release-gate workflow enforcing 10pp F1 delta"
```

---

## Phase 5: Release

### Task 17: Upload model artifacts to HuggingFace

**MANUAL release task. Requires a trained model from Task 7.**

- [ ] **Step 1: Log in to HuggingFace CLI**

Run:
```bash
pip install huggingface_hub[cli]
huggingface-cli login
# paste your HF write token
```
Expected: `Login successful`.

- [ ] **Step 2: Create the model repo**

Run:
```bash
huggingface-cli repo create nli-popia-v1 --type model
```
Expected: repo URL printed.

- [ ] **Step 3: Author the model card**

Create `out/nli-popia-v1/README.md`:

```markdown
---
license: apache-2.0
language: en
tags:
  - nli
  - natural-language-inference
  - popia
  - compliance
  - south-africa
datasets:
  - labrat-akhona/popia-nli-seeds
base_model: cross-encoder/nli-MiniLM2-L6-H768
---

# nli-popia-v1

Fine-tune of `cross-encoder/nli-MiniLM2-L6-H768` on a hand-authored POPIA
(Protection of Personal Information Act, South Africa, Act 4 of 2013)
natural-language-inference dataset.

## Intended use

Drop-in replacement for the base model when validating LLM outputs against
POPIA-specific intents. Use via the [semantix-ai](https://github.com/labrat-akhona/semantix-ai)
library:

```python
from semantix import validate
from semantix.judges.popia import POPIAJudge
from semantix.presets.popia import POPIA_CROSS_BORDER

v = validate(
    output="We store all user data in eu-central-1.",
    intent=POPIA_CROSS_BORDER,
    judge=POPIAJudge(),
)
```

## Training data

- 60 hand-authored seed NLI pairs covering 7 POPIA concepts
  ([source in repo](https://github.com/labrat-akhona/semantix-ai/blob/master/data/popia_seeds.jsonl))
- ~1500 synthetic variants via LLM paraphrase with stock-NLI sanity filter
- 150 hand-labeled held-out eval pairs (**never seen during training**)

## Evaluation

Held-out eval set is bundled in this repo as `eval.jsonl`. Reproduce locally:

```bash
pip install semantix-ai[popia]
semantix eval popia
```

Release-gate requirement (enforced in CI): macro-F1 delta >= 10pp over base,
no per-clause regression.

## Limitations

- Engineering heuristic, not legal advice. Consult your DPIA and legal counsel.
- Trained on English POPIA-flavored data. Afrikaans and isiZulu not covered.
- 150-pair eval set is tight; confidence intervals are wide. Gate threshold
  of 10pp is deliberately outside noise at that n.

## License

Apache 2.0 on the model weights. MIT on semantix-ai integration code.
CC-BY-4.0 on seed and eval datasets.
```

- [ ] **Step 4: Upload the artifacts**

Run:
```bash
cd out/nli-popia-v1
huggingface-cli upload labrat-akhona/nli-popia-v1 . --repo-type model
```
Expected: each file uploaded, final "Upload complete" message.

- [ ] **Step 5: Verify the model is live and cached**

Run:
```bash
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('labrat-akhona/nli-popia-v1', 'model_quint8_avx2.onnx')
print(f'downloaded to {p}')
"
```
Expected: a local cache path printed.

- [ ] **Step 6: Commit the model card source (NOT the ONNX files)**

```bash
git add docs/model-card-popia-v1.md  # copy the README content here for repo-side reference
# out/ is already gitignored
git commit -m "docs: mirror HuggingFace model card for nli-popia-v1"
```

---

### Task 18: Run the release gate end-to-end and cut the release

- [ ] **Step 1: Run the release gate against the live HF model**

Run:
```bash
pip install -e ".[popia]"
semantix eval popia
```
Expected: table output with `Release gate (>= 0.10 F1 delta, no per-clause regression): PASS`. Exit code 0.

- [ ] **Step 2: If gate fails, DO NOT proceed**

If the gate fails, return to Task 3-7 to improve seeds/expansion/training. Do **not** relax the gate to match achieved delta. Per spec: iterate, do not ship softer.

- [ ] **Step 3: Bump version**

In `pyproject.toml`, bump `version` to the next patch or minor (e.g. `0.1.13` -> `0.2.0` if this is a minor feature release).

- [ ] **Step 4: Update README with POPIA quickstart**

Add a section to `README.md` under existing integration sections:

```markdown
## POPIA support (South Africa)

Validate LLM outputs against POPIA (Protection of Personal Information Act):

    pip install semantix-ai[popia]

```python
from semantix import validate
from semantix.judges.popia import POPIAJudge
from semantix.presets.popia import POPIA_CROSS_BORDER

v = validate(
    output="All user data is stored in eu-central-1.",
    intent=POPIA_CROSS_BORDER,
    judge=POPIAJudge(),
)
```

Reproduce the release claim that POPIAJudge beats base NLI by >=10 pp F1:

    semantix eval popia

See [docs/popia.md](docs/popia.md) for preset reference.
```

- [ ] **Step 5: Commit and tag**

```bash
git add pyproject.toml README.md
git commit -m "release: v0.2.0 — POPIA fine-tune support"
git tag v0.2.0-popia
git push origin master v0.2.0-popia
```

Expected: CI workflow `POPIA release gate` triggers and passes.

- [ ] **Step 6: Publish to PyPI**

Run:
```bash
rm -rf dist/
python -m build
python -m twine upload dist/*
```
Expected: uploaded to PyPI.

- [ ] **Step 7: Verify install from PyPI works**

Run:
```bash
pip install --upgrade --force-reinstall "semantix-ai[popia]"
python -c "from semantix import POPIAJudge; from semantix.presets.popia import POPIA_CROSS_BORDER; print('ok')"
```
Expected: `ok`.

---

## Post-release checklist

Not part of the implementation plan, but track after shipping:
- [ ] Draft announcement blog post at `articles/drafts/popia-finetune.md` citing only numbers from the latest CI `report.json`
- [ ] Update `docs/` mkdocs site with a POPIA page linking to presets and model card
- [ ] Add a PR to any awesome-list where POPIA coverage is a differentiator (compliance lists, SA-focused tooling lists)
- [ ] File the POPIA-v1 release as a reference memory entry

---

## Summary

**Phases:** 5
**Tasks:** 18
**Runtime code added:** ~200 LOC across 4 new files
**Test code added:** ~23 unit tests + 3 integration tests
**Dev-only code added:** 3 scripts + 2 dataset files
**Manual human effort:** ~3 days (seed authoring + eval authoring + HF upload + release)
**Compute:** ~30 min GPU or ~3 hours CPU for training

**Key invariants enforced:**
1. Eval set hash is pinned; training aborts if eval set changes without a visible commit
2. Release gate: >=10pp F1 delta and no per-clause regression
3. No silent fallback: missing HF model raises RuntimeError, never silently degrades to base NLI
4. Public claims limited to the most recent CI `report.json`
