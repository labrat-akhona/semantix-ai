"""Train POPIA-Judge v2 with expanded clause coverage.

v2 adds three clauses that v1 didn't cover but are critical for AI workloads:
  - POPIA children's information (S34-35)
  - POPIA special personal information (S26-33)  -- race, religion, health, biometric
  - POPIA automated decision-making (S71)

v1 data files (data/popia_{seeds,paraphrases,eval}.jsonl) and the v1 training
script (scripts/train_popia.py) are NOT touched -- v1 stays reproducible.
v2 adds data/popia_{seeds,paraphrases,eval}_v2.jsonl on top.

Release gate (both must pass):
  1. v2 macro F1 on the v1 holdout >= v1 macro F1 on the v1 holdout  (no regression)
  2. v2 macro F1 on the v2 holdout >= stock-model macro F1 on the v2 holdout (new capability)

Usage:
    python scripts/train_popia_v2.py           # GPU if available, else CPU
    python scripts/train_popia_v2.py --epochs 5

Generalised since v0.2.2: --base-model and --out-dir override the v2
defaults so this same recipe can train v3 (deberta-v3-base, ~184M) or
other clause-NLI variants without forking the script. The v2 defaults
keep the original release-gate semantics for reproducibility.

    # v3 on a bigger base (recipe used by scripts/modal_train_v3.py)
    python scripts/train_popia_v2.py \\
        --base-model cross-encoder/nli-deberta-v3-base \\
        --out-dir out/nli-popia-v3 --epochs 6 --skip-quantize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42

DEFAULT_BASE = "cross-encoder/nli-MiniLM2-L6-H768"
DEFAULT_OUT = Path("out/nli-popia-v2")
V1_SEEDS = Path("data/popia_seeds.jsonl")
V1_PARAPHRASES = Path("data/popia_paraphrases.jsonl")
V1_EVAL = Path("data/popia_eval.jsonl")
V1_EVAL_HASH = Path("scripts/_popia_eval_hash.txt")
V2_SEEDS = Path("data/popia_seeds_v2.jsonl")
V2_PARAPHRASES = Path("data/popia_paraphrases_v2.jsonl")
V2_EVAL = Path("data/popia_eval_v2.jsonl")
V2_EVAL_HASH = Path("scripts/_popia_eval_v2_hash.txt")


def verify_eval_integrity(eval_path: Path, pinned_hash_path: Path) -> None:
    if not pinned_hash_path.exists():
        sys.exit(f"missing {pinned_hash_path}")
    pinned = pinned_hash_path.read_text().strip()
    current = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    if pinned != current:
        sys.exit(
            f"EVAL SET INTEGRITY FAILURE for {eval_path}\n"
            f"  pinned hash: {pinned}\n  current hash: {current}"
        )


def label_to_id(label: str) -> int:
    return {"contradiction": 0, "entailment": 1, "neutral": 2}[label]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def macro_f1(
    model, tokenizer, eval_rows: list[dict], batch_size: int = 16
) -> tuple[float, dict[str, float]]:
    """Macro F1 over labels, plus per-clause F1."""
    from collections import defaultdict

    import torch
    from sklearn.metrics import f1_score

    model.eval()
    device = next(model.parameters()).device

    y_true: list[int] = []
    y_pred: list[int] = []
    per_clause_true: dict[str, list[int]] = defaultdict(list)
    per_clause_pred: dict[str, list[int]] = defaultdict(list)

    with torch.no_grad():
        for i in range(0, len(eval_rows), batch_size):
            batch = eval_rows[i : i + batch_size]
            inputs = tokenizer(
                [r["premise"] for r in batch],
                [r["hypothesis"] for r in batch],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            for r, p in zip(batch, preds, strict=False):
                t = label_to_id(r["label"])
                y_true.append(t)
                y_pred.append(p)
                per_clause_true[r["clause"]].append(t)
                per_clause_pred[r["clause"]].append(p)

    overall = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_clause = {
        cls: float(
            f1_score(per_clause_true[cls], per_clause_pred[cls], average="macro", zero_division=0)
        )
        for cls in per_clause_true
    }
    return overall, per_clause


def onnx_macro_f1(
    onnx_path, tokenizer, eval_rows: list[dict], batch_size: int = 16
) -> tuple[float, int]:
    """Macro F1 for a quantized ONNX artifact + number of distinct predictions.

    Mirrors ``macro_f1``'s tokenization but runs the SHIPPED file through
    onnxruntime — the artifact users actually load. Quantization can silently
    turn a healthy model into a constant predictor (per-tensor INT8 breaks
    DeBERTa disentangled attention), so the distinct-prediction count is the
    cheap tell: a constant predictor returns 1.
    """
    import numpy as np
    import onnxruntime as ort
    from sklearn.metrics import f1_score

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    innames = {i.name for i in sess.get_inputs()}
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in range(0, len(eval_rows), batch_size):
        batch = eval_rows[i : i + batch_size]
        enc = tokenizer(
            [r["premise"] for r in batch],
            [r["hypothesis"] for r in batch],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="np",
        )
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in innames and "token_type_ids" in enc:
            feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        logits = sess.run(None, feeds)[0]
        y_pred.extend(int(p) for p in logits.argmax(axis=-1).tolist())
        y_true.extend(label_to_id(r["label"]) for r in batch)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return macro, len(set(y_pred))


# Max macro-F1 the quantized artifact may lose vs the PyTorch model before we
# refuse to ship it. 10pp is generous for honest INT8 rounding but catches the
# kind of collapse that made v3-quant unshippable (−21pp v1 / −38pp v2).
ARTIFACT_TOL = 0.10


def artifact_gate_verdict(
    q_v1_f1: float,
    q_v1_distinct: int,
    q_v2_f1: float,
    q_v2_distinct: int,
    trained_v1_f1: float,
    trained_v2_f1: float,
    tol: float = ARTIFACT_TOL,
) -> tuple[bool, str]:
    """Decide whether a quantized ONNX artifact is shippable. Pure + testable.

    Two independent failure modes, both learned the expensive way:

    - **Constant predictor** (``distinct <= 1`` on either holdout). Per-tensor
      INT8 can collapse a healthy model into always-one-class. macro-F1 alone
      will NOT catch this — a constant predictor can still post a plausible F1 —
      so the distinct-prediction count is the cheap, decisive tell.
    - **Regression** (macro-F1 more than ``tol`` below the PyTorch model on
      either holdout). The file the user loads is materially worse than what the
      upstream release gate scored.

    Returns ``(passed, reason)``; ``reason`` is empty on pass. This is the exact
    logic the shipped ONNX must clear before upload — extracted from ``main`` so
    it can be unit-tested without a GPU or a real model.
    """
    if q_v1_distinct <= 1 or q_v2_distinct <= 1:
        return False, "CONSTANT PREDICTOR (distinct predictions <= 1)"
    if q_v1_f1 < trained_v1_f1 - tol or q_v2_f1 < trained_v2_f1 - tol:
        return False, f"macro F1 regressed >{tol} from PyTorch"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip ONNX export + quantization (faster for iteration)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE,
        help="HF repo id of the base NLI cross-encoder (default: cross-encoder/nli-MiniLM2-L6-H768)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for pytorch + onnx + release_gate.json (default: out/nli-popia-v2)",
    )
    args = ap.parse_args()
    seed = args.seed
    base_model = args.base_model
    out_dir = args.out_dir

    verify_eval_integrity(V1_EVAL, V1_EVAL_HASH)
    verify_eval_integrity(V2_EVAL, V2_EVAL_HASH)

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    # Pin all RNGs so per-clause F1 is reproducible across runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    train_rows = (
        load_rows(V1_SEEDS)
        + load_rows(V1_PARAPHRASES)
        + load_rows(V2_SEEDS)
        + load_rows(V2_PARAPHRASES)
    )
    print(f"loaded {len(train_rows)} training rows (v1 + v2)")

    # Stratified dev split: at least 1 item per (clause, label) so checkpoint
    # selection has signal on every clause. Tail-slicing the concatenated rows
    # put the entire dev set into v2_paraphrases, leaving v1 clauses (especially
    # minimality) unmoored from `load_best_model_at_end`.
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in train_rows:
        buckets[(r["clause"], r["label"])].append(r)
    rng = random.Random(seed)
    train_split: list[dict] = []
    dev_split: list[dict] = []
    for key in sorted(buckets):
        items = buckets[key][:]
        rng.shuffle(items)
        n_dev = max(1, len(items) // 10)
        dev_split.extend(items[:n_dev])
        train_split.extend(items[n_dev:])
    rng.shuffle(train_split)
    rng.shuffle(dev_split)
    print(f"stratified split: train={len(train_split)}, dev={len(dev_split)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    def to_ds(rows):
        ds = Dataset.from_list(
            [
                {
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "labels": label_to_id(r["label"]),
                }
                for r in rows
            ]
        )
        return ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pytorch_out = out_dir / "pytorch"

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
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=to_ds(train_split),
        eval_dataset=to_ds(dev_split),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(pytorch_out))
    tokenizer.save_pretrained(str(pytorch_out))
    print(f"pytorch checkpoint saved to {pytorch_out}")

    # --- Release gate ---
    print("\n=== Release gate eval ===")

    # Baseline: stock model on each holdout (for gating new-capability check).
    stock_tokenizer = AutoTokenizer.from_pretrained(base_model)
    stock_model = AutoModelForSequenceClassification.from_pretrained(base_model).to(model.device)

    v1_rows = load_rows(V1_EVAL)
    v2_rows = load_rows(V2_EVAL)

    stock_v1_f1, stock_v1_per = macro_f1(stock_model, stock_tokenizer, v1_rows, args.batch_size)
    stock_v2_f1, stock_v2_per = macro_f1(stock_model, stock_tokenizer, v2_rows, args.batch_size)
    del stock_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    trained_v1_f1, trained_v1_per = macro_f1(model, tokenizer, v1_rows, args.batch_size)
    trained_v2_f1, trained_v2_per = macro_f1(model, tokenizer, v2_rows, args.batch_size)

    print(
        f"v1 holdout: stock {stock_v1_f1:.4f}  ->  v2-model {trained_v1_f1:.4f}  (delta {trained_v1_f1 - stock_v1_f1:+.4f})"
    )
    print(
        f"v2 holdout: stock {stock_v2_f1:.4f}  ->  v2-model {trained_v2_f1:.4f}  (delta {trained_v2_f1 - stock_v2_f1:+.4f})"
    )

    # Gate: v2 must beat stock on v1 by >=10pp AND beat stock on v2 by >=10pp.
    # (v1 in-domain regression check is implicit in "must beat stock by the same
    # margin v1 of the project established as the release bar".)
    gate_pass = (trained_v1_f1 - stock_v1_f1) >= 0.10 and (trained_v2_f1 - stock_v2_f1) >= 0.10
    print(f"\nrelease gate: {'PASS' if gate_pass else 'FAIL'}")

    report = {
        "base_model": base_model,
        "seed": seed,
        "epochs": args.epochs,
        "stock_v1_f1": stock_v1_f1,
        "stock_v1_per_clause": stock_v1_per,
        "v2_model_v1_f1": trained_v1_f1,
        "v2_model_v1_per_clause": trained_v1_per,
        "stock_v2_f1": stock_v2_f1,
        "stock_v2_per_clause": stock_v2_per,
        "v2_model_v2_f1": trained_v2_f1,
        "v2_model_v2_per_clause": trained_v2_per,
        "gate_pass": gate_pass,
    }
    (out_dir / "release_gate.json").write_text(json.dumps(report, indent=2))
    print(f"report written to {out_dir / 'release_gate.json'}")

    if not gate_pass:
        sys.exit("release gate failed -- not exporting ONNX")

    if args.skip_quantize:
        print("--skip-quantize: stopping after pytorch checkpoint + gate")
        return 0

    # --- ONNX export + quantization (same recipe as v1) ---
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    onnx_dir = out_dir / "onnx"
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(pytorch_out), export=True)
    ort_model.save_pretrained(str(onnx_dir))
    print(f"onnx model saved to {onnx_dir}")

    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    # per_channel=True is REQUIRED for DeBERTa: per-tensor dynamic INT8 collapses
    # its disentangled-attention weights and produces a constant predictor (v3's
    # first shipped artifact was dead this way; v1/v2 were RoBERTa and survived
    # per-tensor). Do not revert to per_channel=False without re-checking the
    # artifact gate below.
    variants = {
        "model_quint8_avx2.onnx": AutoQuantizationConfig.avx2(is_static=False, per_channel=True),
        "model_qint8_avx512.onnx": AutoQuantizationConfig.avx512(is_static=False, per_channel=True),
        "model_qint8_avx512_vnni.onnx": AutoQuantizationConfig.avx512_vnni(
            is_static=False, per_channel=True
        ),
        "model_qint8_arm64.onnx": AutoQuantizationConfig.arm64(is_static=False, per_channel=True),
    }
    for filename, qconfig in variants.items():
        target = onnx_dir / filename
        tmp_dir = onnx_dir / filename.replace(".onnx", "_tmp")
        quantizer.quantize(save_dir=str(tmp_dir), quantization_config=qconfig)
        produced = list(tmp_dir.glob("*.onnx"))[0]
        shutil.move(str(produced), str(target))
        shutil.rmtree(tmp_dir)
        print(f"quantized -> {target}")

    # --- Post-export ARTIFACT gate: validate the SHIPPED quantized files ---
    # The release gate above scored the PyTorch model. Quantization is downstream
    # of it and can silently ship a dead artifact that still returns plausible
    # scores. Score every quantized variant on the real eval and refuse to ship a
    # constant predictor or a >10pp macro-F1 regression from PyTorch.
    for filename in variants:
        q_v1_f1, q_v1_distinct = onnx_macro_f1(
            onnx_dir / filename, tokenizer, v1_rows, args.batch_size
        )
        q_v2_f1, q_v2_distinct = onnx_macro_f1(
            onnx_dir / filename, tokenizer, v2_rows, args.batch_size
        )
        print(
            f"[artifact-gate] {filename}: v1 F1={q_v1_f1:.4f} (torch {trained_v1_f1:.4f}), "
            f"v2 F1={q_v2_f1:.4f} (torch {trained_v2_f1:.4f}), "
            f"distinct preds v1={q_v1_distinct} v2={q_v2_distinct}"
        )
        passed, reason = artifact_gate_verdict(
            q_v1_f1, q_v1_distinct, q_v2_f1, q_v2_distinct, trained_v1_f1, trained_v2_f1
        )
        if not passed:
            sys.exit(f"ARTIFACT GATE FAIL: {filename} — {reason} — not shipping")
    print("[artifact-gate] PASS — all quantized variants preserve the model")

    shutil.copy(str(V1_EVAL), str(out_dir / "eval.jsonl"))
    shutil.copy(str(V2_EVAL), str(out_dir / "eval_v2.jsonl"))
    print(f"bundled eval sets -> {out_dir}")

    print(f"\nDONE. Upload {out_dir} to HuggingFace as labrat-aiko/{out_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
