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
SEEDS_PATH = Path("data/popia_seeds.jsonl")
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
    # Must match the base model's config.id2label (contradiction=0, entailment=1,
    # neutral=2) so fine-tuning reinforces -- not scrambles -- the inherited labels.
    return {"contradiction": 0, "entailment": 1, "neutral": 2}[label]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    verify_eval_integrity()

    from datasets import Dataset
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if TRAIN_PATH.exists():
        source = TRAIN_PATH
    elif SEEDS_PATH.exists():
        source = SEEDS_PATH
        print(f"no {TRAIN_PATH} -- falling back to seeds-only training from {SEEDS_PATH}")
    else:
        sys.exit(f"missing both {TRAIN_PATH} and {SEEDS_PATH}")

    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    print(f"loaded {len(rows)} training rows from {source}")

    split_idx = int(len(rows) * 0.9)
    train_rows, dev_rows = rows[:split_idx], rows[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

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

    onnx_dir = OUT_DIR / "onnx"
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(pytorch_out),
        export=True,
    )
    ort_model.save_pretrained(str(onnx_dir))
    print(f"onnx model saved to {onnx_dir}")

    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    variants = {
        "model_quint8_avx2.onnx": AutoQuantizationConfig.avx2(is_static=False, per_channel=False),
        "model_qint8_avx512.onnx": AutoQuantizationConfig.avx512(
            is_static=False, per_channel=False
        ),
        "model_qint8_avx512_vnni.onnx": AutoQuantizationConfig.avx512_vnni(
            is_static=False, per_channel=False
        ),
        "model_qint8_arm64.onnx": AutoQuantizationConfig.arm64(is_static=False, per_channel=False),
    }
    for filename, qconfig in variants.items():
        target = onnx_dir / filename
        tmp_dir = onnx_dir / filename.replace(".onnx", "_tmp")
        quantizer.quantize(save_dir=str(tmp_dir), quantization_config=qconfig)
        produced = list(tmp_dir.glob("*.onnx"))[0]
        shutil.move(str(produced), str(target))
        shutil.rmtree(tmp_dir)
        print(f"quantized -> {target}")

    shutil.copy(str(EVAL_PATH), str(OUT_DIR / "eval.jsonl"))
    print(f"bundled eval set -> {OUT_DIR / 'eval.jsonl'}")

    print(f"\nDONE. Upload {OUT_DIR} to HuggingFace as labrat-aiko/nli-popia-v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
