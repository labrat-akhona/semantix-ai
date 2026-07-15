"""Error analysis for POPIA-Judge v2 — per-clause 3-way confusion matrices.

Runs inference on the v1 + v2 holdouts (197 pairs total) and emits:
  - reports/v2_confusion_matrices.json — per-clause confusion + summary
  - stdout LaTeX-ready table snippets for drop-in to the preprint

Same ONNX artifact and label order as scripts/calibrate_popia_v2.py
(contradiction=0, entailment=1, neutral=2). CPU-only, deterministic.

Usage:
    python scripts/error_analysis_popia_v2.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ONNX_DIR = Path("out/nli-popia-v2/onnx")
MODEL_PATH = ONNX_DIR / "model.onnx"
EVAL_PATHS = [
    Path("out/nli-popia-v2/eval.jsonl"),
    Path("out/nli-popia-v2/eval_v2.jsonl"),
]
OUT_JSON = Path("reports/v2_confusion_matrices.json")

LABELS = ["contradiction", "entailment", "neutral"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for p in EVAL_PATHS:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_inference(model_path: Path, tokenizer, rows: list[dict], batch: int = 16) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_infos = {i.name: i.type for i in sess.get_inputs()}
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        enc = tokenizer(
            [r["premise"] for r in chunk],
            [r["hypothesis"] for r in chunk],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="np",
        )
        feed = {}
        for name, type_str in input_infos.items():
            if name not in enc:
                continue
            arr = enc[name]
            if type_str == "tensor(int64)" and arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            feed[name] = arr
        logits = sess.run(None, feed)[0]
        out.append(logits)
    return np.concatenate(out, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def confusion_matrix(true: list[int], pred: list[int], n: int = 3) -> list[list[int]]:
    m = [[0] * n for _ in range(n)]
    for t, p in zip(true, pred, strict=False):
        m[t][p] += 1
    return m


def main() -> int:
    rows = load_rows()
    print(f"loaded {len(rows)} eval pairs")

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
    logits = run_inference(MODEL_PATH, tokenizer, rows)
    probs = softmax(logits)
    preds = probs.argmax(axis=-1)
    confs = probs.max(axis=-1)
    truths = np.array([LABEL_TO_ID[r["label"]] for r in rows])

    # Per-clause confusion matrices
    by_clause = defaultdict(lambda: {"true": [], "pred": [], "conf": []})
    for r, t, p, c in zip(rows, truths.tolist(), preds.tolist(), confs.tolist(), strict=False):
        b = by_clause[r["clause"]]
        b["true"].append(t)
        b["pred"].append(p)
        b["conf"].append(c)

    per_clause: dict[str, dict] = {}
    for clause, d in by_clause.items():
        cm = confusion_matrix(d["true"], d["pred"])
        n = len(d["true"])
        n_correct = sum(1 for t, p in zip(d["true"], d["pred"], strict=False) if t == p)
        per_clause[clause] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n,
            "confusion": cm,  # rows = true, cols = pred; index order = LABELS
            "labels": LABELS,
        }

    # Confidently-wrong: highest-confidence misclassifications, top 8
    wrong_idx = [i for i in range(len(rows)) if truths[i] != preds[i]]
    wrong_idx.sort(key=lambda i: -confs[i])
    top_wrong = []
    for i in wrong_idx[:8]:
        top_wrong.append(
            {
                "clause": rows[i]["clause"],
                "scenario": rows[i].get("scenario", ""),
                "premise": rows[i]["premise"],
                "hypothesis": rows[i]["hypothesis"],
                "true_label": LABELS[truths[i]],
                "predicted_label": LABELS[preds[i]],
                "confidence": float(confs[i]),
            }
        )

    # Aggregate
    overall_correct = int((truths == preds).sum())
    summary = {
        "n_eval": len(rows),
        "n_correct": overall_correct,
        "argmax_accuracy": overall_correct / len(rows),
        "labels_order": LABELS,
        "label_id_map": LABEL_TO_ID,
        "artifact": "labrat-aiko/nli-popia-v2 (onnx/model.onnx, FP32)",
        "eval_sources": [str(p) for p in EVAL_PATHS],
    }

    out = {
        "summary": summary,
        "per_clause": per_clause,
        "top_confidently_wrong": top_wrong,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")
    print()

    # --- LaTeX summary table to stdout for paper-section prep ---
    print(
        "=== Per-clause confusion (rows=true, cols=pred; C=contradiction, E=entailment, N=neutral) ===\n"
    )
    print(f"{'Clause':45s} {'n':>4s}  {'acc':>6s}  C->C E->C N->C  C->E E->E N->E  C->N E->N N->N")
    for clause in sorted(per_clause):
        d = per_clause[clause]
        cm = d["confusion"]
        flat = " ".join(f"{cm[t][p]:4d}" for p in range(3) for t in range(3))
        print(f"{clause[:45]:45s} {d['n']:>4d}  {d['accuracy']:>6.3f}  {flat}")
    print()
    print(
        f"overall: {summary['n_correct']}/{summary['n_eval']}  argmax-accuracy={summary['argmax_accuracy']:.4f}"
    )
    print()
    print("=== Top 5 confidently-wrong examples (confidence-sorted) ===\n")
    for ex in top_wrong[:5]:
        print(
            f"[{ex['clause']}] truth={ex['true_label']}  pred={ex['predicted_label']}  conf={ex['confidence']:.3f}"
        )
        print(f"  P: {ex['premise'][:120]}")
        print(f"  H: {ex['hypothesis'][:120]}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
