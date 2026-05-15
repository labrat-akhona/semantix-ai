"""Score a POPIA-Bench v1 submission.

Usage:
    python bench/popia-v1/score.py path/to/submission.jsonl

Submission format: one JSON object per line, with fields:
    {"id": <0-based example index in popia_bench.jsonl>,
     "prediction": "contradiction" | "entailment" | "neutral"}

Output: JSON-formatted scorecard with macro F1 overall + per-clause +
disclosed runtime baseline cells.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).parent / "popia_bench.jsonl"
LABEL_KEY = {"contradiction": 0, "entailment": 1, "neutral": 2}


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute macro F1 over 3 classes without scikit-learn dependency."""
    if not y_true:
        return 0.0
    f1s = []
    for cls in (0, 1, 2):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 3.0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    submission_path = Path(argv[1])

    bench = [json.loads(line) for line in BENCH.read_text().splitlines() if line.strip()]
    pred_by_id: dict[int, str] = {}
    for line in submission_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            pred_by_id[int(row["id"])] = row["prediction"]

    if len(pred_by_id) != len(bench):
        print(f"WARNING: submission has {len(pred_by_id)} predictions, bench has {len(bench)} examples")

    y_true, y_pred = [], []
    per_clause_true: dict[str, list[int]] = defaultdict(list)
    per_clause_pred: dict[str, list[int]] = defaultdict(list)
    missing = 0

    for i, ex in enumerate(bench):
        if i not in pred_by_id:
            missing += 1
            continue
        t = LABEL_KEY[ex["label"]]
        p_str = pred_by_id[i]
        if p_str not in LABEL_KEY:
            print(f"ERROR: example {i} predicts invalid label '{p_str}'")
            return 3
        p = LABEL_KEY[p_str]
        y_true.append(t)
        y_pred.append(p)
        per_clause_true[ex["clause"]].append(t)
        per_clause_pred[ex["clause"]].append(p)

    result = {
        "examples_scored": len(y_true),
        "missing_predictions": missing,
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "per_clause_f1": {
            clause: round(macro_f1(per_clause_true[clause], per_clause_pred[clause]), 4)
            for clause in sorted(per_clause_true)
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
