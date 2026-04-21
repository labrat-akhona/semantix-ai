"""Eval each training checkpoint against the pinned eval set to find the sweet spot.

Uses PyTorch directly (no ONNX) for speed. Mimics QuantizedNLIJudge scoring semantics:
softmax over logits, take index 2 (entailment) for the score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from semantix.eval.popia import evaluate_popia
from semantix.judges import Judge, Verdict
from semantix.judges.nli import _to_hypothesis
from semantix.judges.quantized_nli import QuantizedNLIJudge

CKPT_DIR = Path("out/nli-popia-v1/pytorch")
EVAL_PATH = Path("data/popia_eval.jsonl")


class PyTorchPopiaJudge(Judge):
    recommended_threshold = 0.75

    def __init__(self, ckpt_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, output, intent_description, threshold=0.5):
        hypothesis = _to_hypothesis(intent_description)
        enc = self.tokenizer(output, hypothesis, return_tensors="pt",
                             truncation=True, padding="max_length", max_length=256)
        logits = self.model(**enc).logits[0].numpy()
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        # PyTorch model label order: {0: entailment, 1: neutral, 2: contradiction}
        # (verified by label_to_id in train_popia.py)
        entailment_score = float(probs[0])
        return Verdict(passed=entailment_score >= threshold, score=entailment_score)

    @classmethod
    def clauses(cls):
        return [
            "POPIA consent", "POPIA minimality / purpose limitation",
            "POPIA security safeguards", "POPIA breach notification",
            "POPIA cross-border transfers", "POPIA general processing",
            "POPIA data subject rights",
        ]


def main():
    stock = QuantizedNLIJudge()
    checkpoints = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    checkpoints.append(CKPT_DIR)  # final best
    results = []
    for ckpt in checkpoints:
        name = ckpt.name if ckpt.name.startswith("checkpoint") else "best(final)"
        try:
            judge = PyTorchPopiaJudge(ckpt)
        except Exception as e:
            print(f"{name}: load failed -- {e}", file=sys.stderr)
            continue
        report = evaluate_popia(EVAL_PATH, judge, stock)
        regressions = sum(1 for s, p in report.per_clause.values() if p < s)
        results.append((name, report.popia_f1_macro, report.delta_f1, regressions,
                        report.release_gate_passed))
        print(f"{name:22s}  POPIA F1={report.popia_f1_macro:.3f}  "
              f"delta={report.delta_f1:+.3f}  regressions={regressions}  "
              f"gate={'PASS' if report.release_gate_passed else 'FAIL'}")
    print()
    print(f"stock reference      POPIA F1={stock.__class__.__name__}  stock F1={report.stock_f1_macro:.3f}")
    best = max(results, key=lambda r: (r[4], r[2]))
    print(f"best: {best[0]}")


if __name__ == "__main__":
    main()
