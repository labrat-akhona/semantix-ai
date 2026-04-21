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
    per_clause: dict[str, tuple[float, float]]
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
