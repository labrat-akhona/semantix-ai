"""One-shot local eval driver — validates a freshly-trained model before HF upload.

Monkey-patches quantized_nli loaders so POPIAJudge reads from ``out/nli-popia-v1``
while the stock baseline still downloads from HF. Not shipped — remove after use.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from tokenizers import Tokenizer

from semantix.judges import quantized_nli as _qnli

MODEL_DIR = Path("out/nli-popia-v1")
POPIA_REPO = "labrat-aiko/nli-popia-v1"

_orig_session = _qnli._load_session
_orig_tokenizer = _qnli._load_tokenizer


def _local_session(variant: str, repo_id: str = _qnli._REPO_ID):
    if repo_id == POPIA_REPO:
        path = MODEL_DIR / variant
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return _orig_session(variant, repo_id)


def _local_tokenizer(repo_id: str = _qnli._REPO_ID):
    if repo_id == POPIA_REPO:
        return Tokenizer.from_file(str(MODEL_DIR / "pytorch" / "tokenizer.json"))
    return _orig_tokenizer(repo_id)


_qnli._load_session = _local_session
_qnli._load_tokenizer = _local_tokenizer

from semantix.eval.popia import evaluate_popia  # noqa: E402
from semantix.judges.popia import POPIAJudge  # noqa: E402
from semantix.judges.quantized_nli import QuantizedNLIJudge  # noqa: E402

report = evaluate_popia(Path("data/popia_eval.jsonl"), POPIAJudge(), QuantizedNLIJudge())

print(f"n_pairs={report.n_pairs}")
print(f"stock  F1 macro = {report.stock_f1_macro:.4f}  acc = {report.stock_accuracy:.4f}")
print(f"POPIA  F1 macro = {report.popia_f1_macro:.4f}  acc = {report.popia_accuracy:.4f}")
print(f"delta  F1       = {report.delta_f1:+.4f}  (gate: >= 0.10)")
print()
print("per-clause (stock_f1, popia_f1, delta):")
for clause, (s, p) in report.per_clause.items():
    delta = p - s
    marker = " ✓" if delta >= 0 else " ✗ REGRESSION"
    print(f"  {clause:45s}  stock={s:.3f}  popia={p:.3f}  delta={delta:+.3f}{marker}")
print()
print(f"RELEASE GATE: {'PASS' if report.release_gate_passed else 'FAIL'}")
