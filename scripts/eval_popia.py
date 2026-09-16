"""Reproducibility wrapper: run the POPIA release gate on the local data/popia_eval.jsonl.

Unlike `semantix eval popia`, which downloads eval.jsonl from HF, this script uses the
exact file in the repo. Models still load from Hugging Face.

Usage:
    python scripts/eval_popia.py                 # gate the file this machine auto-selects
    python scripts/eval_popia.py --all-files     # gate every shipped ONNX file
    python scripts/eval_popia.py --use-hf        # use HF eval.jsonl instead of the local file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semantix.eval.popia import evaluate_popia, evaluate_popia_matrix
from semantix.judges.popia import POPIAJudge
from semantix.judges.quantized_nli import QuantizedNLIJudge

LOCAL_EVAL = Path("data/popia_eval.jsonl")


def _load_judges(variant: str):
    return POPIAJudge(model_variant=variant), QuantizedNLIJudge(model_variant=variant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hf", action="store_true", help="Use HF eval.jsonl instead of local.")
    ap.add_argument("--all-files", action="store_true", help="Gate every shipped ONNX file.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.use_hf:
        from huggingface_hub import hf_hub_download

        eval_path = hf_hub_download(repo_id="labrat-aiko/nli-popia-v1", filename="eval.jsonl")
    else:
        if not LOCAL_EVAL.exists():
            print(f"missing {LOCAL_EVAL}", file=sys.stderr)
            return 2
        eval_path = LOCAL_EVAL

    if args.all_files:
        matrix = evaluate_popia_matrix(eval_path, _load_judges)
        if args.json:
            print(json.dumps(matrix.as_dict(), indent=2))
        else:
            print(f"runtime: {matrix.runtime}")
            for variant, r in matrix.results.items():
                gate = "PASS" if r.release_gate_passed else "FAIL"
                print(
                    f"{variant:<36} stock F1={r.stock_f1_macro:.3f}  "
                    f"POPIA F1={r.popia_f1_macro:.3f}  delta={r.delta_f1:+.3f}  gate: {gate}"
                )
        return 0 if matrix.all_passed else 1

    report = evaluate_popia(eval_path, POPIAJudge(), QuantizedNLIJudge())

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"n_pairs={report.n_pairs}")
        print(
            f"stock F1={report.stock_f1_macro:.3f}  "
            f"POPIA F1={report.popia_f1_macro:.3f}  delta={report.delta_f1:+.3f}"
        )
        print(f"gate: {'PASS' if report.release_gate_passed else 'FAIL'}")

    return 0 if report.release_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
