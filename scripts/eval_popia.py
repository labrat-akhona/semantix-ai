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
        print(
            f"stock F1={report.stock_f1_macro:.3f}  "
            f"POPIA F1={report.popia_f1_macro:.3f}  delta={report.delta_f1:+.3f}"
        )
        print(f"gate: {'PASS' if report.release_gate_passed else 'FAIL'}")

    return 0 if report.release_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
