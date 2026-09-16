"""Throwaway experiment: AVX2 file vs a qint8 file, scores AND latency, on one CPU.

Roadmap Now item 3 (`docs/superpowers/specs/2026-09-15-development-roadmap-design.md`) needs a
decision between (a) shipping one qint8 file everywhere, (b) re-quantizing the AVX2 file, and
(c) keeping it and documenting the failure. v1's AVX2 file fails the release gate on minimality,
and Windows / Intel macOS always load it.

The missing evidence is AVX2-only hardware. On a VNNI runner the qint8 files score better, but the
AVX2 variant exists for *speed* on pre-VNNI CPUs: u8s8 weights need VNNI for the fast integer
kernel path, so a qint8 file could be slower there. This measures both halves on whatever CPU it
lands on and prints the CPU flags, so a VNNI run is still a useful control.

Run it in CI (see .github/workflows/exp-avx2.yml) and re-dispatch until `lscpu` reports a runner
without avx512f. Output is a decision, not a shipped artifact -- this file is not meant for master.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from semantix.eval.popia import GATE_THRESHOLD, evaluate_popia
from semantix.judges.popia import POPIAJudge
from semantix.judges.quantized_nli import QuantizedNLIJudge, runtime_info

EVAL = Path("data/popia_eval.jsonl")
EVAL_HASH = Path("scripts/_popia_eval_hash.txt")

AVX2 = "onnx/model_quint8_avx2.onnx"
QINT8 = "onnx/model_qint8_avx512_vnni.onnx"

# One fixed pair, so latency is comparable across variants.
LATENCY_PREMISE = (
    "We collect your ID number and physical address to verify your identity, and we asked "
    "for your permission before doing so."
)
LATENCY_HYPOTHESIS = "The organisation obtained consent before processing personal information."
LATENCY_WARMUP = 5
LATENCY_N = 50


def _verify_eval() -> None:
    if not EVAL.exists():
        sys.exit(f"missing {EVAL}")
    pinned = EVAL_HASH.read_text().strip()
    current = hashlib.sha256(EVAL.read_bytes()).hexdigest()
    if pinned != current:
        sys.exit(f"EVAL SET INTEGRITY FAILURE\n  pinned : {pinned}\n  current: {current}")


def _latency_ms(judge, n: int = LATENCY_N) -> dict[str, float]:
    for _ in range(LATENCY_WARMUP):
        judge.evaluate(LATENCY_PREMISE, LATENCY_HYPOTHESIS, GATE_THRESHOLD)
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        judge.evaluate(LATENCY_PREMISE, LATENCY_HYPOTHESIS, GATE_THRESHOLD)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "n": float(n),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (n - 1))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def _measure(variant: str) -> dict:
    popia = POPIAJudge(model_variant=variant)
    stock = QuantizedNLIJudge(model_variant=variant)
    report = evaluate_popia(EVAL, popia, stock, GATE_THRESHOLD)
    return {
        "variant": variant,
        "stock_f1_macro": report.stock_f1_macro,
        "popia_f1_macro": report.popia_f1_macro,
        "delta_f1": report.delta_f1,
        "release_gate_passed": report.release_gate_passed,
        "regressed_clauses": report.regressed_clauses,
        "per_clause": {k: list(v) for k, v in report.per_clause.items()},
        "popia_latency": _latency_ms(popia),
        "stock_latency": _latency_ms(stock),
    }


def main() -> int:
    _verify_eval()
    runtime = runtime_info()
    flags = runtime["cpu_flags"]
    has_avx512 = "avx512f" in flags
    print(f"runtime: {runtime}")
    print(f"AVX-512 present: {has_avx512}  ({'CONTROL run' if has_avx512 else 'TARGET run'})\n")

    results = [_measure(AVX2), _measure(QINT8)]

    print(f"{'file':<36} {'stock F1':>9} {'POPIA F1':>9} {'gate':>6} {'POPIA p50':>10}")
    for r in results:
        gate = "PASS" if r["release_gate_passed"] else "FAIL"
        print(
            f"{r['variant']:<36} {r['stock_f1_macro']:>9.4f} {r['popia_f1_macro']:>9.4f} "
            f"{gate:>6} {r['popia_latency']['p50_ms']:>9.1f}ms"
        )
        if r["regressed_clauses"]:
            print(f"    regressed vs stock: {', '.join(r['regressed_clauses'])}")

    avx2_r, qint8_r = results
    f1_gain = qint8_r["popia_f1_macro"] - avx2_r["popia_f1_macro"]
    lat_cost = qint8_r["popia_latency"]["p50_ms"] - avx2_r["popia_latency"]["p50_ms"]
    print(
        f"\nqint8 vs AVX2 on this CPU: POPIA F1 {f1_gain:+.4f}, "
        f"p50 latency {lat_cost:+.1f} ms ({lat_cost / avx2_r['popia_latency']['p50_ms']:+.0%})"
    )
    print(
        "Decision input: option (a) one qint8 file everywhere is justified when F1 gain is "
        "positive and the latency cost stays inside the published ~15-70 ms envelope."
    )

    Path("exp_avx2_report.json").write_text(
        json.dumps({"runtime": runtime, "has_avx512": has_avx512, "results": results}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
