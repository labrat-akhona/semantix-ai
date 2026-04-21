"""Semantix CLI — run semantic checks from the terminal.

Usage
-----
    semantix check "Thank you for your patience" --intent "polite and professional"
    semantix check "some text" --intent "helpful" --threshold 0.85
    semantix check "some text" --intent "polite" --judge nli
    semantix check "I recommend aspirin" --intent "medical advice" --negate
    semantix prove
    semantix prove --text "..." --intent "..." --n 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ANSI colour helpers — suppress via NO_COLOR env var or --no-color flag.
_NO_COLOR = os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty()
_GREEN = "" if _NO_COLOR else "\033[32m"
_RED = "" if _NO_COLOR else "\033[31m"
_YELLOW = "" if _NO_COLOR else "\033[33m"
_DIM = "" if _NO_COLOR else "\033[2m"
_BOLD = "" if _NO_COLOR else "\033[1m"
_RESET = "" if _NO_COLOR else "\033[0m"


def _disable_color() -> None:
    """Disable ANSI colour codes at runtime (called by --no-color)."""
    global _GREEN, _RED, _YELLOW, _DIM, _BOLD, _RESET
    _GREEN = _RED = _YELLOW = _DIM = _BOLD = _RESET = ""


def _default_judge():
    """Resolve the default judge: QuantizedNLI -> NLI fallback."""
    try:
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    except ImportError:
        from semantix.judges.nli import NLIJudge

        return NLIJudge()


def _resolve_judge(name: str | None):
    """Return a Judge instance for the given name (lazy imports)."""
    if name is None:
        return _default_judge()

    name = name.lower()
    if name == "nli":
        from semantix.judges.nli import NLIJudge

        return NLIJudge()
    elif name == "embedding":
        from semantix.judges.embedding import EmbeddingJudge

        return EmbeddingJudge()
    elif name == "quantized":
        from semantix.judges.quantized_nli import QuantizedNLIJudge

        return QuantizedNLIJudge()
    else:
        print(f"Unknown judge: {name!r}. Choose from: nli, embedding, quantized", file=sys.stderr)
        sys.exit(2)


def _resolve_threshold(explicit: float | None, judge) -> float:
    """Threshold resolution: explicit > judge.recommended_threshold > 0.8."""
    if explicit is not None:
        return explicit
    if getattr(judge, "recommended_threshold", None) is not None:
        return judge.recommended_threshold
    return 0.8


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantix",
        description="Semantix — semantic checks from the terminal.",
    )
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="Check text against a semantic intent")
    check.add_argument("text", help="The text to validate")
    check.add_argument("--intent", required=True, help="Semantic intent description")
    check.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum score to pass (default: judge-recommended or 0.8)",
    )
    check.add_argument(
        "--judge",
        default=None,
        choices=["nli", "embedding", "quantized"],
        help="Judge backend (default: quantized with nli fallback)",
    )
    check.add_argument(
        "--negate",
        action="store_true",
        help="Invert the check — pass means the intent is NOT satisfied",
    )

    prove = sub.add_parser(
        "prove",
        help="Prove determinism — run the same validation N times, verify identical scores",
    )
    prove.add_argument(
        "--text",
        default=(
            "Thank you for reaching out. I've reviewed your complaint about the "
            "delayed delivery and I'm issuing a full refund which will reflect in "
            "your account within 3 business days."
        ),
        help="Text to validate (default: a built-in customer-service example)",
    )
    prove.add_argument(
        "--intent",
        default=(
            "The response must acknowledge the customer's issue and propose a "
            "concrete next step, in a polite tone."
        ),
        help="Intent to validate against (default: a built-in customer-service intent)",
    )
    prove.add_argument(
        "--n", type=int, default=100, help="Number of repetitions (default: 100)",
    )
    prove.add_argument(
        "--judge",
        default=None,
        choices=["nli", "embedding", "quantized"],
        help="Judge backend (default: quantized with nli fallback)",
    )
    prove.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for pass/fail (default: judge-recommended or 0.8)",
    )
    prove.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour output",
    )

    return parser


def _run_check(args) -> int:
    """Execute semantic check, print results, return exit code."""
    judge = _resolve_judge(args.judge)
    threshold = _resolve_threshold(args.threshold, judge)

    verdict = judge.evaluate(args.text, args.intent, threshold=threshold)

    passed = not verdict.passed if args.negate else verdict.passed
    tag = f"{_BOLD}{_GREEN}PASS{_RESET}" if passed else f"{_BOLD}{_RED}FAIL{_RESET}"

    score_str = f"{verdict.score:.4f}" if verdict.score is not None else "n/a"

    print(f"{tag}  score={score_str}  intent={args.intent!r}")
    if not passed and verdict.reason:
        print(f"  reason: {verdict.reason}")

    return 0 if passed else 1


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the pct-th percentile of a pre-sorted list (linear interpolation)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _run_prove(args) -> int:
    """Run the same validation N times and report on determinism."""
    if getattr(args, "no_color", False):
        _disable_color()

    judge = _resolve_judge(args.judge)
    threshold = _resolve_threshold(args.threshold, judge)
    n = max(1, args.n)

    print(f"{_BOLD}Determinism proof{_RESET}  (n={n}, judge={type(judge).__name__})")
    print(f"  intent: {args.intent!r}")
    print(f"  text:   {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print()

    scores: list[float] = []
    latencies_ms: list[float] = []
    start_wall = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        verdict = judge.evaluate(args.text, args.intent, threshold=threshold)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        # Treat None score as NaN-ish — force float for set-collapse test.
        scores.append(float(verdict.score) if verdict.score is not None else float("nan"))
    wall_ms = (time.perf_counter() - start_wall) * 1000

    # Determinism check: all scores identical to 5 decimal places.
    rounded = {round(s, 5) for s in scores}
    deterministic = len(rounded) == 1

    latencies_sorted = sorted(latencies_ms)
    p50 = _percentile(latencies_sorted, 50)
    p95 = _percentile(latencies_sorted, 95)
    p99 = _percentile(latencies_sorted, 99)

    if deterministic:
        rep = scores[0]
        rep_str = f"{rep:.5f}" if rep == rep else "nan"  # NaN-safe
        print(f"{_BOLD}{_GREEN}DETERMINISM VERIFIED{_RESET}")
        print(f"  Score: {rep_str} -- {n}/{n} runs agreed to 5 decimal places.")
    else:
        print(f"{_BOLD}{_RED}NON-DETERMINISTIC{_RESET}")
        print(f"  Observed {len(rounded)} distinct scores across {n} runs:")
        for s in sorted(rounded):
            count = sum(1 for x in scores if round(x, 5) == s)
            print(f"    {s:.5f} × {count}")

    print()
    print(f"{_DIM}Wall-clock: {wall_ms:.0f} ms total -- {wall_ms / n:.2f} ms/call{_RESET}")
    print(
        f"{_DIM}Latency p50/p95/p99: "
        f"{p50:.1f} / {p95:.1f} / {p99:.1f} ms{_RESET}"
    )
    print()
    print(
        f"{_DIM}Determinism matters for CI reward loops, compliance audits, and{_RESET}\n"
        f"{_DIM}optimization runs. LLM-as-judge scores drift run-to-run even at{_RESET}\n"
        f"{_DIM}temperature=0 -- local NLI models do not.{_RESET}"
    )

    return 0 if deterministic else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "check":
        return _run_check(args)

    if args.command == "prove":
        return _run_prove(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
