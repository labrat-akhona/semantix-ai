"""Semantix CLI — run semantic checks from the terminal.

Usage
-----
    semantix check "Thank you for your patience" --intent "polite and professional"
    semantix check "some text" --intent "helpful" --threshold 0.85
    semantix check "some text" --intent "polite" --judge nli
    semantix check "I recommend aspirin" --intent "medical advice" --negate
"""

from __future__ import annotations

import argparse
import sys

# ANSI colour helpers (no dependency needed)
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


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


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "check":
        return _run_check(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
