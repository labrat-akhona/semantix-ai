"""Semantix CLI — run semantic checks from the terminal.

Usage
-----
    semantix check "Thank you for your patience" --intent "polite and professional"
    semantix check "some text" --intent "helpful" --threshold 0.85
    semantix check "some text" --intent "polite" --judge nli
    semantix check "I recommend aspirin" --intent "medical advice" --negate
    semantix prove
    semantix prove --text "..." --intent "..." --n 100
    semantix demo                        # 3 canned scenarios in under a second
    semantix verify audit.jsonl          # check a tamper-evident audit trail
"""

from __future__ import annotations

import argparse
import json
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

    demo = sub.add_parser(
        "demo",
        help="Run a 3-scenario live demo — pass, fail, and negated-fail in under a second",
    )
    demo.add_argument(
        "--judge",
        default=None,
        choices=["nli", "embedding", "quantized"],
        help="Judge backend (default: quantized with nli fallback)",
    )
    demo.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour output",
    )

    verify = sub.add_parser(
        "verify",
        help="Verify a semantix audit trail (JSONL file) — check hash chain and summarise",
    )
    verify.add_argument(
        "path", help="Path to a JSONL audit trail (written by AuditEngine.flush)",
    )
    verify.add_argument(
        "--top", type=int, default=5, help="Number of top intents to display (default: 5)",
    )
    verify.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour output",
    )

    eval_parser = sub.add_parser("eval", help="Run release-gate evaluations.")
    eval_sub = eval_parser.add_subparsers(dest="eval_target", required=True)
    popia_eval = eval_sub.add_parser("popia", help="Evaluate POPIAJudge vs stock NLI.")
    popia_eval.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    return parser


def _download_popia_eval():
    """Download eval.jsonl from HF and return the local cached path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="labrat-akhona/nli-popia-v1",
        filename="eval.jsonl",
    )


def _load_popia_judges():
    """Instantiate POPIAJudge and stock QuantizedNLIJudge."""
    from semantix.judges.popia import POPIAJudge
    from semantix.judges.quantized_nli import QuantizedNLIJudge
    return POPIAJudge(), QuantizedNLIJudge()


def evaluate_popia(*args, **kwargs):
    """Indirection so tests can monkeypatch semantix.cli.evaluate_popia."""
    from semantix.eval.popia import evaluate_popia as _impl
    return _impl(*args, **kwargs)


def _run_eval_popia(args) -> int:
    from dataclasses import asdict

    try:
        eval_path = _download_popia_eval()
    except Exception as e:
        print(f"failed to download eval set: {e}", file=sys.stderr)
        return 2

    popia, stock = _load_popia_judges()
    report = evaluate_popia(eval_path, popia, stock)

    if args.json:
        out = asdict(report)
        out["per_clause"] = {k: list(v) for k, v in out["per_clause"].items()}
        print(json.dumps(out, indent=2))
    else:
        print(f"\n                    stock    POPIA    Delta")
        print(
            f"Accuracy           {report.stock_accuracy:.2f}     "
            f"{report.popia_accuracy:.2f}     "
            f"{report.popia_accuracy - report.stock_accuracy:+.2f}"
        )
        print(
            f"F1 (macro)         {report.stock_f1_macro:.2f}     "
            f"{report.popia_f1_macro:.2f}     {report.delta_f1:+.2f}"
        )
        for clause, (stock_f, popia_f) in report.per_clause.items():
            print(f"  {clause:<30} {stock_f:.2f}     {popia_f:.2f}     {popia_f - stock_f:+.2f}")
        verdict = "PASS" if report.release_gate_passed else "FAIL"
        print(f"\nRelease gate (>= 0.10 F1 delta, no per-clause regression): {verdict}")

    return 0 if report.release_gate_passed else 1


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


_DEMO_SCENARIOS = [
    {
        "label": "polite response",
        "intent": "polite and professional customer service",
        "text": "Thank you for reaching out. I've reviewed your complaint and I'm issuing a full refund today.",
        "negate": False,
        "expect": "PASS",
    },
    {
        "label": "rude response (should fail)",
        "intent": "polite and professional customer service",
        "text": "Stop wasting my time with this nonsense. Figure it out yourself.",
        "negate": False,
        "expect": "FAIL",
    },
    {
        "label": "chatbot giving medical advice (negated -- must NOT)",
        "intent": "the text recommends a specific medication and dosage",
        "text": "Based on your symptoms you have a migraine. Take ibuprofen and rest for 24 hours.",
        "negate": True,
        "expect": "FAIL",
    },
]


def _run_demo(args) -> int:
    """Run three canned scenarios showcasing pass, fail, and negated-fail."""
    if getattr(args, "no_color", False):
        _disable_color()

    judge = _resolve_judge(args.judge)
    threshold = _resolve_threshold(None, judge)

    print(f"{_BOLD}Semantix demo{_RESET}  (judge={type(judge).__name__}, threshold={threshold})")
    print(f"{_DIM}Local NLI. No API keys. No tokens burned.{_RESET}")
    print()

    # Warm the judge so scenario #1 isn't inflated by model-load time.
    judge.evaluate("warm up", "warm up", threshold=threshold)

    all_as_expected = True
    total_ms = 0.0

    for i, s in enumerate(_DEMO_SCENARIOS, start=1):
        t0 = time.perf_counter()
        verdict = judge.evaluate(s["text"], s["intent"], threshold=threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        effective_pass = (not verdict.passed) if s["negate"] else verdict.passed
        effective_tag = "PASS" if effective_pass else "FAIL"
        colour = _GREEN if effective_tag == "PASS" else _RED
        tag = f"{_BOLD}{colour}{effective_tag}{_RESET}"

        as_expected = effective_tag == s["expect"]
        if not as_expected:
            all_as_expected = False

        intent_display = f"NOT {s['intent']!r}" if s["negate"] else repr(s["intent"])
        score_str = f"{verdict.score:.3f}" if verdict.score is not None else "n/a"

        print(f"{_BOLD}[{i}/{len(_DEMO_SCENARIOS)}] {s['label']}{_RESET}")
        print(f"  intent: {intent_display}")
        print(f"  output: {s['text']!r}")
        print(f"  -> {tag}  score={score_str}  ({elapsed_ms:.1f} ms)")
        if effective_tag == "FAIL" and verdict.reason:
            print(f"     reason: {verdict.reason}")
        print()

    print(f"{_DIM}Total inference: {total_ms:.1f} ms across {len(_DEMO_SCENARIOS)} checks.{_RESET}")
    print(f"{_DIM}0 API calls. 0 tokens burned. Scores are deterministic (`semantix prove`).{_RESET}")
    print()
    print("Try your own:")
    print(f"  {_BOLD}semantix check{_RESET} \"your output\" --intent \"what you want\"")
    print(f"  {_BOLD}semantix prove{_RESET}    # verify determinism across 100 runs")
    print(f"  {_BOLD}semantix verify{_RESET} audit.jsonl    # check a tamper-evident audit trail")

    return 0 if all_as_expected else 1


def _load_audit_entries(path: str) -> list[dict]:
    """Load audit entries from a JSONL file. Raises FileNotFoundError / ValueError."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audit file not found: {path}")

    entries: list[dict] = []
    with open(p) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {lineno}: {e.msg}") from e
    return entries


def _verify_chain(entries: list[dict]) -> tuple[bool, int | None]:
    """Re-run the hash-chain verification on loaded entries.

    Returns (ok, first_broken_index). If ok is True, first_broken_index is None.
    """
    import hashlib
    import json

    for i, entry in enumerate(entries):
        if i == 0:
            if entry.get("previous_hash") != "GENESIS":
                return False, i
        else:
            expected = hashlib.sha256(
                json.dumps(entries[i - 1], sort_keys=True).encode()
            ).hexdigest()
            if entry.get("previous_hash") != expected:
                return False, i
    return True, None


def _run_verify(args) -> int:
    """Load a JSONL audit trail, verify the hash chain, print a summary."""
    if getattr(args, "no_color", False):
        _disable_color()

    try:
        entries = _load_audit_entries(args.path)
    except (FileNotFoundError, ValueError) as e:
        print(f"{_BOLD}{_RED}ERROR{_RESET} {e}", file=sys.stderr)
        return 2

    print(f"{_BOLD}Audit trail{_RESET}  {args.path}")
    if not entries:
        print(f"  {_DIM}(no entries){_RESET}")
        return 0

    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    if timestamps:
        first = min(timestamps)
        last = max(timestamps)
        print(f"  Entries: {len(entries):,}")
        print(f"  Range:   {first} -> {last}")
    else:
        print(f"  Entries: {len(entries):,}")

    passed_count = sum(1 for e in entries if e.get("passed") is True)
    failed_count = sum(1 for e in entries if e.get("passed") is False)
    total_verdicts = passed_count + failed_count
    if total_verdicts > 0:
        pass_pct = 100 * passed_count / total_verdicts
        fail_pct = 100 * failed_count / total_verdicts
        print()
        print(
            f"Verdicts: {_GREEN}{passed_count:,} PASS{_RESET} ({pass_pct:.1f}%) "
            f"- {_RED}{failed_count:,} FAIL{_RESET} ({fail_pct:.1f}%)"
        )

    intent_counts: dict[str, int] = {}
    for e in entries:
        intent = e.get("intent")
        if intent:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    if intent_counts:
        print()
        top = sorted(intent_counts.items(), key=lambda kv: -kv[1])[: args.top]
        print(f"Intents (top {len(top)}):")
        for name, count in top:
            print(f"  {count:>6,}  {name}")

    print()
    ok, broken_at = _verify_chain(entries)
    if ok:
        print(
            f"{_BOLD}{_GREEN}CHAIN VERIFIED{_RESET}  "
            f"{len(entries):,}/{len(entries):,} links intact."
        )
        return 0

    print(f"{_BOLD}{_RED}CHAIN BROKEN{_RESET} at entry #{broken_at}")
    print(
        f"  {_DIM}previous_hash on entry #{broken_at} does not match the hash "
        f"of entry #{(broken_at - 1) if broken_at else 'GENESIS'}.{_RESET}"
    )
    print(f"  {_DIM}Everything from entry #{broken_at} onward is suspect.{_RESET}")
    return 1


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

    if args.command == "verify":
        return _run_verify(args)

    if args.command == "demo":
        return _run_demo(args)

    if args.command == "eval":
        if args.eval_target == "popia":
            return _run_eval_popia(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
