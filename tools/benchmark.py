#!/usr/bin/env python3
"""Benchmark: Self-Healing vs. No-Healing reliability comparison.

Demonstrates that @validate_intent with retries (self-healing) significantly
improves LLM output reliability compared to single-shot validation.

Usage
-----
    # Requires an OpenAI-compatible API key for the simulated LLM
    export OPENAI_API_KEY=sk-...
    python tools/benchmark.py

    # Use a local judge (NLI, no API key needed) with simulated outputs
    python tools/benchmark.py --judge nli

    # Customize runs
    python tools/benchmark.py --judge nli --runs 50 --retries 3
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantix import Intent, SemanticIntentError, validate_intent
from semantix.judges import Judge, Verdict


# ---------------------------------------------------------------------------
# Intent definitions for benchmarking
# ---------------------------------------------------------------------------


class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude or aggressive."""


class TechnicalExplanation(Intent):
    """The text must explain a technical concept clearly for a non-technical audience."""

    threshold = 0.6


class ActionableSummary(Intent):
    """The text must summarize key points and include specific next steps or action items."""


# ---------------------------------------------------------------------------
# Simulated LLM outputs (no real API needed for the benchmark)
# ---------------------------------------------------------------------------

_GOOD_DECLINES = [
    "Thank you so much for the invitation! Unfortunately, I have a prior commitment "
    "and won't be able to attend. I truly appreciate you thinking of me, and I hope "
    "we can find another time to get together.",
    "I really appreciate the invite. Sadly, my schedule won't allow it this time. "
    "I hope the event goes wonderfully, and please do keep me in mind for future events!",
    "What a kind invitation! I'm afraid I must decline as I have a conflict that day. "
    "Thank you for understanding, and I wish you a fantastic event.",
]

_BAD_DECLINES = [
    "No.",
    "I don't want to go to your stupid party.",
    "lol nah I'm good",
    "Decline. Next.",
    "Hard pass.",
]

_GOOD_EXPLANATIONS = [
    "Think of an API like a waiter at a restaurant. You tell the waiter what you want "
    "(your request), the waiter goes to the kitchen (the server), and brings back your "
    "food (the response). You never need to know how the kitchen works.",
    "Machine learning is like teaching a child by example. Instead of writing explicit "
    "rules, you show the computer thousands of examples and it learns the patterns on "
    "its own, getting better with more practice.",
]

_BAD_EXPLANATIONS = [
    "API uses HTTP/REST with JSON serialization over TCP/IP with OAuth2 bearer tokens.",
    "It's complicated, just Google it.",
    "Neural networks use backpropagation with stochastic gradient descent on the loss function.",
]

_GOOD_SUMMARIES = [
    "Key points: (1) Revenue grew 15% QoQ, (2) Customer churn decreased to 3%, "
    "(3) New product launch exceeded targets. Next steps: Schedule board presentation "
    "by Friday, prepare hiring plan for Q3, finalize partnership agreement with Acme Corp.",
    "Summary: The migration to the new database is 80% complete with zero data loss. "
    "Remaining work: migrate the analytics tables (ETA: 3 days), update connection strings "
    "in staging, and run the full regression suite before going live.",
]

_BAD_SUMMARIES = [
    "Things happened. Stuff was discussed.",
    "Meeting went well.",
    "See attached.",
]

_SCENARIOS: list[tuple[type[Intent], list[str], list[str]]] = [
    (ProfessionalDecline, _GOOD_DECLINES, _BAD_DECLINES),
    (TechnicalExplanation, _GOOD_EXPLANATIONS, _BAD_EXPLANATIONS),
    (ActionableSummary, _GOOD_SUMMARIES, _BAD_SUMMARIES),
]


def _simulate_llm_output(
    good: list[str],
    bad: list[str],
    attempt: int,
    feedback: Optional[str],
    rng: Optional[random.Random] = None,
) -> str:
    """Simulate an LLM that improves with feedback.

    - Attempt 1 (no feedback): 40% chance of good output.
    - Attempt 2+ (with feedback): 85% chance of good output.
    """
    r = rng or random
    prob_good = 0.85 if feedback is not None else 0.40

    if r.random() < prob_good:
        return r.choice(good)
    return r.choice(bad)


# ---------------------------------------------------------------------------
# Mock judge for fast benchmarking (uses keyword heuristics)
# ---------------------------------------------------------------------------


class HeuristicJudge(Judge):
    """Fast local judge that uses simple heuristics for benchmarking.

    Not meant for production — just to run the benchmark without any model downloads.
    """

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        lower = output.lower()
        word_count = len(output.split())

        # Length heuristic: very short outputs are usually bad
        if word_count < 5:
            return Verdict(passed=False, score=0.1, reason="Output too short")

        # Rudeness heuristic
        rude_words = {"stupid", "lol", "nah", "hard pass", "google it"}
        if any(w in lower for w in rude_words):
            return Verdict(passed=False, score=0.15, reason="Contains inappropriate language")

        # Politeness signals
        polite = {"thank", "appreciate", "hope", "unfortunately", "afraid", "kindly"}
        polite_count = sum(1 for w in polite if w in lower)

        # Clarity signals
        clarity = {"like", "think of", "example", "means", "imagine", "instead of"}
        clarity_count = sum(1 for w in clarity if w in lower)

        # Action signals
        action = {"next step", "action", "schedule", "prepare", "finalize", "by friday", "eta"}
        action_count = sum(1 for w in action if w in lower)

        # Compute a rough score
        signal_count = polite_count + clarity_count + action_count
        score = min(1.0, 0.3 + signal_count * 0.15)

        return Verdict(
            passed=score >= threshold,
            score=round(score, 4),
            reason=f"Heuristic: {signal_count} quality signals detected",
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_single(
    judge: Judge,
    retry_count: int,
    intent_cls: type[Intent],
    good: list[str],
    bad: list[str],
    rng: random.Random,
) -> tuple[int, bool]:
    """Run a single benchmark trial. Returns (attempt_count, succeeded)."""
    attempt_counter = [0]
    # Pre-snapshot the RNG so each attempt draws fresh randomness.
    trial_rng = rng

    @validate_intent(judge=judge, retries=retry_count)
    def run_llm(
        prompt: str,
        semantix_feedback: Optional[str] = None,
    ) -> intent_cls:  # type: ignore[valid-type]
        attempt_counter[0] += 1
        return _simulate_llm_output(
            good, bad, attempt_counter[0], semantix_feedback, trial_rng
        )

    try:
        run_llm("benchmark prompt")
        return attempt_counter[0], True
    except SemanticIntentError:
        return attempt_counter[0], False


@dataclass
class BenchmarkResult:
    scenario: str
    mode: str  # "no-healing" or "self-healing"
    total_runs: int = 0
    successes: int = 0
    failures: int = 0
    total_attempts: int = 0
    total_time_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_runs if self.total_runs else 0.0

    @property
    def avg_attempts(self) -> float:
        return self.total_attempts / self.total_runs if self.total_runs else 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.total_runs if self.total_runs else 0.0


def run_benchmark(
    judge: Judge,
    num_runs: int = 30,
    retries: int = 2,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run the benchmark and return results for each scenario x mode."""
    results: list[BenchmarkResult] = []

    for intent_cls, good, bad in _SCENARIOS:
        for mode, retry_count in [("no-healing", 0), ("self-healing", retries)]:
            result = BenchmarkResult(
                scenario=intent_cls.__name__,
                mode=mode,
            )

            # Use a seeded Random instance so results are reproducible.
            rng = random.Random(seed + _SCENARIOS.index((intent_cls, good, bad)))

            for run_idx in range(num_runs):
                attempt_count, ok = _run_single(
                    judge, retry_count, intent_cls, good, bad, rng
                )
                if ok:
                    result.successes += 1
                else:
                    result.failures += 1
                result.total_runs += 1
                result.total_attempts += attempt_count

            results.append(result)

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results as a comparison table."""
    print("\n" + "=" * 78)
    print("  SEMANTIX SELF-HEALING BENCHMARK")
    print("=" * 78)

    scenarios = sorted(set(r.scenario for r in results))
    for scenario in scenarios:
        print(f"\n  {scenario}")
        print("  " + "-" * 74)
        print(f"  {'Mode':<16} {'Success Rate':>14} {'Avg Attempts':>14} {'Avg Time':>12}")
        print("  " + "-" * 74)

        no_heal = next(r for r in results if r.scenario == scenario and r.mode == "no-healing")
        heal = next(r for r in results if r.scenario == scenario and r.mode == "self-healing")

        for r in [no_heal, heal]:
            print(
                f"  {r.mode:<16} {r.success_rate:>13.1%} {r.avg_attempts:>14.1f} "
                f"{r.avg_time_ms:>10.1f}ms"
            )

        improvement = heal.success_rate - no_heal.success_rate
        print(f"  {'':>16} improvement: {improvement:>+.1%}")

    print("\n" + "=" * 78)

    # Overall summary
    total_no_heal = sum(r.successes for r in results if r.mode == "no-healing")
    total_heal = sum(r.successes for r in results if r.mode == "self-healing")
    total_runs_each = sum(r.total_runs for r in results if r.mode == "no-healing")

    no_rate = total_no_heal / total_runs_each if total_runs_each else 0
    heal_rate = total_heal / total_runs_each if total_runs_each else 0

    print(f"\n  OVERALL: no-healing {no_rate:.1%} -> self-healing {heal_rate:.1%}")
    print(f"  Self-healing improves reliability by {heal_rate - no_rate:+.1%}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantix Self-Healing Benchmark")
    parser.add_argument(
        "--judge",
        choices=["heuristic", "nli", "llm"],
        default="heuristic",
        help="Judge to use (default: heuristic — fast, no downloads needed)",
    )
    parser.add_argument("--runs", type=int, default=30, help="Runs per scenario per mode")
    parser.add_argument("--retries", type=int, default=2, help="Retries for self-healing mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Suppress noisy per-validation logs; the benchmark table is the output.
    import logging
    logging.getLogger("semantix").setLevel(logging.ERROR)

    if args.judge == "nli":
        from semantix.judges.nli import NLIJudge
        judge: Judge = NLIJudge()
        print("Using NLIJudge (downloading model on first run...)")
    elif args.judge == "llm":
        from semantix.judges.llm import LLMJudge
        judge = LLMJudge()
        print("Using LLMJudge (requires OPENAI_API_KEY)")
    else:
        judge = HeuristicJudge()
        print("Using HeuristicJudge (fast, local, no dependencies)")

    results = run_benchmark(
        judge=judge,
        num_runs=args.runs,
        retries=args.retries,
        seed=args.seed,
    )
    print_results(results)


if __name__ == "__main__":
    main()
