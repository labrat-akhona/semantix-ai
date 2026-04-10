#!/usr/bin/env python3
"""Semantix Flywheel Demo — from guardrail to training pipeline in 60 seconds.

This script demonstrates the complete semantix self-improvement loop:

    Validate -> Fail -> Correct -> Capture -> Export -> (Fine-tune)

No API keys required. Uses a simple keyword judge for reliable demonstration.
In production, swap DemoJudge for NLIJudge or QuantizedNLIJudge.

Usage:
    python examples/flywheel_demo.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

# Suppress semantix validation logs for clean demo output
logging.getLogger("semantix").setLevel(logging.ERROR)

from semantix import Intent, Judge, Verdict, validate_intent
from semantix.training import TrainingCollector
from semantix.training.exporters import export_openai


# ---------------------------------------------------------------------------
# Step 1: Define what "correct" means — as a natural language intent
# ---------------------------------------------------------------------------

class ProfessionalDecline(Intent):
    """The text must politely and professionally decline an invitation
    without being rude, aggressive, or dismissive."""


class EncouragingFeedback(Intent):
    """The text must provide constructive, encouraging feedback
    that acknowledges effort and suggests specific improvements."""


# ---------------------------------------------------------------------------
# Step 2: A simple demo judge (swap for NLIJudge in production)
# ---------------------------------------------------------------------------

class DemoJudge(Judge):
    """Keyword-based judge for demonstration purposes.

    Scores text based on presence of polite/rude indicators.
    In production, use NLIJudge or QuantizedNLIJudge for real NLI scoring.
    """

    POLITE = {"thank", "appreciate", "unfortunately", "hope", "great", "effort",
              "solid", "consider", "sorry", "regret", "kindly", "constructive",
              "improve", "suggest", "well done", "good"}
    RUDE = {"terrible", "no way", "sloppy", "didn't try", "do better",
            "hate", "awful", "stupid", "count me out", "waste"}

    def evaluate(self, text: str, description: str, threshold: float) -> Verdict:
        lower = text.lower()
        polite_hits = sum(1 for w in self.POLITE if w in lower)
        rude_hits = sum(1 for w in self.RUDE if w in lower)
        total = polite_hits + rude_hits or 1
        score = polite_hits / total
        passed = score >= threshold
        reason = None if passed else f"Found {rude_hits} negative indicator(s), only {polite_hits} positive"
        return Verdict(passed=passed, score=score, reason=reason)


# ---------------------------------------------------------------------------
# Step 3: Simulate an LLM that improves with feedback
# ---------------------------------------------------------------------------

DECLINE_RESPONSES = [
    # Attempt 1: rude — will fail validation
    "No way, your event sounds terrible. Count me out.",
    # Attempt 2: polite — will pass validation
    "Thank you so much for the invitation. Unfortunately, I have a prior "
    "commitment and won't be able to attend. I hope the event goes well!",
]

FEEDBACK_RESPONSES = [
    # Attempt 1: harsh — will fail validation
    "This work is sloppy. You clearly didn't try. Do better.",
    # Attempt 2: encouraging — will pass validation
    "Great effort on this project! The core structure is solid. "
    "Consider adding error handling in the data processing section "
    "and expanding test coverage for edge cases.",
]


def make_simulated_llm(responses: list[str]):
    """Create a function that returns successive responses, simulating retries."""
    call_count = 0

    def llm_fn(prompt: str, semantix_feedback: Optional[str] = None) -> str:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return responses[idx]

    return llm_fn


# ---------------------------------------------------------------------------
# Step 4: Run the flywheel
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(tempfile.mkdtemp(prefix="semantix_demo_"))
    training_file = output_dir / "training_data.jsonl"
    finetune_file = output_dir / "openai_finetune.jsonl"

    judge = DemoJudge()
    collector = TrainingCollector(training_file)

    print("=" * 65)
    print("  SEMANTIX FLYWHEEL DEMO")
    print("  Validate -> Fail -> Correct -> Capture -> Export")
    print("=" * 65)
    print()

    # --- Scenario 1: Declining an invitation ---

    print("[Scenario 1] Declining an invitation")
    print("-" * 45)

    _decline_llm = make_simulated_llm(DECLINE_RESPONSES)

    @validate_intent(judge=judge, retries=2, collector=collector)
    def decline_invite(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
        return _decline_llm(event, semantix_feedback)

    result = decline_invite("the annual company picnic")
    print(f"  Result: {result}")
    print()

    # --- Scenario 2: Giving feedback ---

    print("[Scenario 2] Giving feedback on student work")
    print("-" * 45)

    _feedback_llm = make_simulated_llm(FEEDBACK_RESPONSES)

    @validate_intent(judge=judge, retries=2, collector=collector)
    def give_feedback(work: str, semantix_feedback: Optional[str] = None) -> EncouragingFeedback:
        return _feedback_llm(work, semantix_feedback)

    result = give_feedback("a first draft of the data pipeline")
    print(f"  Result: {result}")
    print()

    # --- Inspect captured training data ---

    print("=" * 65)
    print("  CAPTURED TRAINING DATA")
    print("=" * 65)
    print()

    stats = collector.stats()
    print(f"  Correction pairs: {stats['total_pairs']}")
    print(f"  Intents:          {list(stats['intents'].keys())}")
    print()

    with open(training_file) as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            rej_score = record['rejected_score']
            acc_score = record['accepted_score']
            print(f"  Pair {i}: {record['intent']}")
            print(f"    REJECTED (score={rej_score:.2f}): {record['rejected_output'][:65]}")
            print(f"    ACCEPTED (score={acc_score:.2f}): {record['accepted_output'][:65]}")
            print()

    # --- Export to OpenAI fine-tuning format ---

    print("=" * 65)
    print("  OPENAI FINE-TUNING EXPORT")
    print("=" * 65)
    print()

    export_openai(training_file, finetune_file)

    with open(finetune_file) as f:
        for i, line in enumerate(f, 1):
            example = json.loads(line)
            msgs = example["messages"]
            print(f"  Example {i}:")
            print(f"    system: {msgs[0]['content'][:70]}...")
            print(f"    user:   {msgs[1]['content']}")
            print(f"    asst:   {msgs[2]['content'][:70]}...")
            print()

    # --- Summary ---

    print("=" * 65)
    print("  THE FLYWHEEL")
    print("=" * 65)
    print()
    print("  Every retry that succeeds after a failure produces a training")
    print("  example. Fine-tune your model on these examples. The fine-tuned")
    print("  model fails less. When it does fail, new pairs are captured.")
    print("  The model improves continuously from its own corrections.")
    print()
    print("    Validate -> Fail -> Correct -> Capture -> Fine-tune")
    print("        ^                                        |")
    print("        +----------------------------------------+")
    print()
    print(f"  Output files:")
    print(f"    {training_file}")
    print(f"    {finetune_file}")
    print()
    print("  In production, swap DemoJudge for NLIJudge (zero API cost):")
    print("    @validate_intent(retries=2, collector=collector)")
    print()


if __name__ == "__main__":
    main()
