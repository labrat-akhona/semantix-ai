"""One-shot: generate 200 (customer_message, intent) pairs balanced across 3 intents.

This is a deterministic synthetic dataset seeded by the curated templates below.
Run once; commit dataset.json. Re-run only if intents/templates change.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

INTENTS = ["polite", "on_topic", "declines_without_being_rude"]

TEMPLATES = {
    "polite": [
        "My package hasn't arrived and I'm very frustrated.",
        "I've been on hold for 30 minutes and still no answer.",
        "This is the third time your service has failed.",
        "Your app crashed in the middle of my order.",
        "The delivery driver left my package in the rain.",
    ],
    "on_topic": [
        "What's the return policy for a laptop I bought last week?",
        "Can I change the delivery address on order #12345?",
        "How do I cancel my subscription?",
        "The charger I received doesn't match my device.",
        "I need a refund for the duplicate charge on my card.",
    ],
    "declines_without_being_rude": [
        "Can I return this item even though it's past the 30-day window?",
        "Will you give me a full refund plus a coupon for my trouble?",
        "Can you ship this overnight for free?",
        "I'd like to cancel my order after it's already been delivered.",
        "Can you waive the restocking fee?",
    ],
}


def main() -> None:
    random.seed(42)
    dataset: list[dict] = []
    target_per_intent = 200 // len(INTENTS) + 1
    for intent in INTENTS:
        templates = TEMPLATES[intent]
        for i in range(target_per_intent):
            dataset.append(
                {
                    "example_id": f"{intent}-{i:03d}",
                    "customer_message": templates[i % len(templates)],
                    "intent_name": intent,
                    "intent_description": {
                        "polite": "The response must be polite and professional.",
                        "on_topic": (
                            "The response must directly address the customer's specific question."
                        ),
                        "declines_without_being_rude": (
                            "The response must decline the request without being rude "
                            "or dismissive."
                        ),
                    }[intent],
                }
            )
    random.shuffle(dataset)
    dataset = dataset[:200]  # exactly 200

    out = Path(__file__).parent / "dataset.json"
    out.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"wrote {len(dataset)} examples to {out}")


if __name__ == "__main__":
    main()
