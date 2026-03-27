"""Semantix — Usage Examples
============================

Run any example with:  python examples.py
Requires:  pip install openai sentence-transformers
"""

from __future__ import annotations

# ── 1.  Define your Intent types ────────────────────────────────────────

from semantix import (
    EmbeddingJudge,
    Intent,
    LLMJudge,
    SemanticIntentError,
    validate_intent,
)


class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude or aggressive."""


class PositiveSentiment(Intent):
    """The text must express a clearly positive, optimistic, or encouraging sentiment."""
    threshold = 0.85          # stricter than the default 0.8


class FactualSummary(Intent):
    """The text must be a concise, factual summary without opinions or speculation."""


# ── 2.  Use with the default LLM judge (OpenAI) ────────────────────────

@validate_intent                       # uses LLMJudge() by default
def decline_event(event: str) -> ProfessionalDecline:
    """Simulate an LLM call that should politely decline."""
    # In production, this would be your actual LLM call:
    #   return openai.chat.completions.create(...).choices[0].message.content
    return (
        f"Thank you so much for the invitation to {event}. "
        "Unfortunately I won't be able to attend, but I truly appreciate "
        "you thinking of me. I hope it goes wonderfully!"
    )


# ── 3.  Use with the fast Embedding judge ──────────────────────────────

@validate_intent(judge=EmbeddingJudge())
def summarize(text: str) -> FactualSummary:
    """Simulate an LLM call that should produce a factual summary."""
    return f"The document discusses {text[:60]}. Key points include data analysis and results."


# ── 4.  Async support works out of the box ──────────────────────────────

@validate_intent(judge=EmbeddingJudge())
async def async_encourage(name: str) -> PositiveSentiment:
    return f"You're doing amazing work, {name}! Keep pushing forward — the best is yet to come."


# ── 5.  Demonstrating a validation failure ──────────────────────────────

@validate_intent(judge=EmbeddingJudge())
def bad_decline(event: str) -> ProfessionalDecline:
    """This will FAIL because the output is hostile, not polite."""
    return f"No way I'm going to your stupid {event}. Leave me alone."


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    # Example with EmbeddingJudge (no API key needed)
    print("=== Factual Summary (Embedding Judge) ===")
    result = summarize("machine learning trends in healthcare")
    print(f"  Intent type : {type(result).__name__}")
    print(f"  Text        : {result.text}\n")

    # Async example
    print("=== Positive Sentiment — async (Embedding Judge) ===")
    result2 = asyncio.run(async_encourage("Alice"))
    print(f"  Intent type : {type(result2).__name__}")
    print(f"  Text        : {result2.text}\n")

    # Validation failure
    print("=== Expected Failure ===")
    try:
        bad_decline("birthday party")
    except SemanticIntentError as exc:
        print(f"  Caught: {exc}\n")

    # LLM judge example (requires OPENAI_API_KEY)
    import os

    if os.environ.get("OPENAI_API_KEY"):
        print("=== Professional Decline (LLM Judge) ===")
        result3 = decline_event("the company retreat")
        print(f"  Intent type : {type(result3).__name__}")
        print(f"  Text        : {result3.text}\n")
    else:
        print("=== Skipping LLM Judge example (set OPENAI_API_KEY to enable) ===\n")
