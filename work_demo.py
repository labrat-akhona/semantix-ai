"""Semantix — Work Demo
======================

Three concrete use cases tailored for teams using AI in:
  1. Chatbots       — validate tone, safety, and helpfulness before responding
  2. Data Analysis  — validate AI-generated insights are factual, not speculative
  3. Automations    — validate AI-generated action descriptions are safe and scoped

Run with:
    pip install "semantix[embeddings]"
    python work_demo.py

No API key required — uses the local EmbeddingJudge.
"""

from __future__ import annotations

import logging

from semantix import (
    NLIJudge,
    Intent,
    SemanticIntentError,
    get_last_failure,
    validate_intent,
)

# Uncomment to see per-validation scores and latency:
# logging.basicConfig(level=logging.INFO)

judge = NLIJudge()

# ─────────────────────────────────────────────────────────────────────────────
# USE CASE 1: CHATBOTS
# Problem: Your customer support bot occasionally gives off-tone or unhelpful
# replies. You can't manually review every message. Semantix catches bad
# outputs before they reach users.
# ─────────────────────────────────────────────────────────────────────────────


class HelpfulCustomerReply(Intent):
    """The text must directly address the customer's problem with a clear,
    actionable solution or next step. It must be empathetic and professional."""
    threshold = 0.7


class SafeReply(Intent):
    """The text must not contain harmful advice, discriminatory language,
    or anything that could expose the company to legal or reputational risk."""
    threshold = 0.7


# Composite: must be BOTH helpful AND safe to pass
HelpfulAndSafe = HelpfulCustomerReply & SafeReply


@validate_intent(judge=judge, retries=1)
def chatbot_reply(customer_message: str) -> HelpfulAndSafe:
    """Simulates your chatbot LLM call."""
    # In production: return your_llm_client.chat(customer_message)
    # With retry feedback:
    hint = ""
    if failure := get_last_failure():
        hint = f" [Previous attempt scored {failure.score:.2f} — improve helpfulness]"
    _ = hint  # In production, append this to your LLM prompt
    return (
        "I completely understand your frustration. "
        "To resolve your billing issue, please go to Account Settings → Billing "
        "and click 'Dispute Charge'. Our team will review it within 24 hours. "
        "Let me know if you need further help!"
    )


@validate_intent(judge=judge)
def bad_chatbot_reply(customer_message: str) -> HelpfulAndSafe:
    """Simulates a bad bot reply — will fail validation."""
    return "That's not our problem. Read the FAQ."


# ─────────────────────────────────────────────────────────────────────────────
# USE CASE 2: DATA ANALYSIS
# Problem: Your AI generates summaries and insights from data. Without
# validation, speculative or hallucinated claims go into reports unchecked.
# ─────────────────────────────────────────────────────────────────────────────


class FactualDataInsight(Intent):
    """The text must describe observable patterns or trends from data.
    It must not include speculation, predictions, or personal opinions."""


class ExecutiveSummary(Intent):
    """The text must be a concise summary (2-4 sentences) covering the main
    findings. It must be suitable for a non-technical executive audience."""


@validate_intent(judge=judge)
def generate_insight(dataset_description: str) -> FactualDataInsight:
    """Simulates an AI generating a data insight."""
    return (
        f"Analysis of {dataset_description} shows a 23% increase in user engagement "
        "over the past quarter. The highest activity was recorded on Tuesdays between "
        "10am–12pm. Conversion rates improved from 3.1% to 4.8% across all segments."
    )


@validate_intent(judge=judge)
def generate_executive_summary(report_content: str) -> ExecutiveSummary:
    """Simulates an AI generating a summary for leadership."""
    return (
        "This quarter's results exceeded targets across all key metrics. "
        "Revenue grew 18% year-on-year, driven by strong performance in the enterprise segment. "
        "Customer retention improved to 94%, the highest in company history."
    )


@validate_intent(judge=judge)
def bad_insight(dataset_description: str) -> FactualDataInsight:
    """Simulates a hallucinated/speculative insight — will fail validation."""
    return (
        f"I believe {dataset_description} will probably lead to massive growth next year. "
        "In my opinion, this is a once-in-a-decade opportunity that the company must seize."
    )


# ─────────────────────────────────────────────────────────────────────────────
# USE CASE 3: AUTOMATIONS
# Problem: Your AI generates descriptions of automated actions to execute
# (e.g., sending emails, modifying records, triggering workflows). A badly
# worded or out-of-scope action could cause real damage. Validate before run.
# ─────────────────────────────────────────────────────────────────────────────


class ScopedAutomationAction(Intent):
    """The text must describe a single, specific, reversible action with a
    clear target and scope. It must not describe bulk operations, deletions,
    or actions affecting production systems without explicit approval."""


class AutomationConfirmationMessage(Intent):
    """The text must clearly confirm what action was taken, what was affected,
    and provide a reference ID or timestamp. It must be concise and factual."""


@validate_intent(judge=judge)
def describe_automation(trigger: str) -> ScopedAutomationAction:
    """Simulates AI generating an automation action description."""
    return (
        f"Send a follow-up email to the single lead '{trigger}' "
        "with the Q1 proposal PDF attached. "
        "Schedule delivery for 9am their local time tomorrow."
    )


@validate_intent(judge=judge)
def confirm_automation(action_taken: str) -> AutomationConfirmationMessage:
    """Simulates AI generating a confirmation after an automation runs."""
    return (
        f"Action completed: {action_taken}. "
        "Affected records: 1 (lead ID: #48291). "
        "Timestamp: 2026-03-27T09:00:00Z. Reference: AUTO-2891."
    )


@validate_intent(judge=judge)
def dangerous_automation(trigger: str) -> ScopedAutomationAction:
    """Simulates a dangerous/out-of-scope automation — will fail validation."""
    return f"Delete all records matching '{trigger}' from the production database immediately."


# ─────────────────────────────────────────────────────────────────────────────
# DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL (caught)"


def run(label: str, fn, *args):
    print(f"  [{label}]")
    try:
        result = fn(*args)
        print(f"  Status  : {PASS}")
        print(f"  Intent  : {type(result).__name__}")
        print(f"  Output  : {result.text[:120]}")
    except SemanticIntentError as e:
        print(f"  Status  : {FAIL}")
        print(f"  Intent  : {e.intent_name}")
        score_str = f"{e.score:.4f}" if e.score is not None else "N/A"
        print(f"  Score   : {score_str}  ← below threshold, blocked before sending")
    print()


if __name__ == "__main__":
    sep = "─" * 60

    print(f"\n{sep}")
    print("  USE CASE 1: CHATBOT REPLY VALIDATION")
    print(f"  Validate tone & helpfulness before reply reaches user")
    print(sep)
    run("Good reply  ", chatbot_reply, "I was charged twice on my last invoice")
    run("Bad reply   ", bad_chatbot_reply, "I was charged twice on my last invoice")

    print(f"{sep}")
    print("  USE CASE 2: DATA ANALYSIS INSIGHT VALIDATION")
    print(f"  Catch speculation and hallucination in AI-generated reports")
    print(sep)
    run("Good insight", generate_insight, "Q1 user engagement data")
    run("Good summary", generate_executive_summary, "Q1 performance report")
    run("Bad insight ", bad_insight, "Q1 user engagement data")

    print(f"{sep}")
    print("  USE CASE 3: AUTOMATION ACTION VALIDATION")
    print(f"  Gate dangerous or out-of-scope AI actions before execution")
    print(sep)
    run("Good action ", describe_automation, "Acme Corp renewal lead")
    run("Good confirm", confirm_automation, "Follow-up email sent to Acme Corp")
    run("Bad action  ", dangerous_automation, "churned customers")

    print(f"{sep}")
    print("  SUMMARY")
    print(sep)
    print("  Semantix validated every output before it could cause harm.")
    print("  Bad outputs raised SemanticIntentError — caught, not shipped.")
    print(f"  Install: pip install 'semantix-ai[embeddings]'")
    print(f"{sep}\n")
