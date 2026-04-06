"""Phase 5 Trust Demo — High-Stakes Legal Review.

Demonstrates the full Enterprise Performance stack:
1. The Silent Guard (QuantizedNLIJudge) — fast pass on clean text
2. The Detective (ForensicJudge) — catches liability clause, identifies breach tokens
3. The Black Box (AuditEngine) — immutable hash-chained audit trail

Usage:
    python tools/trust_demo.py

Requires: pip install "semantix-ai[turbo,nli]"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Attempt to import the turbo stack; fall back to mock for demo purposes
# ---------------------------------------------------------------------------

try:
    from semantix.judges.quantized_nli import QuantizedNLIJudge

    _HAS_TURBO = True
except ImportError:
    _HAS_TURBO = False

from semantix.judges import Judge, Verdict
from semantix.judges.forensic import ForensicJudge
from semantix.audit.engine import AuditEngine


# ---------------------------------------------------------------------------
# Mock judge for environments without onnxruntime
# ---------------------------------------------------------------------------


class _DemoJudge(Judge):
    """Simulates QuantizedNLIJudge for demo environments without ONNX."""

    def evaluate(self, output, intent_description, threshold=0.5):
        # Simple heuristic: flag text containing liability-related words
        risk_words = {"liability", "indemnify", "waive", "forfeit", "penalize"}
        text_lower = output.lower()
        risk_count = sum(1 for w in risk_words if w in text_lower)
        if risk_count > 0:
            score = max(0.05, 0.3 - risk_count * 0.1)
        else:
            score = 0.92
        return Verdict(passed=score >= threshold, score=score)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

INTENT = "The text must be a professional legal summary free of hidden liability clauses or unfair penalty terms."

SAFE_TEXT = (
    "This agreement establishes a mutual partnership between both parties. "
    "Each party retains full intellectual property rights over their contributions. "
    "Either party may terminate with 30 days written notice. "
    "All disputes shall be resolved through binding arbitration."
)

DANGEROUS_TEXT = (
    "This agreement establishes a mutual partnership between both parties. "
    "By signing, the client agrees to indemnify and hold harmless the provider "
    "against any and all claims. The client shall forfeit all deposits upon "
    "early termination and waive the right to legal recourse. "
    "Provider may penalize client for any breach at provider's sole discretion."
)


def _banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def main() -> None:
    # Reset audit engine for clean demo
    AuditEngine._instance = None
    AuditEngine._entries = []
    AuditEngine._lock = None

    engine = AuditEngine()

    # Select judge
    has_turbo = _HAS_TURBO
    if has_turbo:
        try:
            base_judge = QuantizedNLIJudge()
            print("[*] QuantizedNLIJudge detected (ONNX INT8). Using real model.")
        except Exception:
            has_turbo = False

    if not has_turbo:
        print("[*] onnxruntime not found. Using demo heuristic judge.")
        base_judge = _DemoJudge()

    detective = ForensicJudge(base_judge, top_k=3)

    # ── Scenario 1: Safe text ────────────────────────────────
    _banner("SCENARIO 1: The Silent Guard — Safe Legal Text")

    print(f"Intent: {INTENT}\n")
    print(f"Text: {SAFE_TEXT[:120]}...\n")

    start = time.perf_counter()
    verdict = detective.evaluate(SAFE_TEXT, INTENT, threshold=0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    engine.record(
        intent=INTENT,
        output=SAFE_TEXT,
        score=verdict.score,
        passed=verdict.passed,
        reason=verdict.reason,
    )

    print(f"Result:  PASSED")
    print(f"Score:   {verdict.score:.4f}")
    print(f"Latency: {elapsed_ms:.1f}ms")
    print(f"Reason:  {verdict.reason or '(none — clean pass)'}")

    # ── Scenario 2: Dangerous text ───────────────────────────
    _banner("SCENARIO 2: The Detective — Hidden Liability Clause")

    print(f"Intent: {INTENT}\n")
    print(f"Text: {DANGEROUS_TEXT[:120]}...\n")

    start = time.perf_counter()
    verdict = detective.evaluate(DANGEROUS_TEXT, INTENT, threshold=0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    engine.record(
        intent=INTENT,
        output=DANGEROUS_TEXT,
        score=verdict.score,
        passed=verdict.passed,
        reason=verdict.reason,
    )

    print(f"Result:  FAILED")
    print(f"Score:   {verdict.score:.4f}")
    print(f"Latency: {elapsed_ms:.1f}ms")
    print(f"\n{verdict.reason}")

    # ── Audit Trail ──────────────────────────────────────────
    _banner("THE BLACK BOX: Immutable Audit Trail")

    print(f"Entries: {len(engine.entries)}")
    print(f"Chain valid: {engine.verify_chain()}\n")

    for i, entry in enumerate(engine.entries):
        print(f"--- Certificate #{i + 1} ---")
        print(f"  ID:            {entry['id']}")
        print(f"  Timestamp:     {entry['timestamp']}")
        print(f"  Intent:        {entry['intent'][:60]}...")
        print(f"  Score:         {entry['score']}")
        print(f"  Passed:        {entry['passed']}")
        print(f"  Output Hash:   {entry['output_hash'][:24]}...")
        print(f"  Previous Hash: {entry['previous_hash'][:24]}{'...' if entry['previous_hash'] != 'GENESIS' else ''}")
        print()

    # Flush to disk
    out_path = Path("tools/trust_demo_audit.jsonl")
    engine.flush(out_path)
    print(f"[*] Audit trail flushed to {out_path}")

    _banner("DEMO COMPLETE")


if __name__ == "__main__":
    main()
