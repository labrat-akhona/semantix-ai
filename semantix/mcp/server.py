"""Semantix MCP server — exposes verify_text_intent as an MCP tool."""

from __future__ import annotations

import functools
import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Semantix-Verify")


@functools.lru_cache(maxsize=1)
def _get_judge():
    """Lazy-load NLIJudge singleton (model loads on first call, not at import)."""
    from semantix.judges.nli import NLIJudge

    return NLIJudge()


def _build_correction(text: str, intent_description: str, score: float, threshold: float) -> str:
    """Structured feedback mirroring decorator.py:79-105 _build_feedback()."""
    score_str = f"{score:.4f}"
    return (
        f"## Semantix Verification Failed\n\n"
        f"### What went wrong\n"
        f"- **Score:** {score_str} (threshold {threshold} not met)\n\n"
        f"### What is required\n"
        f"{intent_description.strip()}\n\n"
        f"### Rejected output\n"
        f"```\n{text[:400]}\n```\n\n"
        f"Please generate a new response that satisfies the requirement above."
    )


@mcp.tool()
def verify_text_intent(text: str, intent_description: str, threshold: float = 0.5) -> str:
    """Check whether text satisfies a semantic intent using NLI.

    Args:
        text: The text to verify.
        intent_description: What the text should convey.
        threshold: Minimum entailment score to pass (0-1, default 0.5).

    Returns:
        JSON with score, passed, reason, and correction_suggestion on failure.
    """
    try:
        judge = _get_judge()
    except ImportError:
        return json.dumps(
            {
                "error": "sentence-transformers not installed. Run: pip install semantix-ai[nli]",
                "passed": False,
            }
        )

    verdict = judge.evaluate(text, intent_description, threshold)
    result = {"score": verdict.score, "passed": verdict.passed, "reason": verdict.reason}

    if not verdict.passed:
        result["correction_suggestion"] = _build_correction(
            text, intent_description, verdict.score, threshold
        )

    return json.dumps(result)


if __name__ == "__main__":
    mcp.run()
