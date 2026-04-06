"""Forensic judge — wraps any judge and explains failures via token attribution.

When the base judge returns ``passed=False``, runs mask-perturbation saliency
to identify the top tokens that drove the contradiction score highest.
Injects a structured Breach Report into ``Verdict.reason``.

On ``passed=True``, returns the base verdict unmodified (zero overhead).
"""

from __future__ import annotations

from semantix.judges import Judge, Verdict

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}


def _mask_perturbation_saliency(
    tokens: list[str],
    score_fn: callable,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Identify tokens whose removal most reduces the contradiction score.

    Parameters
    ----------
    tokens:
        The tokenized output text (individual word/subword strings).
    score_fn:
        A callable that takes a text string and returns a contradiction
        score (float, 0-1). Higher = more contradictory.
    top_k:
        Number of top-contributing tokens to return.

    Returns
    -------
    List of ``(token, score_drop)`` tuples sorted by drop descending.
    """
    if not tokens:
        return []

    # Filter out special tokens and subword fragments
    candidate_indices = [
        i
        for i, t in enumerate(tokens)
        if t not in _SPECIAL_TOKENS and not t.startswith("##")
    ]

    if not candidate_indices:
        return []

    baseline_text = " ".join(tokens)
    baseline_score = score_fn(baseline_text)

    drops: list[tuple[str, float]] = []
    for idx in candidate_indices:
        masked = tokens[:idx] + ["[MASK]"] + tokens[idx + 1 :]
        masked_text = " ".join(masked)
        masked_score = score_fn(masked_text)
        drop = baseline_score - masked_score
        drops.append((tokens[idx], drop))

    drops.sort(key=lambda x: x[1], reverse=True)
    return drops[:top_k]


def _build_breach_report(
    base_verdict: Verdict,
    breach_tokens: list[tuple[str, float]],
) -> str:
    """Format the Breach Report as structured Markdown."""
    score_str = f"{base_verdict.score:.4f}" if base_verdict.score is not None else "N/A"
    base_reason = base_verdict.reason or "No reason provided by base judge"

    if breach_tokens:
        token_list = ", ".join(f"**{tok}** ({drop:.2f})" for tok, drop in breach_tokens)
        token_names = [tok for tok, _ in breach_tokens]
        suspect_line = f"Suspect Tokens: [{', '.join(token_names)}]"
    else:
        token_list = "No individual token dominated the contradiction signal."
        suspect_line = "Suspect Tokens: [none identified]"

    return (
        f"## Breach Report\n\n"
        f"**Score:** {score_str}\n"
        f"**Base judge reason:** {base_reason}\n\n"
        f"### Token Attribution\n"
        f"{token_list}\n\n"
        f"### Summary\n"
        f"Intent failed. High contradiction detected. {suspect_line}"
    )


def _run_forensics(
    output: str,
    intent_description: str,
    base_judge: Judge,
    base_verdict: Verdict,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Run mask-perturbation saliency using the base judge's evaluate method.

    Tokenizes the output with a simple whitespace split (sufficient for
    identifying suspect *words* — we're not doing subword-level attribution).
    """
    tokens = output.split()

    def score_fn(text: str) -> float:
        """Get contradiction-proxy score: 1 - entailment."""
        v = base_judge.evaluate(text, intent_description, threshold=0.0)
        return 1.0 - (v.score or 0.0)

    return _mask_perturbation_saliency(tokens, score_fn, top_k=top_k)


class ForensicJudge(Judge):
    """Wraps any judge and adds token-level attribution on failure.

    On ``passed=True``: returns base verdict unchanged (zero overhead).
    On ``passed=False``: runs mask-perturbation saliency to find the
    top tokens driving the contradiction, then injects a structured
    Breach Report into ``Verdict.reason``.

    Parameters
    ----------
    base_judge:
        Any ``Judge`` instance to wrap.
    top_k:
        Number of breach tokens to identify (default 3).
    """

    def __init__(self, base_judge: Judge, top_k: int = 3) -> None:
        self._base = base_judge
        self._top_k = top_k

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        verdict = self._base.evaluate(output, intent_description, threshold)

        if verdict.passed:
            return verdict

        # Failure path — run forensics
        breach_tokens = _run_forensics(
            output, intent_description, self._base, verdict, self._top_k
        )
        report = _build_breach_report(verdict, breach_tokens)

        return Verdict(
            passed=False,
            score=verdict.score,
            reason=report,
        )
