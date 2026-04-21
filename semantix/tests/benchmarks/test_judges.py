from benchmarks.common.judges import SemantixJudge, JudgeResult


def test_semantix_judge_returns_score_for_polite_text():
    judge = SemantixJudge()
    result = judge.evaluate(
        text="Thank you for reaching out. I'll help you right away.",
        intent="The text must be polite and professional.",
    )
    assert isinstance(result, JudgeResult)
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5  # Should clearly pass
    assert result.latency_ms > 0
    assert result.cost_usd == 0.0
    assert result.paid_equivalent_usd == 0.0
    assert result.error is None
    assert judge.name == "semantix"


def test_semantix_judge_scores_rude_text_lower():
    judge = SemantixJudge()
    rude = judge.evaluate(
        text="Deal with it yourself, not my problem.",
        intent="The text must be polite and professional.",
    )
    polite = judge.evaluate(
        text="I understand — let me help you resolve this.",
        intent="The text must be polite and professional.",
    )
    assert rude.score < polite.score
