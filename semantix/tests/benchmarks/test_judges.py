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


import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from benchmarks.common.judges import GroqJudge

FIXTURES = Path(__file__).parent / "fixtures"


@respx.mock
def test_groq_judge_parses_numeric_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=Response(200, json=json.loads((FIXTURES / "groq_response.json").read_text()))
    )
    judge = GroqJudge()
    result = judge.evaluate("Thank you for reaching out.", "The text must be polite.")
    assert result.score == 0.9
    assert result.latency_ms > 0
    assert result.cost_usd == 0.0
    assert result.paid_equivalent_usd > 0  # Paid-tier rate applied even in test
    assert result.error is None


@respx.mock
def test_groq_judge_handles_429_with_retry(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=[
            Response(429, headers={"Retry-After": "0"}),
            Response(200, json=json.loads((FIXTURES / "groq_response.json").read_text())),
        ]
    )
    judge = GroqJudge()
    result = judge.evaluate("test", "test")
    assert result.score == 0.9
    assert result.error is None


@respx.mock
def test_groq_judge_records_error_on_non_numeric_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    bad = json.loads((FIXTURES / "groq_response.json").read_text())
    bad["choices"][0]["message"]["content"] = "I cannot comply."
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=Response(200, json=bad)
    )
    judge = GroqJudge()
    result = judge.evaluate("test", "test")
    assert result.score != result.score  # NaN
    assert result.error is not None
