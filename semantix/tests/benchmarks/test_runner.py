from benchmarks.common.judges import JudgeResult
from benchmarks.common.runner import Example, run_agreement


class StubJudge:
    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self._score = score
        self.calls: list[tuple[str, str]] = []

    def evaluate(self, text: str, intent: str) -> JudgeResult:
        self.calls.append((text, intent))
        return JudgeResult(
            score=self._score,
            latency_ms=1.0,
            cost_usd=0.0,
            paid_equivalent_usd=0.0,
        )


def test_run_agreement_calls_every_judge_once_per_example():
    examples = [
        Example(example_id="1", text="hello", intent="polite"),
        Example(example_id="2", text="goodbye", intent="polite"),
    ]
    judges = [StubJudge("A", 0.9), StubJudge("B", 0.7)]
    rows = run_agreement(examples, judges)
    assert len(rows) == 4  # 2 examples × 2 judges
    assert all(r.experiment == "agreement" for r in rows)
    assert all(j_calls == 2 for j_calls in (len(j.calls) for j in judges))


def test_run_agreement_preserves_example_and_intent():
    examples = [Example(example_id="x", text="t", intent="i")]
    judges = [StubJudge("A", 0.5)]
    rows = run_agreement(examples, judges)
    assert rows[0].example_id == "x"
    assert rows[0].intent == "i"
    assert rows[0].text == "t"
    assert rows[0].judge == "A"
    assert rows[0].score == 0.5
