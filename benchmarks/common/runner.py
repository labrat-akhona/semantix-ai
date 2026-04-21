"""Benchmark execution loops: reward-agreement and optimization-impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.common.io import Row
from benchmarks.common.judges import Judge, JudgeResult


@dataclass(frozen=True)
class Example:
    example_id: str
    text: str           # The generated output to be judged
    intent: str
    input: dict[str, Any] | None = None  # DSPy program input, used in optimization mode


def run_agreement(examples: list[Example], judges: list[Judge]) -> list[Row]:
    """Every judge scores every example once. Produces one Row per (example, judge)."""
    rows: list[Row] = []
    for ex in examples:
        for judge in judges:
            result = judge.evaluate(ex.text, ex.intent)
            rows.append(_row("agreement", ex, judge.name, result))
    return rows


def run_optimization(
    examples: list[Example],
    *,
    program_fn,            # Callable: (input_dict, reward_fn) -> final_text
    reward_judges: list[Judge],
    final_judge: Judge,
    on_example_done=None,  # Optional callback(ex_rows: list[Row]) after each example completes
) -> list[Row]:
    """Run BestOfN via `program_fn` for each reward judge; final output scored by final_judge.

    program_fn receives an example's input dict and a reward_fn(output) -> float,
    and returns the selected final text. This indirection keeps DSPy out of the
    runner's import surface so the module stays testable without DSPy installed.

    If ``on_example_done`` is provided, it is invoked with the rows produced for each
    example after that example completes — enabling incremental disk writes so a
    mid-flight crash only loses the current example's work.
    """
    rows: list[Row] = []
    for ex in examples:
        ex_rows: list[Row] = []
        for reward_judge in reward_judges:
            def reward_fn(output: str, *, _judge=reward_judge, _intent=ex.intent) -> float:
                score = _judge.evaluate(output, _intent).score
                return 0.0 if score != score else float(score)  # NaN-safe for DSPy
            final_text = program_fn(ex.input or {}, reward_fn)
            result = final_judge.evaluate(final_text, ex.intent)
            ex_rows.append(
                _row(
                    "optimization",
                    Example(example_id=ex.example_id, text=final_text, intent=ex.intent),
                    f"final_judge::{final_judge.name}__reward::{reward_judge.name}",
                    result,
                )
            )
        rows.extend(ex_rows)
        if on_example_done is not None:
            on_example_done(ex_rows)
    return rows


def _row(experiment: str, ex: Example, judge_name: str, result: JudgeResult) -> Row:
    return Row(
        example_id=ex.example_id,
        experiment=experiment,
        judge=judge_name,
        intent=ex.intent,
        text=ex.text,
        score=result.score,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        paid_equivalent_usd=result.paid_equivalent_usd,
        raw=result.raw,
        error=result.error,
    )
