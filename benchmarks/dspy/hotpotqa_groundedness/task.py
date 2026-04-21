"""DSPy program + HotpotQA subset loader for groundedness benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import dspy
from datasets import load_dataset

from benchmarks.common.runner import Example

INDICES = Path(__file__).parent / "indices.json"
INTENT = "The answer must be grounded in the provided context and not hallucinate facts."


class GroundedAnswer(dspy.Signature):
    """Answer the question strictly using facts from the provided context."""

    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def load_examples() -> list[Example]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    idx_list = json.loads(INDICES.read_text(encoding="utf-8"))
    out: list[Example] = []
    for i in idx_list:
        item = ds[i]
        paras = item["context"]["sentences"]
        context = "\n\n".join(" ".join(s) for s in paras)
        out.append(Example(
            example_id=f"hotpot-{i}",
            text="",
            intent=INTENT,
            input={"context": context, "question": item["question"]},
        ))
    return out


def make_program() -> dspy.Module:
    return dspy.ChainOfThought(GroundedAnswer)


def generate_all(examples: list[Example], program: dspy.Module) -> list[Example]:
    out: list[Example] = []
    for ex in examples:
        try:
            pred = program(**(ex.input or {}))
            text = pred.answer
        except Exception as exc:  # noqa: BLE001
            text = f"__GENERATION_ERROR__:{type(exc).__name__}:{exc}"
        out.append(Example(
            example_id=ex.example_id, text=text, intent=ex.intent, input=ex.input,
        ))
    return out
